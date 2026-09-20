"""Local llama.cpp provider: llama-server subprocess + OpenAI-compatible HTTP.

Two modes, selected by the presence of ``base_url`` in the provider config:

* managed (no ``base_url``): a ``llama-server`` subprocess is spawned lazily on
  the first :meth:`LlamaClient.generate` call with the configured GGUF model,
  kept loaded across calls, and torn down on :meth:`aclose` / process exit.
  This is the "local model" path — nothing to install beyond the llama.cpp
  binaries, and the model loads once no matter how many pages are compiled.
* external (``base_url`` set): only talks HTTP to an already-running
  llama-server; useful when the server is managed by something else.

Kept: single-turn system+user chat, token usage, ``reasoning_content`` when the
server splits reasoning out (``--reasoning-format openai``), non-2xx/timeout
mapping onto the vendored error hierarchy so the compile pipeline's retry /
circuit-breaker machinery keeps working unchanged.
Stripped: tools, streaming, multi-turn history (the RAG compile step needs
single turns only, same contract as the other vendored providers).

Config extras (read from the provider-config JSON, i.e. ``ProviderConfig.raw``)::

    {
      "type": "llama",
      "model": "Qwen3.5-9B-Q4_K_M.gguf",   # managed: GGUF path (or model_path);
                                            # external: served model name
      "server_bin": "D:/unity_manual/llama_cpp/llama-server.exe",
      "server_cmd": null,         # optional full argv template; replaces the
                                   # default -m/-host/-port/-ngl/-c invocation,
                                   # "{model}" / "{host}" / "{port}" substituted
      "model_path": "D:/unity_manual/models/Qwen3.5-9B/Qwen3.5-9B-Q4_K_M.gguf",
      "host": "127.0.0.1",
      "port": 0,               # 0 = pick a free port (managed mode)
      "ngl": 99,               # GPU layers to offload
      "ctx_size": 8192,
      "extra_args": ["--mlock"],   # extra llama-server CLI flags
      "extra_body": {},        # merged into every chat.completions request,
                               # e.g. {"chat_template_kwargs": {"enable_thinking": false}}
                               # to skip Qwen3.5's thinking phase
      "start_timeout": 600     # seconds to wait for the model to load
    }
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import shutil
import socket
import subprocess
import tempfile
import weakref
from pathlib import Path
from typing import Any

import httpx

from ._openai_shared import clamp_max_tokens
from .base import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    GenerationResult,
    LLMError,
    RateLimitError,
)
from .config import ProviderConfig

log = logging.getLogger(__name__)

#: subprocess spawn/kill is wrapped so interpreter shutdown (atexit/finalize
#: during a crashed compile run) can never raise and mask the real error.
_LIVE_PROCS: set[subprocess.Popen] = set()


def _reap_live_procs() -> None:
    for proc in list(_LIVE_PROCS):
        _kill_proc(proc)


def _kill_proc(proc: Any) -> None:
    """Best-effort kill that never raises (safe from atexit / weakref.finalize)."""
    try:
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass  # already dead or loop gone — nothing useful left to do


def _popen_of(proc: Any) -> Any:
    """The real subprocess.Popen behind an asyncio.subprocess.Process.

    Both transport flavours keep it as ``_proc``: the transport itself on
    Windows (ProactorEventLoop) and on Unix (SelectorEventLoop), never on the
    Process wrapper — so look through the transport and fall back to the
    wrapper (whose kill() is still safe to call).
    """
    popen = getattr(proc, "_proc", None)
    if popen is None:
        popen = getattr(getattr(proc, "_transport", None), "_proc", None)
    return popen if popen is not None else proc


atexit.register(_reap_live_procs)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _convert_httpx_error(exc: BaseException) -> BaseException:
    if isinstance(exc, httpx.TimeoutException):
        return APITimeoutError(str(exc))
    if isinstance(exc, httpx.HTTPError):
        return APIConnectionError(str(exc))
    return exc


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


class LlamaClient:
    """Single-turn chat client over a local llama-server (managed or external)."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int = 0,  # noqa: ARG002 - compile pipeline retries upstream
        **client_kwargs: Any,
    ):
        self._config = config
        raw = config.raw
        # base_url set -> external server; absent -> we spawn and own one.
        self._managed = config.base_url is None
        self._host = str(raw.get("host") or "127.0.0.1")
        self._port = int(raw.get("port") or 0)
        self._server_bin = str(raw.get("server_bin") or "llama-server")
        self._server_cmd = [str(a) for a in raw.get("server_cmd") or []]
        self._model_path = str(raw.get("model_path") or config.model)
        self._ngl = int(raw.get("ngl", 99))
        self._ctx_size = int(raw.get("ctx_size", 8192))
        self._extra_args = [str(a) for a in raw.get("extra_args") or []]
        self._extra_body = dict(raw.get("extra_body") or {})
        self._start_timeout = float(raw.get("start_timeout") or 600)
        timeout = client_kwargs.get("timeout", config.timeout)
        self._timeout = float(timeout) if timeout else None

        self._http = http_client
        self._own_http = http_client is None
        self._proc: Any = None  # asyncio.subprocess.Process
        self._popen: Any = None  # underlying subprocess.Popen (kill target)
        self._finalizer: weakref.finalize | None = None
        self._server_log: str | None = None
        self._start_lock = asyncio.Lock()
        self._closed = False

    # ------------------------------------------------------------------ misc

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def base_url(self) -> str:
        """The server's HTTP origin; in managed mode valid only while running."""
        if self._managed:
            return f"http://{self._host}:{self._port}"
        return (self._config.base_url or "").rstrip("/")

    # -------------------------------------------------------------- lifecycle

    async def _ensure_server(self) -> None:
        if not self._managed or (self._proc is not None and self._proc.returncode is None):
            return
        async with self._start_lock:
            if self._proc is not None and self._proc.returncode is None:
                return
            await self._start_server()

    async def _start_server(self) -> None:
        if not Path(self._model_path).is_file():
            raise LLMError(f"GGUF model not found: {self._model_path}")

        if self._port == 0:
            self._port = _find_free_port(self._host)

        if self._server_cmd:
            # Fully custom argv (wrappers, docker, test doubles); placeholders
            # keep it declarative without string-splitting semantics.
            args = [
                a.format(model=self._model_path, host=self._host, port=self._port)
                for a in self._server_cmd
            ]
        else:
            binary = shutil.which(self._server_bin) or (
                self._server_bin if Path(self._server_bin).is_file() else None
            )
            if binary is None:
                raise LLMError(
                    f"llama-server binary not found: {self._server_bin!r} "
                    "(set 'server_bin' in the provider config)"
                )
            args = [
                binary,
                "-m", self._model_path,
                "--host", self._host,
                "--port", str(self._port),
                "-ngl", str(self._ngl),
                "-c", str(self._ctx_size),
                "--jinja",
                *self._extra_args,
            ]
        log_file = tempfile.NamedTemporaryFile(
            mode="w", prefix="llama-server-", suffix=".log", delete=False
        )
        self._server_log = log_file.name
        log.info("starting llama-server: %s", " ".join(args))
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=_NO_WINDOW,
            )
        except OSError as exc:
            log_file.close()
            raise LLMError(f"failed to start llama-server: {exc}") from exc
        log_file.close()  # child inherited its own copy of the handle
        self._proc = proc
        # Proc survives the client object; both atexit and the finalizer try to
        # kill it, and both swallow "already dead" — belt and suspenders so a
        # crashed compile run never leaves a GPU-resident server behind.
        popen = _popen_of(proc)
        self._popen = popen
        _LIVE_PROCS.add(popen)
        self._finalizer = weakref.finalize(self, _kill_proc, popen)

        # Poll /health until the model is loaded (llama-server 503s while loading).
        deadline = asyncio.get_running_loop().time() + self._start_timeout
        url = f"{self.base_url}/health"
        client = self._client()
        while True:
            if proc.returncode is not None:
                raise LLMError(
                    f"llama-server exited with code {proc.returncode} during startup; "
                    f"see log: {self._server_log}"
                )
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    break
            except httpx.HTTPError:
                pass  # not listening yet
            if asyncio.get_running_loop().time() >= deadline:
                raise LLMError(
                    f"llama-server did not become healthy within {self._start_timeout}s; "
                    f"see log: {self._server_log}"
                )
            await asyncio.sleep(0.5)
        log.info("llama-server ready at %s", self.base_url)

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=self._timeout or httpx.Timeout(600.0)
            )
        return self._http

    async def aclose(self) -> None:
        """Shut down a managed server. External-mode clients just close HTTP."""
        self._closed = True
        if self._finalizer is not None:
            self._finalizer()  # kills the proc if still running
            self._finalizer = None
        proc, self._proc = self._proc, None
        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        if self._popen is not None:
            _LIVE_PROCS.discard(self._popen)
            self._popen = None
        if self._own_http and self._http is not None:
            try:
                await self._http.aclose()
            except Exception:
                pass

    # ------------------------------------------------------------- generation

    async def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        if self._closed:
            raise LLMError("LlamaClient is closed")
        await self._ensure_server()

        cfg = self._config
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "stream": False,
        }
        if cfg.max_tokens:
            body["max_tokens"] = cfg.max_tokens
            clamp_max_tokens(body)
        body.update(self._extra_body)

        url = f"{self.base_url}/v1/chat/completions"
        try:
            resp = await self._client().post(url, json=body)
        except Exception as exc:
            raise _convert_httpx_error(exc) from exc

        if resp.status_code != 200:
            if resp.status_code == 429:
                raise RateLimitError(resp.text)
            raise APIStatusError(
                f"llama-server returned {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise APIStatusError(f"invalid JSON from llama-server: {exc!r}") from exc

        try:
            msg = data["choices"][0]["message"]
            usage = data.get("usage") or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise APIStatusError(f"malformed chat.completions response: {data!r}") from exc
        thinking = msg.get("reasoning_content")
        return GenerationResult(
            text=msg.get("content") or "",
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            thinking=thinking if isinstance(thinking, str) and thinking else None,
        )
