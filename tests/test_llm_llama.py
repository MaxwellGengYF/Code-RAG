"""Local llama.cpp provider tests.

External mode is exercised with httpx MockTransport (no socket traffic at all).
Managed mode is exercised against a tiny fake ``llama-server`` script run with
sys.executable on 127.0.0.1 — it speaks just enough HTTP (/health +
/v1/chat/completions) to prove spawn / health-wait / generate / teardown without
needing a real GGUF model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

from rag.llm import create_llm
from rag.llm.base import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    LLMError,
    RateLimitError,
)
from rag.llm.config import ProviderConfig

FAKE_SERVER = r'''
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200); self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        if self.path == "/v1/chat/completions":
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            assert req["stream"] is False
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "id": "chatcmpl-fake", "object": "chat.completion", "created": 0,
                "model": req["model"],
                "choices": [{"index": 0, "message": {"role": "assistant",
                             "content": "fake-local-reply"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3,
                          "total_tokens": 10},
            }).encode())
        else:
            self.send_response(404); self.end_headers()
HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def make_transport(capture: dict, status: int = 200,
                   raise_exc: Exception | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if raise_exc is not None:
            raise raise_exc
        capture["body"] = json.loads(request.content)
        capture["url"] = str(request.url)
        if status != 200:
            return httpx.Response(status, text="boom")
        return httpx.Response(200, json={
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "message": {
                "role": "assistant", "content": "hello world response",
                "reasoning_content": "hmm"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "total_tokens": 15},
        })

    return httpx.MockTransport(handler)


def make_client(capture: dict, cfg_dict: dict, **kwargs):
    cfg = ProviderConfig.from_dict(cfg_dict)
    return create_llm(cfg, http_client=httpx.AsyncClient(
        transport=make_transport(capture)), **kwargs)


# ---------------------------------------------------------------------------
# external mode (base_url set): pure wire-format checks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llama_external_toolfree_payload():
    capture: dict = {}
    client = make_client(capture, {
        "model": "qwen3.5-9b-q4_k_m.gguf", "type": "llama",
        "base_url": "http://127.0.0.1:9999",
    })
    result = await client.generate("You are a corpus builder.", "PAGE MARKDOWN HERE")
    assert result.text == "hello world response"
    assert result.thinking == "hmm"  # reasoning_content surfaced when split out
    assert result.input_tokens == 10 and result.output_tokens == 5
    assert capture["url"] == "http://127.0.0.1:9999/v1/chat/completions"
    body = capture["body"]
    assert "tools" not in body and "tool_choice" not in body
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["stream"] is False


@pytest.mark.asyncio
async def test_llama_external_no_system_prompt_omitted():
    capture: dict = {}
    client = make_client(capture, {
        "model": "m.gguf", "type": "llama",
        "base_url": "http://127.0.0.1:9999",
    })
    await client.generate("", "USER ONLY")
    assert [m["role"] for m in capture["body"]["messages"]] == ["user"]


@pytest.mark.asyncio
async def test_llama_external_max_tokens_and_extra_body():
    capture: dict = {}
    client = make_client(capture, {
        "model": "m.gguf", "type": "llama",
        "base_url": "http://127.0.0.1:9999", "max_tokens": 999999999,
        "extra_body": {"temperature": 0.2, "chat_template_kwargs": {"enable_thinking": False}},
    })
    await client.generate("s", "u")
    body = capture["body"]
    assert body["max_tokens"] == 384_000  # clamped to the safe bound
    assert body["temperature"] == 0.2
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_llama_error_mapping():
    cfg_dict = {"model": "m.gguf", "type": "llama",
                "base_url": "http://127.0.0.1:9999"}

    cfg = ProviderConfig.from_dict(cfg_dict)

    async def once(transport: httpx.MockTransport):
        return await create_llm(
            cfg, http_client=httpx.AsyncClient(transport=transport),
        ).generate("s", "u")

    with pytest.raises(RateLimitError):
        await once(make_transport({}, status=429))
    with pytest.raises(APIStatusError) as ei:
        await once(make_transport({}, status=500))
    assert ei.value.status_code == 500
    with pytest.raises(APITimeoutError):
        await once(make_transport({}, raise_exc=httpx.ReadTimeout("slow")))
    with pytest.raises(APIConnectionError):
        await once(make_transport({}, raise_exc=httpx.ConnectError("refused")))


# ---------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------

def test_llama_type_aliases():
    for alias in ("llama", "llama_cpp", "llama.cpp", "llama-server"):
        cfg = ProviderConfig.from_dict({"model": "m.gguf", "type": alias})
        assert cfg.type == "llama"


def test_llama_config_extras_kept_in_raw():
    cfg = ProviderConfig.from_dict({
        "model": "m.gguf", "type": "llama",
        "server_bin": "/opt/llama-server", "model_path": "/models/m.gguf",
        "port": 18121, "ngl": 42, "ctx_size": 4096,
        "extra_args": ["--reasoning-format", "none"],
    })
    assert cfg.raw["ngl"] == 42
    assert cfg.raw["extra_args"] == ["--reasoning-format", "none"]


# ---------------------------------------------------------------------------
# managed mode (no base_url): spawn a fake llama-server with sys.executable
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_server(tmp_path: Path) -> Path:
    script = tmp_path / "fake_llama_server.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    return script


@pytest.mark.asyncio
async def test_llama_managed_roundtrip(fake_server: Path, tmp_path: Path):
    model = tmp_path / "dummy.gguf"
    model.write_bytes(b"dummy")  # only existence is checked pre-spawn
    cfg = ProviderConfig.from_dict({
        "model": "qwen3.5-9b-q4_k_m.gguf", "type": "llama",
        # python.exe as a stand-in for llama-server.exe: a fully custom argv via
        # server_cmd so the fake script (not python's own -m) parses the flags.
        "server_cmd": [sys.executable, str(fake_server), "--host", "{host}",
                       "--port", "{port}"],
        "model_path": str(model), "start_timeout": 60,
    })
    client = create_llm(cfg)
    assert client.model_name == "qwen3.5-9b-q4_k_m.gguf"
    try:
        result = await client.generate("sys", "user")
        assert result.text == "fake-local-reply"
        assert result.input_tokens == 7 and result.output_tokens == 3
        assert client.base_url.startswith("http://127.0.0.1:")
    finally:
        await client.aclose()
    # the managed subprocess must be gone after aclose
    assert client._proc is None


@pytest.mark.asyncio
async def test_llama_managed_missing_binary_raises():
    cfg = ProviderConfig.from_dict({
        "model": "m.gguf", "type": "llama", "server_bin": "no-such-llama-server-xyz",
    })
    client = create_llm(cfg)
    with pytest.raises(LLMError, match="not found"):
        await client.generate("s", "u")


@pytest.mark.asyncio
async def test_llama_managed_missing_model_raises(tmp_path: Path):
    cfg = ProviderConfig.from_dict({
        "model": "ghost.gguf", "type": "llama", "server_bin": sys.executable,
        "model_path": str(tmp_path / "ghost.gguf"),
    })
    with pytest.raises(LLMError, match="GGUF model not found"):
        await create_llm(cfg).generate("s", "u")
