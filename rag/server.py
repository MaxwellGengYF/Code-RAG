"""HTTP server mode: ``uv run python -m rag.server`` — the search engine behind
a loopback JSON API, so a coding-agent client can query a warm index without
paying the engine-load cost per invocation.

Stdlib only (http.server + urllib). Endpoints:
    GET  /health  -> {"ok": true, "chunks": N, "dense": b, "embed": b, "model": s}
    POST /search   -> engine.search dict + injected resp["_meta"]
    POST /mentions -> {term: engine.mentions(...)}
    POST /read     -> full-page markdown (_read_page)

The socket binds FIRST and /health answers 503 {"starting": true} during
warmup, so clients can poll for readiness while the index loads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 8642


class ServerUnavailable(Exception):
    """The rag.server HTTP API is not reachable (down, starting, or timeout)."""


# ---------------------------------------------------------------------- config

def resolve_port(cfg=None, override=None) -> int:
    """--port flag > env RAG_SERVER_PORT > config server_port > default 8642."""
    if override is not None:
        return int(override)
    env = os.environ.get("RAG_SERVER_PORT")
    if env:
        return int(env)
    if cfg is not None:
        try:
            port = cfg.get("server_port")
        except AttributeError:
            port = None
        if port:
            return int(port)
    return DEFAULT_PORT


# ------------------------------------------------------------------ http server

class _State:
    """Shared server state: engine + readiness, guarded by one lock."""

    def __init__(self, cfg, engine, base):
        self.cfg = cfg
        self.engine = engine
        self.base = base
        self.ready = threading.Event()
        self.embed_ready = False
        self.lock = threading.Lock()


def _make_handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence per-request logging
            pass

        # ---------------------------------------------------------- plumbing
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._health()
            else:
                self._send(404, {"error": f"unknown path: {self.path}"})

        def do_POST(self) -> None:
            routes = {"/search": self._search, "/mentions": self._mentions,
                      "/read": self._read}
            fn = routes.get(self.path)
            if fn is None:
                self._send(404, {"error": f"unknown path: {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length > 0 else b""
            except Exception:
                raw = b""
            try:
                req = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send(400, {"error": f"invalid JSON body: {exc}"})
                return
            if not isinstance(req, dict):
                self._send(400, {"error": "request body must be a JSON object"})
                return
            try:
                fn(req)
            except (KeyError, ValueError, TypeError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        # ---------------------------------------------------------- endpoints
        def _health(self) -> None:
            if not state.ready.is_set():
                self._send(503, {"ok": False, "starting": True})
                return
            self._send(200, {"ok": True, "chunks": state.engine.n_chunks,
                             "dense": state.engine.has_dense,
                             "embed": state.embed_ready,
                             "model": state.engine.embed_model})

        def _search(self, req: dict) -> None:
            if not state.ready.is_set():
                self._send(503, {"error": "server starting"})
                return
            query = req["query"]
            k = int(req.get("k", state.cfg.get("final_k", 8)))
            mode = req.get("mode", state.cfg.get("mode", "hybrid"))
            kwargs = {}
            for key in ("per_file", "explain", "no_rerank", "snippet_width"):
                if key in req:
                    kwargs[key] = req[key]
            with state.lock:
                resp = state.engine.search(query, k=k, mode=mode, **kwargs)
            resp["_meta"] = {"engine": "rag.server",
                             "chunks": state.engine.n_chunks,
                             "dense": state.engine.has_dense,
                             "embed": state.embed_ready}
            self._send(200, resp)

        def _mentions(self, req: dict) -> None:
            if not state.ready.is_set():
                self._send(503, {"error": "server starting"})
                return
            term = req["term"]
            with state.lock:
                rows = state.engine.mentions(
                    term, limit=int(req.get("limit", 60)),
                    context=int(req.get("context", 0)))
            self._send(200, {term: rows})

        def _read(self, req: dict) -> None:
            if not state.ready.is_set():
                self._send(503, {"error": "server starting"})
                return
            source = req["source"]
            from rag.cli.repl_cmd import _read_page
            self._send(200, _read_page(source, req.get("max_chars"), state.base))

    return Handler


def create_server(cfg, engine, base, port) -> tuple[ThreadingHTTPServer, _State]:
    """Bind 127.0.0.1:port and return (httpd, state) — engine not yet touched.
    Tests drive httpd.serve_forever() in a thread and set state.ready themselves."""
    state = _State(cfg, engine, base)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(state))
    httpd.daemon_threads = True
    return httpd, state


# -------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag.server",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, metavar="CFG.json",
                        help="Config JSON (default: ./config.json when present)")
    parser.add_argument("--port", type=int, default=None,
                        help=f"Listen port (default: {DEFAULT_PORT}; env "
                             "RAG_SERVER_PORT; config server_port)")
    args = parser.parse_args(argv)

    from rag.config import config_dir, load_settings
    from rag.search.engine import SearchEngine
    cfg = load_settings(args.config)
    port = resolve_port(cfg, args.port)
    engine = SearchEngine(cfg)
    base = config_dir(cfg)
    try:
        httpd, state = create_server(cfg, engine, base, port)
    except OSError as exc:
        print(f"[server] cannot bind 127.0.0.1:{port}: {exc}", file=sys.stderr)
        return 1
    print(f"[server] listening 127.0.0.1:{port}; loading...", file=sys.stderr)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        engine.load()
    except FileNotFoundError as exc:
        print(f"[server] {str(exc).replace(chr(10), ' ')}", file=sys.stderr)
        httpd.shutdown()
        httpd.server_close()
        return 1
    if engine.has_dense:
        try:
            from rag.index.vector_index import ensure_embed_model
            ensure_embed_model(engine.embed_model)
            state.embed_ready = True
        except Exception as exc:
            print(f"[server] embed preload failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
    state.ready.set()
    print(f"[server] ready chunks={engine.n_chunks} dense={engine.has_dense} "
          f"embed={state.embed_ready} model={engine.embed_model}", file=sys.stderr)
    try:
        while True:
            threading.Event().wait(3600)
    except KeyboardInterrupt:
        pass
    httpd.shutdown()
    httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ------------------------------------------------------------------ http client

def server_request(cfg, path: str, payload: dict, timeout: float = 2.0) -> dict:
    """POST JSON to the rag.server loopback API and parse the JSON response.

    Raises ServerUnavailable on connection errors/timeouts and HTTP 503;
    RuntimeError carrying the server's message on other HTTP errors.
    """
    url = f"http://127.0.0.1:{resolve_port(cfg)}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            raise ServerUnavailable(f"server starting ({url})") from exc
        try:
            msg = json.loads(exc.read().decode("utf-8")).get("error", str(exc))
        except Exception:
            msg = str(exc)
        raise RuntimeError(msg) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ServerUnavailable(str(exc)) from exc
