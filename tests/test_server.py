"""rag/server.py tests: loopback HTTP API over a tiny BM25-only index (no external
network; the socket never leaves 127.0.0.1)."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import msgspec
import pytest

from rag.corpus.schema import make_chunk_uid
from rag.index.build import ChunkRow


def _make_rows():
    rows = [
        ChunkRow(chunk_uid=make_chunk_uid("ScriptReference/Rigidbody.html", 0),
                 source="ScriptReference/Rigidbody.html", title="Rigidbody",
                 heading_path=["Rigidbody"],
                 text="Controls the position and velocity of a GameObject."),
        ChunkRow(chunk_uid=make_chunk_uid("ScriptReference/Rigidbody.html", 1),
                 source="ScriptReference/Rigidbody.html", title="Rigidbody",
                 heading_path=["Rigidbody", "Properties"],
                 text="public Vector3 velocity; The velocity of the rigidbody.",
                 summary="velocity property", keywords=["velocity"],
                 synonyms=["linearVelocity"], qa=[]),
        ChunkRow(chunk_uid=make_chunk_uid("Manual/2d-physics.html", 0),
                 source="Manual/2d-physics.html", title="2D Physics",
                 heading_path=["2D Physics"],
                 text="Rigidbody 2D and collider fundamentals for 2D games."),
    ]
    fillers = ["Textures and materials for terrain surfaces.",
               "Audio mixer groups and snapshot automation curves.",
               "Animation state machine transitions and blend trees.",
               "UI canvas scalers and anchoring for multiple resolutions."]
    for i, text in enumerate(fillers):
        rows.append(ChunkRow(
            chunk_uid=make_chunk_uid(f"Manual/Filler{i}.html", 0),
            source=f"Manual/Filler{i}.html", title=f"Filler {i}",
            heading_path=[f"Filler {i}"], text=text))
    return rows


@pytest.fixture()
def tiny_server(tmp_path):
    """Real BM25 index + running server on an ephemeral loopback port.

    Yields (cfg, rows, port, state); caller sets state.ready."""
    from rag.index.bm25_index import build_bm25
    from rag.search.engine import SearchEngine
    from rag import server as srv

    rows = _make_rows()
    cfg = {"index_dir": str(tmp_path / "index"), "min_should_match": 0.6,
           "bm25_k": 200, "dense_k": 200, "rrf_k": 60, "embed_model": "none",
           "final_k": 8, "mode": "bm25", "corpus_dir": str(tmp_path / "corpus")}
    index_dir = Path(cfg["index_dir"])
    index_dir.mkdir()
    (index_dir / "chunks.msgpack").write_bytes(
        msgspec.msgpack.encode([msgspec.to_builtins(r) for r in rows]))
    chunks = [r.to_corpus_chunk() for r in rows]
    index, _s = build_bm25(chunks, [r.source for r in rows],
                           [r.title for r in rows], verbose=False)
    index.save(str(index_dir / "bm25_word.pkl"))

    (tmp_path / "ScriptReference").mkdir()
    (tmp_path / "ScriptReference" / "Rigidbody.html").write_text(
        "<html><body><div id='content-wrap'><p>"
        "Rigidbody velocity documentation" + (" word" * 100) +
        "</p></div></body></html>", encoding="utf-8")

    engine = SearchEngine(cfg)
    engine.load()
    httpd, state = srv.create_server(cfg, engine, tmp_path, 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        yield cfg, rows, port, state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _http(port, path, payload=None, method="POST"):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_health_starting_then_ready(tiny_server):
    cfg, rows, port, state = tiny_server
    status, body = _http(port, "/health", method="GET")
    assert status == 503 and body == {"ok": False, "starting": True}
    status, body = _http(port, "/search", {"query": "rigidbody velocity"})
    assert status == 503 and body["error"] == "server starting"
    state.ready.set()
    status, body = _http(port, "/health", method="GET")
    assert status == 200
    assert body["ok"] is True
    assert body["chunks"] == len(rows)
    assert body["dense"] is False
    assert body["embed"] is False
    assert body["model"] == "none"


def test_search_endpoint(tiny_server):
    _cfg, rows, port, state = tiny_server
    state.ready.set()
    status, body = _http(port, "/search",
                         {"query": "rigidbody velocity", "k": 3, "mode": "bm25"})
    assert status == 200
    assert body["hits"], "expected BM25 hits"
    assert body["_meta"]["engine"] == "rag.server"
    assert body["_meta"]["chunks"] == len(rows)
    assert body["_meta"]["dense"] is False
    hit = body["hits"][0]
    for key in ("rank", "title", "source", "fused_score", "text", "read_more"):
        assert key in hit


def test_mentions_endpoint(tiny_server):
    _cfg, _rows, port, state = tiny_server
    state.ready.set()
    status, body = _http(port, "/mentions", {"term": "velocity"})
    assert status == 200
    rows = body["velocity"]
    assert rows and rows[0]["source"] == "ScriptReference/Rigidbody.html"
    assert rows[0]["count"] >= 2


def test_read_endpoint(tiny_server, tmp_path):
    _cfg, _rows, port, state = tiny_server
    state.ready.set()
    status, body = _http(port, "/read",
                         {"source": "ScriptReference/Rigidbody.html",
                          "max_chars": None})
    assert status == 200
    assert body["source"] == "ScriptReference/Rigidbody.html"
    assert "Rigidbody velocity documentation" in body["markdown"]
    assert "truncated" not in body
    # truncation via max_chars
    _s, trunc = _http(port, "/read",
                      {"source": "ScriptReference/Rigidbody.html",
                       "max_chars": 20})
    assert trunc["truncated"] is True and len(trunc["markdown"]) == 20
    # missing file -> error payload, still 200 (matches _read_page contract)
    _s, missing = _http(port, "/read", {"source": "Nope.html"})
    assert "error" in missing


def test_bad_json_and_unknown_path(tiny_server):
    _cfg, _rows, port, state = tiny_server
    state.ready.set()
    status, _b = _http(port, "/nope", {})
    assert status == 404
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/search", data=b"{not json",
        method="POST", headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req, timeout=10)
    assert ei.value.code == 400
    status, body = _http(port, "/search", {"k": 3})  # missing "query"
    assert status == 400 and "error" in body


def test_server_request_unavailable_on_closed_port():
    import socket

    from rag import server as srv
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens there anymore
    with pytest.raises(srv.ServerUnavailable):
        srv.server_request({"server_port": port}, "/search", {"query": "x"},
                           timeout=1.0)


def test_resolve_port_order(monkeypatch):
    from rag import server as srv
    assert srv.resolve_port() == 8642
    assert srv.resolve_port({"server_port": 9000}) == 9000
    monkeypatch.setenv("RAG_SERVER_PORT", "9001")
    assert srv.resolve_port({"server_port": 9000}) == 9001
    assert srv.resolve_port({"server_port": 9000}, override=9002) == 9002
    assert srv.resolve_port(None, override=0) == 0


def test_main_missing_index_returns_1(tmp_path, capsys):
    """engine.load() FileNotFoundError -> one stderr line, exit 1 (no traceback)."""
    from rag import server as srv
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"index_dir": str(tmp_path / "nope")}),
                        encoding="utf-8")
    assert srv.main(["--config", str(cfg_path), "--port", "0"]) == 1
    err = capsys.readouterr().err
    assert "index artefacts not found" in err
    assert "listening" in err  # socket bound before the load failed
