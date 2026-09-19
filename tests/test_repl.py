"""RAG-backed --repl protocol tests (network-free, tmp index).

The REPL is a persistent JSONL session: one JSON request per stdin line, one JSON
response per line, and it must stay alive across malformed requests. These tests
drive the handler logic directly rather than spawning a subprocess.
"""
from __future__ import annotations

import json
from pathlib import Path

import msgspec
import pytest

from rag.cli.repl_cmd import _build_page_index, _attach_context, _read_page
from rag.corpus.schema import make_chunk_uid
from rag.index.build import ChunkRow


def make_rows():
    """A page with 3 chunks (for neighbour context) + an unrelated page."""
    rows = []
    for i, text in enumerate([
        "Controls the position and velocity of a GameObject through physics.",
        "public Vector3 velocity; The linear velocity of the rigidbody.",
        "public float mass; The mass of the rigidbody in kilograms.",
    ]):
        rows.append(ChunkRow(
            chunk_uid=make_chunk_uid("ScriptReference/Rigidbody.html", i),
            source="ScriptReference/Rigidbody.html", title="Rigidbody",
            heading_path=["Rigidbody"], text=text))
    rows.append(ChunkRow(
        chunk_uid=make_chunk_uid("Manual/Other.html", 0),
        source="Manual/Other.html", title="Other", heading_path=["Other"],
        text="Unrelated documentation about audio mixer groups."))
    # filler so BM25 finalize() does not prune the query terms at df/N >= 0.5
    for i, text in enumerate([
        "Textures and materials for terrain surfaces and layer blending.",
        "Animation state machine transitions, blend trees and root motion.",
        "UI canvas scalers, anchors and layout groups for resolutions.",
        "Shader variant stripping, keyword multi compile and warmup.",
    ]):
        rows.append(ChunkRow(
            chunk_uid=make_chunk_uid(f"Manual/F{i}.html", 0),
            source=f"Manual/F{i}.html", title=f"F{i}", heading_path=[f"F{i}"],
            text=text))
    return rows


@pytest.fixture()
def engine(tmp_path):
    from rag.index.bm25_index import build_bm25
    from rag.search.engine import SearchEngine

    rows = make_rows()
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "chunks.msgpack").write_bytes(
        msgspec.msgpack.encode([msgspec.to_builtins(r) for r in rows]))
    chunks = [r.to_corpus_chunk() for r in rows]
    index, _s = build_bm25(chunks, [r.source for r in rows],
                           [r.title for r in rows], verbose=False)
    index.save(str(index_dir / "bm25_word.pkl"))
    eng = SearchEngine({"index_dir": str(index_dir), "min_should_match": 0.0,
                        "embed_model": "none", "mode": "bm25"})
    eng.load()
    return eng


def test_page_index_groups_by_source(engine):
    pages = _build_page_index(engine)
    rb = pages["ScriptReference/Rigidbody.html"]
    assert len(rb) == 3
    # document order preserved (chunk_uid is derived from source#index)
    assert [r.chunk_uid for r in rb] == [
        make_chunk_uid("ScriptReference/Rigidbody.html", i) for i in range(3)]


def test_attach_context_neighbours(engine):
    pages = _build_page_index(engine)
    hit = {"source": "ScriptReference/Rigidbody.html",
           "chunk_uid": make_chunk_uid("ScriptReference/Rigidbody.html", 1)}
    ctx = _attach_context(pages, hit, 1, 500)
    assert len(ctx) == 2
    assert {c["chunk_index"] for c in ctx} == {0, 2}
    assert all(c["context_of"] == hit["chunk_uid"] for c in ctx)
    assert all(c["text"] for c in ctx)


def test_attach_context_clamps_at_page_edges(engine):
    pages = _build_page_index(engine)
    first = {"source": "ScriptReference/Rigidbody.html",
             "chunk_uid": make_chunk_uid("ScriptReference/Rigidbody.html", 0)}
    ctx = _attach_context(pages, first, 2, 500)
    assert [c["chunk_index"] for c in ctx] == [1, 2]


def test_attach_context_single_chunk_page_returns_empty(engine):
    pages = _build_page_index(engine)
    hit = {"source": "Manual/Other.html",
           "chunk_uid": make_chunk_uid("Manual/Other.html", 0)}
    assert _attach_context(pages, hit, 2, 500) == []


def test_attach_context_unknown_chunk(engine):
    pages = _build_page_index(engine)
    hit = {"source": "ScriptReference/Rigidbody.html", "chunk_uid": "bogus"}
    assert _attach_context(pages, hit, 1, 500) == []


def test_read_page_missing_file_reports_error(tmp_path, monkeypatch):
    import rag.cli.repl_cmd as rc
    monkeypatch.setattr(rc, "resolve_path", lambda p: tmp_path / p)
    out = _read_page("ScriptReference/Nope.html", None)
    assert out["source"] == "ScriptReference/Nope.html"
    assert "error" in out and "not found" in out["error"]


def test_read_page_truncates(tmp_path, monkeypatch):
    import rag.cli.repl_cmd as rc
    monkeypatch.setattr(rc, "resolve_path", lambda p: tmp_path / p)
    f = tmp_path / "page.html"
    f.write_text("<html><body><div id='content-wrap'>"
                 "<p>" + ("word " * 500) + "</p></div></body></html>",
                 encoding="utf-8")
    out = _read_page("page.html", 200)
    assert out["truncated"] is True
    assert len(out["markdown"]) == 200
    assert out["chars"] > 200
    full = _read_page("page.html", None)
    assert "truncated" not in full


def test_repl_session_handles_bad_requests(engine, monkeypatch, capsys):
    """The session must survive malformed input and keep serving."""
    import io

    import rag.cli.repl_cmd as rc

    monkeypatch.setattr(rc, "load_rag_config", lambda *a, **k: engine.cfg)
    monkeypatch.setattr("rag.search.engine.SearchEngine", lambda cfg: engine)

    requests = "\n".join([
        "not json at all",
        "{}",
        json.dumps({"query": "rigidbody velocity", "k": 2}),
        "",
        "# a comment line",
        json.dumps({"mentions": "rigidbody", "limit": 2}),
    ])
    monkeypatch.setattr("sys.stdin", io.StringIO(requests))

    rc.repl_loop()
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 4, lines  # bad json, {}, query, mentions

    err1, err2, q, m = (json.loads(ln) for ln in lines)
    assert "invalid JSON" in err1["error"]
    assert "must contain one of" in err2["error"]
    assert q["hits"] and q["_meta"]["engine"] == "rag.search"
    assert "rigidbody" in m and m["rigidbody"]


def test_repl_query_supports_context_and_dump(engine, monkeypatch, capsys,
                                               tmp_path):
    import io

    import rag.cli.repl_cmd as rc

    monkeypatch.setattr(rc, "load_rag_config", lambda *a, **k: engine.cfg)
    monkeypatch.setattr("rag.search.engine.SearchEngine", lambda cfg: engine)
    # dump reads the page markdown via dumpdoc; stub it so no HTML file is needed
    monkeypatch.setattr(rc, "_read_page",
                        lambda src, mc: {"markdown": f"# {src} stubbed page"})

    req = json.dumps({"query": "rigidbody velocity", "k": 1, "context": 1,
                      "dump": 1})
    monkeypatch.setattr("sys.stdin", io.StringIO(req))
    rc.repl_loop()
    out = json.loads(capsys.readouterr().out.strip())
    hit = out["hits"][0]
    assert "context" in hit and isinstance(hit["context"], list)
    assert hit["page_markdown"].startswith("# ")


def test_repl_mentions_accepts_dirs_filter(engine, monkeypatch, capsys):
    import io

    import rag.cli.repl_cmd as rc

    monkeypatch.setattr(rc, "load_rag_config", lambda *a, **k: engine.cfg)
    monkeypatch.setattr("rag.search.engine.SearchEngine", lambda cfg: engine)
    req = json.dumps({"mentions": "rigidbody", "limit": 10,
                      "dirs": ["ScriptReference"]})
    monkeypatch.setattr("sys.stdin", io.StringIO(req))
    rc.repl_loop()
    out = json.loads(capsys.readouterr().out.strip())
    srcs = [r["source"] for r in out["rigidbody"]]
    assert srcs, "expected at least one ScriptReference match"
    assert all(s.startswith("ScriptReference/") for s in srcs), srcs
