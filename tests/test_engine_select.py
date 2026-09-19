"""Engine auto-selection + manifest audit tests (network-free, tmp dirs)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.search.engine_select import (
    COMPLETENESS_FRACTION,
    choose_engine,
    count_html_pages,
    rag_index_status,
)
from rag.store import CorpusStore, FileManager


@pytest.fixture()
def mirror(tmp_path):
    """A 20-page mirror + a rag index dir we can populate to any coverage."""
    root = tmp_path / "site"
    (root / "Manual").mkdir(parents=True)
    (root / "ScriptReference").mkdir(parents=True)
    for i in range(10):
        (root / "Manual" / f"m{i}.html").write_text("<html></html>", encoding="utf-8")
        (root / "ScriptReference" / f"s{i}.html").write_text("<html></html>",
                                                             encoding="utf-8")
    index_dir = root / "index"
    corpus_dir = root / "corpus"
    index_dir.mkdir()
    corpus_dir.mkdir()
    return root, index_dir, corpus_dir


def write_index(index_dir: Path, *, n_pages: int, n_chunks: int,
                dense: bool = False):
    (index_dir / "manifest.json").write_text(json.dumps(
        {"n_chunks": n_chunks, "n_pages": n_pages,
         "corpus_gen_key": "abc", "embed_model": "BAAI/bge-m3"}), encoding="utf-8")
    (index_dir / "chunks.msgpack").write_bytes(b"x")
    (index_dir / "bm25_word.pkl").write_bytes(b"x")
    if dense:
        (index_dir / "vectors.f32").write_bytes(b"x")


def cfg_for(root: Path, index_dir: Path, corpus_dir: Path) -> dict:
    return {"index_dir": str(index_dir), "corpus_dir": str(corpus_dir),
            "dirs": [str(root / "Manual"), str(root / "ScriptReference")]}


def test_count_html_pages(mirror):
    root, index_dir, corpus_dir = mirror
    cfg = cfg_for(root, index_dir, corpus_dir)
    assert count_html_pages(cfg["dirs"]) == 20


def test_no_index_is_not_complete(mirror):
    root, index_dir, corpus_dir = mirror
    cfg = cfg_for(root, index_dir, corpus_dir)
    st = rag_index_status(cfg)
    assert st["built"] is False and st["complete"] is False
    assert "not built" in st["reason"]
    engine, chosen = choose_engine(cfg)
    assert engine == "legacy"
    # choose_engine enriches the status with legacy availability; the RAG fields
    # must be carried through unchanged
    for key in ("built", "complete", "coverage", "reason"):
        assert chosen[key] == st[key]
    assert "legacy" in chosen and "nothing_built" in chosen


def test_partial_index_selects_legacy(mirror):
    root, index_dir, corpus_dir = mirror
    cfg = cfg_for(root, index_dir, corpus_dir)
    write_index(index_dir, n_pages=8, n_chunks=40)  # 8/20 = 40%
    st = rag_index_status(cfg)
    assert st["built"] is True
    assert st["coverage"] == pytest.approx(0.4)
    assert st["complete"] is False
    assert choose_engine(cfg)[0] == "legacy"
    assert "40.0%" in st["reason"]


def test_complete_index_selects_rag(mirror):
    root, index_dir, corpus_dir = mirror
    cfg = cfg_for(root, index_dir, corpus_dir)
    n = int(20 * COMPLETENESS_FRACTION) + 1  # 20 of 20
    write_index(index_dir, n_pages=n, n_chunks=n * 2, dense=True)
    st = rag_index_status(cfg)
    assert st["complete"] is True
    assert st["has_dense"] is True
    assert choose_engine(cfg)[0] == "rag"


def test_zero_chunk_index_is_not_complete(mirror):
    root, index_dir, corpus_dir = mirror
    cfg = cfg_for(root, index_dir, corpus_dir)
    write_index(index_dir, n_pages=20, n_chunks=0)
    st = rag_index_status(cfg)
    assert st["complete"] is False
    assert st["reason"] == "index has 0 chunks"


def test_forced_preference_overrides(mirror):
    root, index_dir, corpus_dir = mirror
    cfg = cfg_for(root, index_dir, corpus_dir)
    write_index(index_dir, n_pages=1, n_chunks=2)  # clearly partial
    assert choose_engine(cfg, prefer="rag")[0] == "rag"
    assert choose_engine(cfg, prefer="legacy")[0] == "legacy"


def test_nothing_built_when_both_engines_absent(mirror, monkeypatch):
    """Fresh checkout: neither engine built. Must not claim LEGACY would serve."""
    import rag.search.engine_select as es

    root, index_dir, corpus_dir = mirror
    cfg = cfg_for(root, index_dir, corpus_dir)
    monkeypatch.setattr(es, "legacy_index_status",
                        lambda: {"built": False,
                                 "missing": ["chunks.pkl", "index_word.pkl"]})
    engine, st = es.choose_engine(cfg)
    assert engine == "legacy"  # caller's not-found handler prints the rebuild hint
    assert st["nothing_built"] is True
    assert st["legacy"]["missing"] == ["chunks.pkl", "index_word.pkl"]


def test_not_nothing_built_when_legacy_present(mirror, monkeypatch):
    """Partial RAG index + full legacy artefacts: legacy genuinely serves."""
    import rag.search.engine_select as es

    root, index_dir, corpus_dir = mirror
    cfg = cfg_for(root, index_dir, corpus_dir)
    write_index(index_dir, n_pages=8, n_chunks=40)
    monkeypatch.setattr(es, "legacy_index_status",
                        lambda: {"built": True, "missing": []})
    engine, st = es.choose_engine(cfg)
    assert engine == "legacy"
    assert st["nothing_built"] is False


def test_legacy_index_status_reports_missing(monkeypatch, tmp_path):
    """legacy_index_status resolves against the repo root; check its shape."""
    import rag.search.engine_select as es

    monkeypatch.setattr(es, "resolve_path", lambda p: tmp_path / p)
    st = es.legacy_index_status()
    assert st["built"] is False
    assert set(st["missing"]) == {"chunks.pkl", "index_word.pkl"}
    (tmp_path / "chunks.pkl").write_bytes(b"x")
    (tmp_path / "index_word.pkl").write_bytes(b"x")
    st = es.legacy_index_status()
    assert st["built"] is True and st["missing"] == []


def test_corrupt_index_manifest_handled(mirror):
    root, index_dir, corpus_dir = mirror
    cfg = cfg_for(root, index_dir, corpus_dir)
    (index_dir / "manifest.json").write_text("not json {{{", encoding="utf-8")
    st = rag_index_status(cfg)
    assert st["built"] is False
    assert "unreadable" in st["reason"]
    assert choose_engine(cfg)[0] == "legacy"


# --------------------------------------------------------------------------------------
# manifest audit
# --------------------------------------------------------------------------------------


def test_audit_drops_phantom_entries(tmp_path):
    """Manifest claiming pages with no corpus file gets repaired."""
    root = tmp_path / "site"
    (root / "ScriptReference").mkdir(parents=True)
    for i in range(5):
        (root / "ScriptReference" / f"p{i}.html").write_text("<html></html>",
                                                             encoding="utf-8")
    corpus_dir = tmp_path / "corpus"
    fm = FileManager(root=root, dirs=["ScriptReference"], corpus_dir=corpus_dir)
    store = CorpusStore(corpus_dir)
    scanned = fm.scan()

    # only 2 corpus files exist, but the manifest claims all 5 (the old bug)
    for rel in list(scanned)[:2]:
        store.save(rel, {"source": rel, "chunks": []})
    fm.save_manifest(scanned, "gk", {"model": "m"},
                     page_gen_keys={r: "k" for r in scanned},
                     needs_regen=[])
    assert len(fm.load_manifest()["files"]) == 5

    res = fm.audit_manifest(verbose=False)
    assert res["files"] == 2
    assert res["dropped"] == 3
    m = fm.load_manifest()
    assert len(m["files"]) == 2
    assert len(m["page_gen_keys"]) == 2
    # the phantom pages now correctly show as added on the next diff
    assert len(fm.diff(fm.scan()).added) == 3


def test_audit_preserves_needs_regen_flags(tmp_path):
    """Audit must not drop the retry queue: losing needs_regen would silently
    abandon every heuristic-fallback page to aux-less chunks forever."""
    root = tmp_path / "site"
    (root / "ScriptReference").mkdir(parents=True)
    for i in range(4):
        (root / "ScriptReference" / f"p{i}.html").write_text("<html></html>",
                                                             encoding="utf-8")
    corpus_dir = tmp_path / "corpus"
    fm = FileManager(root=root, dirs=["ScriptReference"], corpus_dir=corpus_dir)
    store = CorpusStore(corpus_dir)
    scanned = fm.scan()
    rels = sorted(scanned)
    for r in rels[:2]:
        store.save(r, {"source": r, "chunks": []})
    # one flag on a page WITH a corpus file, one on a phantom page
    fm.save_manifest(scanned, "gk", {}, page_gen_keys={r: "k" for r in rels},
                     needs_regen=[rels[0], rels[3]])
    res = fm.audit_manifest(verbose=False)
    assert res["dropped"] == 2
    m = fm.load_manifest()
    assert rels[0] in m["needs_regen"], "existing flagged page must survive audit"
    # a flag on a dropped phantom page is harmless: the page requeues via `added`
    # anyway, and keeping it costs nothing
    assert len(m["files"]) == 2


def test_audit_noop_when_consistent(tmp_path):
    root = tmp_path / "site"
    (root / "ScriptReference").mkdir(parents=True)
    (root / "ScriptReference" / "p0.html").write_text("<html></html>", encoding="utf-8")
    corpus_dir = tmp_path / "corpus"
    fm = FileManager(root=root, dirs=["ScriptReference"], corpus_dir=corpus_dir)
    store = CorpusStore(corpus_dir)
    scanned = fm.scan()
    store.save("ScriptReference/p0.html", {"source": "ScriptReference/p0.html",
                                           "chunks": []})
    fm.save_manifest(scanned, "gk", {}, page_gen_keys={}, needs_regen=[])
    res = fm.audit_manifest(verbose=False)
    assert res["dropped"] == 0
    assert len(fm.load_manifest()["files"]) == 1
