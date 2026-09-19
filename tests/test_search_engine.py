"""Search engine + RRF fusion tests on a tiny tmp index (network-free, BM25-only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.corpus.schema import make_chunk_uid
from rag.index.fuse import linear_fuse, rrf_fuse
from rag.index.build import ChunkRow
import msgspec


def make_rows():
    rows = [
        ChunkRow(chunk_uid=make_chunk_uid("ScriptReference/Rigidbody.html", 0),
                 source="ScriptReference/Rigidbody.html", title="Rigidbody",
                 heading_path=["Rigidbody"], text="Controls the position and velocity of a GameObject."),
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
    # filler docs keep every term's df/N below finalize()'s 0.5 stop threshold
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
def tiny_index(tmp_path, monkeypatch):
    """Build a real BM25 index over tmp rows in an isolated index dir."""
    from rag.index.bm25_index import build_bm25
    rows = make_rows()
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
    return cfg, rows


def test_rrf_fuse_math():
    bm25 = [(10, 9.0), (20, 8.0), (30, 7.0)]
    dense = [(30, 0.9), (10, 0.8), (40, 0.7)]
    fused = dict(rrf_fuse(bm25, dense, k=60))
    # 10: rank0+rank1  30: rank2+rank0  20: rank1 only  40: rank2 only
    assert fused[10] > fused[30] > fused[20] > fused[40]
    assert fused[10] == pytest.approx(1 / 60 + 1 / 61)
    assert fused[30] == pytest.approx(1 / 62 + 1 / 60)
    assert fused[20] == pytest.approx(1 / 61)
    assert fused[40] == pytest.approx(1 / 62)


def test_rrf_single_list_equals_ranks():
    bm25 = [(5, 3.0), (1, 2.0), (9, 1.0)]
    fused = rrf_fuse(bm25, [], k=60)
    assert [d for d, _s in fused] == [5, 1, 9]  # order preserved, scores 1/(60+r)


def test_linear_fuse_ablation():
    bm25 = [(10, 9.0), (20, 8.0), (30, 1.0)]
    dense = [(30, 0.99), (20, 0.90), (10, 0.50)]
    fused = dict(linear_fuse(bm25, dense, alpha=0.5))
    # doc 20 is strong in BOTH lists -> wins the blend despite losing BM25
    assert fused[20] > fused[10]
    assert fused[20] > fused[30]


def test_engine_bm25_query(tiny_index):
    from rag.search.engine import SearchEngine
    cfg, rows = tiny_index
    engine = SearchEngine(cfg)
    out = engine.search("rigidbody velocity", k=5, mode="bm25")
    assert out["hits"], "expected hits"
    top_sources = [h["source"] for h in out["hits"]]
    assert "ScriptReference/Rigidbody.html" in top_sources[0]
    # aux text (synonym linearVelocity) is in the BM25 index
    hit = next(h for h in out["hits"]
               if h["source"] == "ScriptReference/Rigidbody.html")
    assert "velocity" in hit["text"].lower()


def test_engine_per_file_dedupe(tiny_index):
    from rag.search.engine import SearchEngine
    cfg, rows = tiny_index
    engine = SearchEngine(cfg)
    out = engine.search("rigidbody", k=5, mode="bm25")
    srcs = [h["source"] for h in out["hits"]]
    assert len(srcs) == len(set(srcs)) or srcs.count("ScriptReference/Rigidbody.html") == 1


def test_engine_deterministic(tiny_index):
    from rag.search.engine import SearchEngine
    cfg, rows = tiny_index
    a = SearchEngine(cfg).search("rigidbody velocity", k=5, mode="bm25")
    b = SearchEngine(cfg).search("rigidbody velocity", k=5, mode="bm25")
    assert [(h["source"], h["fused_score"]) for h in a["hits"]] == \
           [(h["source"], h["fused_score"]) for h in b["hits"]]


def test_engine_explain(tiny_index):
    from rag.search.engine import SearchEngine
    cfg, rows = tiny_index
    out = SearchEngine(cfg).search("rigidbody velocity", k=3, mode="bm25",
                                   explain=True)
    terms = {t["term"]: t["in_index"] for t in out["explain"]}
    assert terms.get("rigidbody") is True
    assert out["hits"][0]["explain_scores"]["bm25"] > 0


def test_engine_mentions(tiny_index):
    from rag.search.engine import SearchEngine
    cfg, rows = tiny_index
    res = SearchEngine(cfg).mentions("velocity")
    srcs = {r["source"]: r["count"] for r in res}
    assert srcs.get("ScriptReference/Rigidbody.html", 0) >= 2


def test_engine_missing_index_errors_cleanly(tmp_path):
    from rag.search.engine import SearchEngine
    engine = SearchEngine({"index_dir": str(tmp_path / "nope")})
    with pytest.raises(FileNotFoundError) as exc:
        engine.search("x")
    assert "rag.py compile" in str(exc.value)


# --------------------------------------------------------------------------------------
# dual-index consistency guards (plan risk: "Dual-index drift")
# --------------------------------------------------------------------------------------


def _write_vectors(index_dir, count: int, dim: int = 8, rows_done: int | None = None):
    import numpy as np
    (index_dir / "vector_meta.json").write_text(
        json.dumps({"model": "BAAI/bge-m3", "dim": dim, "count": count,
                    "normalized": True}), encoding="utf-8")
    (index_dir / "vectors.f32").write_bytes(
        np.ones((count, dim), dtype=np.float32).tobytes())
    if rows_done is not None:
        (index_dir / "vectors.f32.progress").write_text(f"sha-x\n{rows_done}\n",
                                                        encoding="utf-8")


def test_engine_refuses_vector_row_count_mismatch(tiny_index):
    """vectors.f32 built from a different chunk table must be refused, not used.

    Dense rows are positional (no ids), so a mismatch would attribute every dense
    score to the wrong chunk — silent corruption, not a crash.
    """
    from rag.search.engine import SearchEngine
    cfg, rows = tiny_index
    index_dir = Path(cfg["index_dir"])
    _write_vectors(index_dir, count=len(rows) + 7)  # wrong row count

    engine = SearchEngine(cfg)
    with pytest.raises(RuntimeError) as exc:
        engine.load()
    msg = str(exc.value)
    assert "different chunk table" in msg
    assert "--steps index --force" in msg  # tells the user how to fix it


def test_engine_refuses_partially_embedded_vectors(tiny_index):
    """An interrupted embed leaves a full-size file whose tail is zeros."""
    from rag.search.engine import SearchEngine
    cfg, rows = tiny_index
    index_dir = Path(cfg["index_dir"])
    n = len(rows)
    _write_vectors(index_dir, count=n, rows_done=n // 2)  # only half embedded

    engine = SearchEngine(cfg)
    with pytest.raises(RuntimeError) as exc:
        engine.load()
    assert "interrupted build" in str(exc.value)
    assert "pre-allocated zeros" in str(exc.value)


def test_engine_accepts_consistent_vectors(tiny_index, monkeypatch):
    """Matching row count + complete progress record loads normally."""
    import numpy as np

    from rag.search.engine import SearchEngine
    cfg, rows = tiny_index
    index_dir = Path(cfg["index_dir"])
    n = len(rows)
    _write_vectors(index_dir, count=n, rows_done=n)

    # stub the query embedder so this stays network-free (no BGE-M3 download).
    # Patch the name bound in the engine module, not its source module.
    monkeypatch.setattr("rag.search.engine.embed_query",
                        lambda q, model=None: np.ones(8, dtype=np.float32) /
                        np.sqrt(8))

    engine = SearchEngine(cfg)
    engine.load()
    assert engine.has_dense is True
    assert engine.n_chunks == n
    out = engine.search("rigidbody velocity", k=3, mode="hybrid")
    assert out["hits"]


def test_engine_accepts_vectors_without_progress_sidecar(tiny_index):
    """A complete file from before the sidecar existed must still load."""
    from rag.search.engine import SearchEngine
    cfg, rows = tiny_index
    index_dir = Path(cfg["index_dir"])
    _write_vectors(index_dir, count=len(rows), rows_done=None)

    engine = SearchEngine(cfg)
    engine.load()
    assert engine.has_dense is True


# --------------------------------------------------------------------------------------
# dense-missing warning (mode=hybrid/dense silently degrades to BM25 without vectors)
# --------------------------------------------------------------------------------------


def test_hybrid_without_vectors_warns_once(tiny_index, capsys):
    """hybrid/dense with no vectors must warn on stderr — the degradation is
    otherwise invisible in --text mode, and dense is what rescues zero-hit queries.
    Also guards the sys import the warning needs (it was missing -> NameError)."""
    from rag.search.engine import SearchEngine
    cfg, rows = tiny_index
    engine = SearchEngine(cfg)
    engine.search("rigidbody velocity", k=3, mode="hybrid")
    engine.search("rigidbody velocity", k=3, mode="hybrid")
    err = capsys.readouterr().err
    assert "no dense vectors" in err
    assert err.count("no dense vectors") == 1, "must warn only once per engine"


def test_bm25_mode_does_not_warn(tiny_index, capsys):
    from rag.search.engine import SearchEngine
    cfg, rows = tiny_index
    SearchEngine(cfg).search("rigidbody velocity", k=3, mode="bm25")
    assert "no dense vectors" not in capsys.readouterr().err
