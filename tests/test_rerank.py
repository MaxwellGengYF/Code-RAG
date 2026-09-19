"""Rerank path tests with a stubbed cross-encoder (network-free).

The optional bge-reranker was default-off and previously untested dead code. Two
bugs found while exercising it are locked in here:
  1. `_rerank` reordered hits but kept the ORIGINAL fusion scores, so the
     reported `fused_score` could not explain the reported order.
  2. reranking ran AFTER truncating to k, so it could only shuffle results that
     were already shown instead of promoting better candidates from the pool.
"""
from __future__ import annotations

import pytest

from rag.search.engine import SearchEngine


@pytest.fixture()
def engine_with_rerank(tmp_path, monkeypatch):
    """A real tiny BM25 index plus a stub cross-encoder with a known preference."""
    import msgspec
    from rag.corpus.schema import make_chunk_uid
    from rag.index.bm25_index import build_bm25
    from rag.index.build import ChunkRow

    rows = [
        # BM25 will rank the keyword-dense one first; the stub reranker prefers
        # the second, so a correct implementation must REORDER them.
        ChunkRow(chunk_uid=make_chunk_uid("ScriptReference/A.html", 0),
                 source="ScriptReference/A.html", title="A",
                 heading_path=["A"],
                 text="velocity velocity velocity velocity velocity rigidbody"),
        ChunkRow(chunk_uid=make_chunk_uid("ScriptReference/B.html", 0),
                 source="ScriptReference/B.html", title="B",
                 heading_path=["B"],
                 text="The authoritative answer about rigidbody velocity semantics."),
        ChunkRow(chunk_uid=make_chunk_uid("Manual/C.html", 0),
                 source="Manual/C.html", title="C",
                 heading_path=["C"],
                 text="unrelated material about audio mixer groups and snapshots"),
    ]
    # Filler docs keep df/N below InvertedIndex.finalize()'s 0.5 stop threshold:
    # with only 3 rows, 'rigidbody'/'velocity' would be pruned as "too common"
    # and every query would return zero hits.
    for i, text in enumerate([
        "Textures and materials for terrain surfaces and layer blending.",
        "Animation state machine transitions, blend trees and root motion.",
        "UI canvas scalers, anchors and layout groups for resolutions.",
        "Shader variant stripping, keyword multi compile and warmup.",
        "Addressables groups, remote catalogs and content update builds.",
        "Navmesh baking, agent carving and off mesh link generation.",
    ]):
        rows.append(ChunkRow(
            chunk_uid=make_chunk_uid(f"Manual/Filler{i}.html", 0),
            source=f"Manual/Filler{i}.html", title=f"Filler {i}",
            heading_path=[f"Filler {i}"], text=text))
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "chunks.msgpack").write_bytes(
        msgspec.msgpack.encode([msgspec.to_builtins(r) for r in rows]))
    chunks = [r.to_corpus_chunk() for r in rows]
    index, _s = build_bm25(chunks, [r.source for r in rows],
                           [r.title for r in rows], verbose=False)
    index.save(str(index_dir / "bm25_word.pkl"))

    cfg = {"index_dir": str(index_dir), "min_should_match": 0.0,
           "bm25_k": 200, "dense_k": 200, "rrf_k": 60, "embed_model": "none",
           "mode": "bm25", "rerank": True,
           "rerank_model": "stub-reranker"}

    calls: list[list[tuple[str, str]]] = []

    def fake_rerank(pairs, *, model):
        calls.append(list(pairs))
        # prefer the passage that reads like a real answer. NOTE: each pair is a
        # (query, passage) TUPLE, so test the passage element, not membership.
        return [10.0 if "authoritative" in passage else 1.0
                for _query, passage in pairs]

    import rag.search.rerank as rerank_mod
    monkeypatch.setattr(rerank_mod, "rerank_pairs", fake_rerank)
    return SearchEngine(cfg), calls


def test_rerank_reorders_hits(engine_with_rerank):
    engine, _calls = engine_with_rerank
    out = engine.search("rigidbody velocity", k=3, mode="bm25")
    srcs = [h["source"] for h in out["hits"]]
    assert srcs[0].endswith("B.html"), srcs  # reranker's pick must come first
    assert out.get("reranked") is True


def test_rerank_score_reported_separately(engine_with_rerank):
    """The rerank score must be exposed, not silently overwrite fused_score."""
    engine, _calls = engine_with_rerank
    out = engine.search("rigidbody velocity", k=3, mode="bm25")
    top = out["hits"][0]
    assert "rerank_score" in top
    assert top["rerank_score"] == pytest.approx(10.0)
    # fused_score still carries the fusion signal (different scale, still present)
    assert "fused_score" in top
    assert top["fused_score"] < top["rerank_score"]


def test_rerank_pool_is_wider_than_k(engine_with_rerank):
    """Reranking must see more candidates than k, or it cannot promote anything."""
    engine, calls = engine_with_rerank
    engine.search("rigidbody velocity", k=1, mode="bm25")
    assert calls, "reranker was never called"
    assert len(calls[0]) > 1, (
        f"reranker saw only {len(calls[0])} pair(s) for k=1 — it must score a "
        f"wider pool so a better page can be promoted into the results")


def test_no_rerank_flag_disables_it(engine_with_rerank):
    engine, calls = engine_with_rerank
    out = engine.search("rigidbody velocity", k=3, mode="bm25", no_rerank=True)
    assert not calls, "reranker ran despite no_rerank=True"
    assert "reranked" not in out
    assert all("rerank_score" not in h for h in out["hits"])
    # without the reranker, BM25's keyword-dense A.html leads
    assert out["hits"][0]["source"].endswith("A.html")


def test_rerank_off_by_default(tmp_path, monkeypatch):
    """rerank is opt-in via config; the default path must not touch the model."""
    import msgspec
    from rag.corpus.schema import make_chunk_uid
    from rag.index.bm25_index import build_bm25
    from rag.index.build import ChunkRow

    rows = [ChunkRow(chunk_uid=make_chunk_uid("ScriptReference/A.html", 0),
                     source="ScriptReference/A.html", title="A",
                     heading_path=["A"], text="rigidbody velocity")]
    # dilute df/N below finalize()'s 0.5 stop threshold (see fixture above)
    for i, text in enumerate([
        "Textures and materials for terrain surfaces and layer blending.",
        "Animation state machine transitions, blend trees and root motion.",
        "UI canvas scalers, anchors and layout groups for resolutions.",
    ]):
        rows.append(ChunkRow(
            chunk_uid=make_chunk_uid(f"Manual/Filler{i}.html", 0),
            source=f"Manual/Filler{i}.html", title=f"Filler {i}",
            heading_path=[f"Filler {i}"], text=text))
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "chunks.msgpack").write_bytes(
        msgspec.msgpack.encode([msgspec.to_builtins(r) for r in rows]))
    index, _s = build_bm25([r.to_corpus_chunk() for r in rows],
                           [r.source for r in rows], [r.title for r in rows],
                           verbose=False)
    index.save(str(index_dir / "bm25_word.pkl"))

    import rag.search.rerank as rerank_mod

    def boom(*_a, **_kw):
        raise AssertionError("reranker must not be imported/called by default")

    monkeypatch.setattr(rerank_mod, "rerank_pairs", boom)
    engine = SearchEngine({"index_dir": str(index_dir), "min_should_match": 0.0,
                           "embed_model": "none", "mode": "bm25"})
    out = engine.search("rigidbody velocity", k=3, mode="bm25")
    assert out["hits"]
    assert "reranked" not in out


# --------------------------------------------------------------------------------------
# SA-6 parity: every legacy CLI flag must reach the engine, not be silently dropped
# --------------------------------------------------------------------------------------


def test_run_search_forwards_mentions_flags(monkeypatch, capsys):
    """--mentions-context/--mentions-limit must reach engine.mentions.

    These were previously accepted by argparse and then dropped, so the output
    looked plausible while silently ignoring the flags -- worse than erroring.
    """
    import rag.cli.search_cmd as sc

    seen = {}

    class FakeEngine:
        cfg = {}
        n_chunks = 1
        has_dense = False

        def load(self):
            pass

        def mentions(self, term, *, limit=60, context=0):
            seen["term"] = term
            seen["limit"] = limit
            seen["context"] = context
            return [{"source": "ScriptReference/A.html", "count": 3,
                     "context": "some surrounding text"}]

    monkeypatch.setattr("rag.search.engine.SearchEngine", lambda cfg: FakeEngine())
    sc.run_search(mentions="Foo", mentions_limit=7, mentions_context=120,
                  text=False, legacy=False)
    assert seen == {"term": "Foo", "limit": 7, "context": 120}
    body = capsys.readouterr().out
    assert '"context"' in body, "context must appear in JSON output"


def test_run_search_legacy_forwards_mentions_flags(monkeypatch):
    """The --legacy delegation must forward the same flags to hybrid_retrieve."""
    import rag.cli.search_cmd as sc

    captured = {}

    class FakeLegacy:
        @staticmethod
        def main(argv):
            captured["argv"] = argv
            return 0

    monkeypatch.setitem(__import__("sys").modules, "hybrid_retrieve", FakeLegacy())
    sc.run_search(mentions="Foo", mentions_limit=5, mentions_context=80,
                  legacy=True, snippet_width=300)
    argv = captured["argv"]
    assert "--mentions" in argv and "Foo" in argv
    assert argv[argv.index("--mentions-limit") + 1] == "5"
    assert argv[argv.index("--mentions-context") + 1] == "80"
    assert argv[argv.index("--snippet-width") + 1] == "300"
