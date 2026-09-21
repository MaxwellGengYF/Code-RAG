"""The `local` extra contract: torch/sentence-transformers are OPT-IN.

Default dependencies are the BM25 engine + corpus generation (remote providers
over HTTP). The CUDA torch wheels are multi-GB, so they live in the ``local``
extra and a plain ``uv sync`` / ``uv run`` must never pull them. That makes the
dense paths (BGE-M3 embed, cross-encoder rerank, Ollama embedder) *optional at
runtime*, so each one has to degrade or fail with an actionable hint instead of
leaking a bare ModuleNotFoundError:

  * ``compile --steps deps`` warns and continues (BM25 workflows are unaffected),
  * ``--steps index`` fails with the install hint when dense is requested,
  * search degrades to BM25-only, warning once, not per query.

Network-free: the missing-stack case is simulated by setting ``sys.modules[name]
= None``, which makes ``import name`` raise ImportError regardless of whether the
package actually is installed in the test environment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HEAVY = ("torch", "sentence-transformers", "ollama")


def _hide(monkeypatch, *mods: str) -> None:
    """Make ``import <mod>`` fail (None in sys.modules -> ImportError)."""
    for mod in mods:
        monkeypatch.setitem(sys.modules, mod, None)


# --------------------------------------------------------------------- pyproject


def test_pyproject_keeps_local_inference_out_of_default_dependencies(repo_root):
    """torch et al. must be in the `local` extra, never in `[project].dependencies`."""
    import tomllib

    data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    default = " ".join(data["project"]["dependencies"]).lower()
    extras = data["project"]["optional-dependencies"]
    local = " ".join(extras.get("local", [])).lower()

    for pkg in HEAVY:
        assert pkg not in default, f"{pkg} must not be a default dependency"
        assert pkg in local, f"{pkg} must be in the `local` extra"
    # moving torch out of the default deps must not lose the CUDA wheel index
    sources = data["tool"]["uv"]["sources"]["torch"]
    assert any(s.get("index") == "pytorch-cu126" for s in sources), sources


# --------------------------------------------------------------------- deps step


def test_deps_step_warns_and_continues_without_local_stack(monkeypatch, capsys):
    import rag.compile as compile_mod

    _hide(monkeypatch, "torch", "sentence_transformers")
    rc = compile_mod.run_deps({"embed_model": "BAAI/bge-m3"})
    err = capsys.readouterr().err

    assert rc == 0, "a BM25-only workflow must not be blocked by the missing extra"
    assert "uv sync --extra local" in err
    assert "torch" in err and "sentence_transformers" in err


def test_deps_step_fails_when_core_dependency_missing(monkeypatch, capsys):
    import rag.compile as compile_mod

    _hide(monkeypatch, "msgspec")
    rc = compile_mod.run_deps({"embed_model": "BAAI/bge-m3"})
    err = capsys.readouterr().err

    assert rc == 1
    assert "MISSING: msgspec" in err


def test_deps_step_skips_download_when_dense_disabled(monkeypatch, capsys):
    """embed_model none/"" -> nothing to pre-download, no local-extra complaint."""
    import rag.compile as compile_mod

    _hide(monkeypatch, "torch", "sentence_transformers")
    for cfg in ({"embed_model": "none"}, {"embed_model": ""}):
        assert compile_mod.run_deps(cfg) == 0
    assert "uv sync --extra local" not in capsys.readouterr().err


# ------------------------------------------------------------------ error surface


def test_ensure_embed_model_hint_and_exception_type(monkeypatch):
    import rag.index.vector_index as vi

    monkeypatch.setattr(vi, "_MODEL", None)
    _hide(monkeypatch, "torch", "sentence_transformers")
    with pytest.raises(vi.LocalStackMissing) as ei:
        vi.ensure_embed_model("BAAI/bge-m3")

    # ImportError for `except ImportError`, RuntimeError for the older guards
    assert isinstance(ei.value, ImportError)
    assert isinstance(ei.value, RuntimeError)
    assert "uv sync --extra local" in str(ei.value)
    assert vi._MODEL is None, "a failed load must not look like a loaded model"


def test_rerank_hint_when_stack_missing(monkeypatch):
    import rag.search.rerank as rerank

    monkeypatch.setattr(rerank, "_MODEL", None)
    _hide(monkeypatch, "sentence_transformers")
    with pytest.raises(ImportError) as ei:
        rerank._get_model("BAAI/bge-reranker-v2-m3")

    assert "uv sync --extra local" in str(ei.value)
    assert "rerank" in str(ei.value)


# --------------------------------------------------------------------- tiny index


def build_tiny_index(tmp_path: Path) -> dict:
    """A real BM25 index over three rows in an isolated index dir."""
    import msgspec
    from rag.corpus.schema import make_chunk_uid
    from rag.index.bm25_index import build_bm25
    from rag.index.build import ChunkRow

    rows = [
        ChunkRow(chunk_uid=make_chunk_uid("ScriptReference/Rigidbody.html", 0),
                 source="ScriptReference/Rigidbody.html", title="Rigidbody",
                 heading_path=["Rigidbody"],
                 text="Public Vector3 velocity; the velocity of the rigidbody."),
        ChunkRow(chunk_uid=make_chunk_uid("Manual/Physics.html", 0),
                 source="Manual/Physics.html", title="Physics",
                 heading_path=["Physics"],
                 text="Rigidbody 2D physics fundamentals for 2D games."),
        ChunkRow(chunk_uid=make_chunk_uid("Manual/Audio.html", 0),
                 source="Manual/Audio.html", title="Audio",
                 heading_path=["Audio"],
                 text="Audio mixer groups and snapshot automation curves."),
    ]
    # filler docs keep every term's df/N below finalize()'s 0.5 stop threshold
    for i, text in enumerate([
            "Textures and materials for terrain surfaces.",
            "Animation state machine transitions and blend trees.",
            "UI canvas scalers and anchoring for multiple resolutions.",
            "Lighting probes and reflection capture settings."]):
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
    return {"index_dir": str(index_dir), "embed_model": "BAAI/bge-m3",
            "min_should_match": 0.6, "bm25_k": 200, "dense_k": 200, "rrf_k": 60,
            "mode": "hybrid", "rerank": False}


def test_hybrid_query_degrades_to_bm25_when_embedder_missing(tmp_path, monkeypatch,
                                                             capsys):
    from rag.search.engine import SearchEngine

    engine = SearchEngine(build_tiny_index(tmp_path))
    engine.load()
    # vectors exist (dense was built) but the embedder cannot load
    engine.vectors = np.zeros((len(engine.rows), 4), dtype=np.float32)
    assert engine.has_dense

    def boom(*_a, **_kw):
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr("rag.search.engine.embed_query", boom)
    out = engine.search("rigidbody velocity", k=5, mode="hybrid")
    assert out["hits"], "must fall back to BM25 instead of raising"
    err = capsys.readouterr().err
    assert "uv sync --extra local" in err and "BM25-only" in err

    # warn ONCE: a repl session must not print the hint per query
    engine.search("rigidbody", k=3, mode="hybrid")
    assert "uv sync --extra local" not in capsys.readouterr().err


def test_rerank_requested_but_unavailable_returns_unreranked(tmp_path, monkeypatch,
                                                             capsys):
    import rag.search.rerank as rerank
    from rag.search.engine import SearchEngine

    cfg = build_tiny_index(tmp_path)
    cfg["rerank"] = True
    cfg["rerank_pool"] = 10
    monkeypatch.setattr(rerank, "_MODEL", None)
    _hide(monkeypatch, "sentence_transformers")

    out = SearchEngine(cfg).search("rigidbody velocity", k=5, mode="bm25")
    assert out["hits"], "must serve the unreranked order, not fail the query"
    assert all("rerank_score" not in h for h in out["hits"])
    assert "uv sync --extra local" in capsys.readouterr().err


# ------------------------------------------------------------------ dense build


def write_corpus(corpus_dir: Path, rel: str, texts: list[str]) -> None:
    path = corpus_dir / (rel + ".rag.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "source": rel, "title": Path(rel).stem, "html_md5": "x", "gen_key": "gk",
        "generated_at": "now",
        "chunks": [{"chunk_uid": "ignored", "heading_path": ["H"], "text": t,
                    "summary": "", "keywords": [], "synonyms": [], "qa": []}
                   for t in texts],
    }, ensure_ascii=False), encoding="utf-8")


def test_dense_index_build_without_stack_fails_with_hint(tmp_path, monkeypatch,
                                                         capsys):
    """A dense build must fail loudly+actionably, and leave no half-manifest."""
    import rag.index.vector_index as vi
    from rag.index.build import build_indexes

    root = tmp_path / "site"
    write_corpus(root / "corpus", "Manual/Physics.html",
                 ["velocity of the rigidbody", "physics overview"])
    cfg = {"_base_dir": str(root), "corpus_dir": "corpus", "index_dir": "index",
           "dirs": [], "embed_model": "BAAI/bge-m3", "mode": "bm25"}

    def missing(*_a, **_kw):
        raise vi.LocalStackMissing(f"No module named 'torch' — {vi.LOCAL_EXTRA_HINT}")

    monkeypatch.setattr(vi, "ensure_embed_model", missing)
    rc = build_indexes(cfg, force=True, verbose=False)
    err = capsys.readouterr().err

    assert rc == 1
    assert "uv sync --extra local" in err and "--skip-dense" in err
    # BM25 is already on disk, but the manifest (written last) is not: no artefact
    # is allowed to look like a finished build.
    assert (root / "index" / "bm25_word.pkl").exists()
    assert not (root / "index" / "manifest.json").exists()

    # the documented escape hatch completes the build
    monkeypatch.undo()
    rc2 = build_indexes(cfg, force=True, skip_dense=True, verbose=False)
    assert rc2 == 0
    assert (root / "index" / "manifest.json").exists()
