"""End-to-end integration test: tiny mirror -> compile -> index -> search.

Plan §5 requires the integration path `compile --max-files N` -> `compile index`
-> search smoke on known-answer queries. Everything here is network-free: a
synthetic HTML mirror under tmp_path, a scripted fake LLM client, and a stub
embedder standing in for BGE-M3 (real dense encoding needs a model download).

This is the only test that exercises the WHOLE pipeline through its public
entry points, so it catches wiring mistakes that per-module unit tests cannot:
manifest threading between stages, chunk-table handoff to the index builder,
gen_key guards, engine loading, and the coverage-based engine switch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rag.cli.compile_cmd import (
    ProviderShard,
    finalize,
    plan_work,
    run_corpus_compile,
    set_gen_key,
    )
from rag.corpus import extract_page
from rag.store import CorpusStore, FileManager


# --------------------------------------------------------------------------------------
# synthetic mirror
# --------------------------------------------------------------------------------------

#: (rel path, title, body prose) — bodies use distinct vocabulary so BM25 has
#: something to discriminate on, and one page's body deliberately omits its own
#: symbol name to exercise the path_terms rescue.
PAGES = [
    ("ScriptReference/Rigidbody.html", "Rigidbody",
     "Controls the position and velocity of a GameObject through physics simulation. "
     "Add a Rigidbody to make an object respond to forces and gravity.", None),
    ("ScriptReference/Rigidbody.AddForce.html", "Rigidbody.AddForce",
     "Applies a force to the rigidbody, optionally using ForceMode.",
     "public void AddForce(Vector3 force, ForceMode mode);"),
    ("ScriptReference/MaterialPropertyBlock.html", "MaterialPropertyBlock",
     "A block of material values to use in addition to a Renderer's material. "
     "SetFloat SetColor SetVector SetMatrix SetTexture SetBuffer.", None),
    # body never mentions its own symbol -> only path terms can find it
    ("ScriptReference/Renderer.SetPropertyBlock.html", "Renderer.SetPropertyBlock",
     "Lets you set or clear per-renderer property overrides without instantiating "
     "a copy of the shared material, keeping draw call batching intact.", None),
    ("Manual/PhysicsOverview.html", "Physics overview",
     "Unity simulates rigid body dynamics with PhysX. Colliders detect overlap and "
     "joints constrain movement between bodies.", None),
]


def write_mirror(root: Path) -> Path:
    for rel, title, body, sig in PAGES:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        # A real Unity API page puts the C# declaration in a div.sig-block, which
        # dumpdoc renders as a ```csharp fence; plain prose goes in <p>.
        sig_html = (f"<div class='sig-block'>Declaration\n{sig}</div>" if sig else "")
        p.write_text(
            "<html><head><title>Unity - Scripting API: "
            f"{title}</title></head><body>"
            "<div id='content-wrap'><div class='content-block'><div class='content'>"
            f"<div class='section'><h1>{title}</h1>"
            f"{sig_html}<p>{body}</p></div>"
            "</div><div class='footer-wrapper'><div class='footer'>"
            "Copyright © 2026 Unity Technologies. All rights reserved.</div></div>"
            "</div></body></html>", encoding="utf-8")
    return root


class FakeLLM:
    """Scripted corpus builder: emits ONE verbatim chunk covering the whole body.

    Uses the real extract_page markdown so the verbatim-substring invariant in
    generate_page_corpus holds exactly as it would against a real model.
    """

    def __init__(self, model: str = "fake-model"):
        self._model = model
        self.calls = 0
        self.root: Path | None = None

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str):
        from rag.llm.base import GenerationResult

        self.calls += 1
        # the user prompt ends with the page markdown; take its last paragraph as
        # a verbatim excerpt (exactly what a compliant model must do)
        lines = [ln.strip() for ln in user_prompt.splitlines() if ln.strip()]
        text = next((ln for ln in reversed(lines)
                     if len(ln) > 40 and not ln.startswith("#")), "")
        title = next((ln.lstrip("# ").strip() for ln in lines
                      if ln.startswith("#")), "Page")
        payload = {"chunks": [{
            "heading_path": [title],
            "text": text,
            "summary": f"Retrieval summary for {title}.",
            "keywords": [w for w in title.split(".") if w],
            "synonyms": [f"{title} alias"],
            "qa": [{"q": f"How do I use {title}?",
                    "a": f"See the {title} documentation."}],
        }]}
        return GenerationResult(text=json.dumps(payload, ensure_ascii=False),
                                input_tokens=100, output_tokens=80)


# --------------------------------------------------------------------------------------
# pipeline stages
# --------------------------------------------------------------------------------------


@pytest.fixture()
def pipeline(tmp_path, monkeypatch):
    """Run corpus generation + BM25 index build over the synthetic mirror.

    Returns everything the assertions need. Dense vectors are stubbed (a real
    BGE-M3 encode would download a model); the stub produces deterministic
    non-zero unit vectors so the hybrid path executes and row alignment holds.
    """
    root = write_mirror(tmp_path / "site")
    corpus_dir = root / "corpus"
    index_dir = root / "index"

    fm = FileManager(root=root, dirs=["Manual", "ScriptReference"],
                     corpus_dir=corpus_dir)
    store = CorpusStore(corpus_dir)
    scanned = fm.scan()
    assert len(scanned) == len(PAGES)

    fake = FakeLLM()
    shards = [ProviderShard(fake, fake.model_name)]
    cfg = {"max_input_chars": 24000, "max_chunk_chars": 1200,
           "corpus_dir": str(corpus_dir), "index_dir": str(index_dir),
           "dirs": [str(root / "Manual"), str(root / "ScriptReference")],
           "path_boost": 3, "embed_model": "stub-model",
           "min_should_match": 0.0, "bm25_k": 200, "dense_k": 200,
           "rrf_k": 60, "mode": "hybrid"}

    import asyncio
    work = plan_work(fm, store, fm.diff(scanned))
    report = asyncio.run(run_corpus_compile(
        shards, cfg, work=work, scanned=scanned, fm=fm,
        workers_per_provider=2, progress=False))
    finalize(fm, fm.diff(scanned), scanned,
             fm.load_manifest().get("page_gen_keys", {}),
             set_gen_key(shards), dict(shards[0].gen_parts), report)

    # stub the embedder so no model is downloaded
    import rag.index.vector_index as vi

    DIM = 16

    class StubEmbedModel:
        def get_embedding_dimension(self):
            return DIM

        def encode(self, texts, batch_size=32, normalize_embeddings=True,
                   convert_to_numpy=True, show_progress_bar=False):
            # build_vectors_resumable calls model.encode() directly (not via
            # embed_texts), so the stub must implement it too
            return fake_embed_texts(list(texts))

    def fake_embed_texts(texts, **kw):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t[:DIM]):
                out[i, j] = float(ord(ch)) / 1000.0
            out[i, -1] = 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out

    def fake_embed_query(query, **kw):
        return fake_embed_texts([query])[0]

    monkeypatch.setattr(vi, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(vi, "embed_query", fake_embed_query)
    monkeypatch.setattr("rag.search.engine.embed_query", fake_embed_query)
    monkeypatch.setattr(vi, "ensure_embed_model", lambda name: StubEmbedModel())

    from rag.index.build import build_indexes
    rc = build_indexes(cfg, force=True, verbose=False)
    assert rc == 0, "index build failed"

    from rag.search.engine import SearchEngine
    engine = SearchEngine(cfg)
    engine.load()
    return {"root": root, "fm": fm, "store": store, "cfg": cfg, "engine": engine,
            "report": report, "fake": fake, "scanned": scanned, "shards": shards}


# --------------------------------------------------------------------------------------
# assertions
# --------------------------------------------------------------------------------------


def test_corpus_stage_generated_every_page(pipeline):
    report, store = pipeline["report"], pipeline["store"]
    assert report.pages_done == len(PAGES)
    assert report.fallback == 0, report.failures  # fake client is always valid
    assert report.first_try == len(PAGES)
    files = dict(store.iterate_all())
    assert len(files) == len(PAGES)
    for rel, data in files.items():
        assert data["chunks"], f"{rel} has no chunks"
        assert data["needs_regen"] is False
        assert data["gen_key"] == pipeline["shards"][0].gen_key


def test_boilerplate_is_stripped_from_corpus(pipeline):
    """doc_clean's footer rules must keep site chrome out of the corpus."""
    store = pipeline["store"]
    for rel, data in store.iterate_all():
        blob = json.dumps(data, ensure_ascii=False)
        assert "Copyright © 2026 Unity Technologies" not in blob, rel


def test_index_stage_built_all_artefacts(pipeline):
    index_dir = Path(pipeline["cfg"]["index_dir"])
    for name in ("chunks.msgpack", "bm25_word.pkl", "vectors.f32",
                 "vector_meta.json", "manifest.json", "vectors.f32.progress"):
        assert (index_dir / name).exists(), f"missing {name}"
    meta = json.loads((index_dir / "vector_meta.json").read_text(encoding="utf-8"))
    n_rows = len(pipeline["engine"].rows)
    assert meta["count"] == n_rows
    assert meta["model"] == "stub-model"


def test_search_finds_known_answers(pipeline):
    """The plan's known-answer smoke queries, on the synthetic mirror."""
    engine = pipeline["engine"]
    cases = [
        ("Rigidbody AddForce", "Rigidbody.AddForce.html"),
        ("MaterialPropertyBlock SetFloat SetColor", "MaterialPropertyBlock.html"),
        ("physics simulation velocity gravity", "Rigidbody.html"),
    ]
    for query, expect in cases:
        hits = engine.search(query, k=5, mode="bm25")["hits"]
        srcs = [h["source"] for h in hits]
        assert any(expect in s for s in srcs), f"{query!r}: expected {expect}, got {srcs}"


def test_path_terms_rescue_page_that_omits_its_own_symbol(pipeline):
    """Renderer.SetPropertyBlock's body never mentions 'SetPropertyBlock'.

    This is the exact case AGENTS.md says the word tokenizer's path/title field
    terms exist for; without path_boost the page is unreachable by symbol name.
    """
    engine = pipeline["engine"]
    hits = engine.search("Renderer.SetPropertyBlock", k=5, mode="bm25")["hits"]
    srcs = [h["source"] for h in hits]
    assert any("Renderer.SetPropertyBlock.html" in s for s in srcs), srcs
    # and the hit carries a snippet from the clean text
    hit = next(h for h in hits if "SetPropertyBlock" in h["source"])
    assert "per-renderer" in hit["text"].lower()


def test_hybrid_mode_runs_and_is_deterministic(pipeline):
    engine = pipeline["engine"]
    a = engine.search("rigidbody velocity", k=5, mode="hybrid")
    b = engine.search("rigidbody velocity", k=5, mode="hybrid")
    assert a["hits"] and [(h["source"], h["chunk_uid"], h["fused_score"])
                          for h in a["hits"]] == \
        [(h["source"], h["chunk_uid"], h["fused_score"]) for h in b["hits"]]
    assert engine.has_dense is True


def test_dense_mode_returns_results(pipeline):
    engine = pipeline["engine"]
    out = engine.search("rigidbody velocity", k=3, mode="dense")
    assert out["hits"]


def test_mentions_enumerates_literally(pipeline):
    engine = pipeline["engine"]
    res = engine.mentions("rigidbody")
    assert res
    srcs = {r["source"] for r in res}
    assert any("Rigidbody.html" in s for s in srcs)
    counts = [r["count"] for r in res]
    assert counts == sorted(counts, reverse=True)


def test_explain_reports_term_df(pipeline):
    engine = pipeline["engine"]
    out = engine.search("MaterialPropertyBlock", k=3, mode="bm25", explain=True)
    terms = {t["term"]: t["df"] for t in out["explain"]}
    assert terms.get("materialpropertyblock", 0) > 0


def test_rerun_compile_is_a_noop(pipeline):
    """DoD#1: after a successful compile, a rerun must process zero pages."""
    fm, store = pipeline["fm"], pipeline["store"]
    shards = pipeline["shards"]
    work = plan_work(fm, store, fm.diff(fm.scan()))
    assert work == [], work
    assert pipeline["fake"].calls == len(PAGES), "no extra LLM calls expected"


def test_touching_one_html_requeues_only_that_page(pipeline):
    """DoD#1: md5-incremental regeneration must be page-scoped."""
    fm, store, root = pipeline["fm"], pipeline["store"], pipeline["root"]
    shards = pipeline["shards"]
    target = "ScriptReference/Rigidbody.AddForce.html"
    p = root / target
    p.write_text(p.read_text(encoding="utf-8").replace("ForceMode", "ForceMode2"),
                 encoding="utf-8")
    work = plan_work(fm, store, fm.diff(fm.scan()))
    assert work == [target], work


def test_index_refuses_stale_gen_key(pipeline):
    """Plan risk 'dual-index drift': build must refuse on a gen_key change."""
    from rag.index.build import build_indexes

    cfg = dict(pipeline["cfg"])
    im = Path(cfg["index_dir"]) / "manifest.json"
    data = json.loads(im.read_text(encoding="utf-8"))
    data["corpus_gen_key"] = "different-generation"
    im.write_text(json.dumps(data), encoding="utf-8")
    rc = build_indexes(cfg, verbose=False)
    assert rc == 1


def test_extract_page_renders_signature_as_code_fence(tmp_path):
    """The dumpdoc renderer must fence C# signatures so the LLM never splits them."""
    root = write_mirror(tmp_path / "site2")
    page = extract_page(root / "ScriptReference/Rigidbody.AddForce.html", root=root)
    assert page is not None
    assert page.title == "Rigidbody.AddForce"
    assert "```csharp" in page.markdown
    assert "public void AddForce(Vector3 force, ForceMode mode);" in page.markdown
