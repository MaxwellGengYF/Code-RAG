"""Does BGE-M3 dense subsume the legacy n-gram back-end's typo-tolerance role?

The plan's goal line says "BM25 + n-gram + dense", but the new index layer is
BM25 + dense only. The legacy n-gram back-end (index.pkl, character trigrams,
fuzziness=AUTO) was kept "for typo tolerance / comparison only" per AGENTS.md,
and measured catastrophically on the gold set (MRR 0.04-0.52 vs word 0.875).

So the question is narrow and empirical: on TYPO queries -- the one thing n-gram
was retained for -- does dense already cover it? If yes, dropping n-gram from the
new architecture costs nothing and the plan's three-way framing reduces to two
layers.

Compares four arms on identical queries:
  legacy ngram   index.pkl      + chunks.pkl   (character trigrams, fuzz AUTO)
  legacy word    index_word.pkl + chunks.pkl   (the proven old default)
  new bm25       index/         (word + path terms over clean+aux text)
  new hybrid     index/ or probe (BM25 + BGE-M3, RRF)

Usage: uv run python eval_typo_tolerance.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# (typo query, substring the intended page must contain)
TYPO_CASES = [
    ("Rigidboddy velocity", "Rigidbody"),
    ("MaterialPropertyBlokc", "MaterialPropertyBlock"),
    ("Renderer.SetProperyBlock", "Renderer.SetPropertyBlock"),
    ("Shader.ProperyToID", "Shader.PropertyToID"),
    ("Graphics.DrawMesch", "Graphics.DrawMesh"),
    ("Terrain.SetSplatMaterialPropetyBlock", "Terrain.SetSplatMaterialPropertyBlock"),
    ("RenderParams matPropz", "RenderParams-matProps"),
    ("gpu instacning per-instance properties", "gpu-instancing"),
]


def score_legacy(index_path: str, tokenizer_name: str, fuzziness, label: str):
    """Legacy engine (hybrid_retrieve) on its own artefacts."""
    from retrieval import InvertedIndex, Searcher
    from hybrid_retrieve import (dedupe_by_source,
                                 get_tokenizer, load_chunks_pickle)

    t0 = time.time()
    chunks = load_chunks_pickle("chunks.pkl")
    idx = InvertedIndex()
    idx.load(index_path)
    tok = get_tokenizer(tokenizer_name)
    searcher = Searcher(idx, tokenizer=tok, fuzziness=fuzziness,
                        min_should_match=0.6)
    load_s = time.time() - t0
    print(f"  [{label}] loaded {len(chunks)} chunks in {load_s:.0f}s", file=sys.stderr)

    found = 0
    for q, gold in TYPO_CASES:
        res = searcher.search(q, top_k=200)
        ranked = dedupe_by_source([d for d, _ in res], chunks, 1)[:5]
        srcs = [Path(str(chunks[d].source)).as_posix() for d in ranked]
        ok = any(gold.lower() in s.lower() for s in srcs)
        found += ok
        print(f"    {'OK  ' if ok else 'MISS'} {q[:40]:<40} -> "
              f"{Path(srcs[0]).name[:38] if srcs else '<0 hits>'}")
    return found


def score_new(mode: str, label: str, config: str = "rag_config.json"):
    from rag.compile import load_rag_config
    from rag.search.engine import SearchEngine

    engine = SearchEngine(load_rag_config(config))
    engine.load()
    print(f"  [{label}] {engine.n_chunks} chunks, dense={engine.has_dense}",
          file=sys.stderr)
    found = 0
    for q, gold in TYPO_CASES:
        hits = engine.search(q, k=5, mode=mode)["hits"]
        srcs = [h["source"] for h in hits]
        ok = any(gold.lower() in s.lower() for s in srcs)
        found += ok
        print(f"    {'OK  ' if ok else 'MISS'} {q[:40]:<40} -> "
              f"{Path(srcs[0]).name[:38] if srcs else '<0 hits>'}")
    return found


def main():
    n = len(TYPO_CASES)
    print(f"== typo tolerance: {n} queries ==\n")

    results = {}
    if Path("index.pkl").exists():
        print("[legacy ngram] character trigrams, fuzziness=AUTO")
        results["legacy ngram"] = score_legacy("index.pkl", "ngram", "AUTO",
                                               "legacy-ngram")
    if Path("index_word.pkl").exists():
        print("\n[legacy word] proven old default, fuzziness=0")
        results["legacy word"] = score_legacy("index_word.pkl", "word", 0,
                                              "legacy-word")

    print("\n[new bm25] word + path terms over clean+aux text")
    results["new bm25"] = score_new("bm25", "new-bm25")

    # hybrid needs vectors; use the probe index when the production one is BM25-only
    from rag.compile import load_rag_config
    from rag.search.engine_select import rag_index_status
    cfg_name = "rag_config.json"
    if not rag_index_status(load_rag_config(cfg_name))["has_dense"]:
        probe = Path("rag_probe_config.json")
        if probe.exists():
            cfg_name = "rag_probe_config.json"
            print(f"\n[new hybrid] (using probe index — production has no vectors yet)")
        else:
            cfg_name = None
    else:
        print("\n[new hybrid] BM25 + BGE-M3, RRF")
    if cfg_name:
        results["new hybrid"] = score_new("hybrid", "new-hybrid", cfg_name)
        results["new dense"] = score_new("dense", "new-dense", cfg_name)

    print("\n== summary: gold page found in top-5 ==")
    for k, v in results.items():
        print(f"  {k:<14} {v}/{n}")
    print("\nNOTE: legacy arms score against the FULL corpus (83k chunks); the new")
    print("arms score against whatever corpus is built so far, so a MISS on the")
    print("new side can mean 'page not generated yet' rather than a ranking failure.")


if __name__ == "__main__":
    main()
