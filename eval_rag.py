"""RAG retrieval quality eval: the new engine vs the old BM25 baseline.

Uses the SAME 24-query gold set as eval_retrieval.py (eval_lib.GOLD) plus
optionally the extended LLM-assisted set (eval_gold_extended.json), so numbers
are directly comparable: baseline word tokenizer MRR 0.875 / hit@1 0.833 /
hit@10 0.917.

Ablations run against the loaded chunk table with in-memory BM25 rebuilds
(aux on/off, path_boost, rrf_k) — no artefact rebuilds needed. Dense is
BGE-M3 over to_embed_text (clean text only).

Usage:
    uv run python eval_rag.py                          # default hybrid config
    uv run python eval_rag.py --mode bm25|dense|hybrid
    uv run python eval_rag.py --sweep-aux              # aux vs no-aux
    uv run python eval_rag.py --sweep-path-boost 0,1,3,5
    uv run python eval_rag.py --sweep-rrf-k 20,40,60,120
    uv run python eval_rag.py --gold-set ext           # extended gold set
    uv run python eval_rag.py --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_lib import GOLD as GOLD_BASE
from eval_lib import evaluate

EXTENDED_GOLD_PATH = Path(__file__).resolve().parent / "eval_gold_extended.json"


def load_gold(name: str):
    if name == "base":
        return GOLD_BASE, "base-24"
    if name == "ext":
        if not EXTENDED_GOLD_PATH.exists():
            raise SystemExit(f"{EXTENDED_GOLD_PATH} not found — generate it first")
        data = json.loads(EXTENDED_GOLD_PATH.read_text(encoding="utf-8"))
        return [(q["query"], q["gold"]) for q in data], f"ext-{len(data)}"
    if name == "all":
        ext, _ = load_gold("ext")
        return GOLD_BASE + ext, f"all-{len(GOLD_BASE) + len(ext)}"
    raise SystemExit(f"unknown gold set {name!r}")


# --------------------------------------------------------------------------------------
# ranking adapters
# --------------------------------------------------------------------------------------


def engine_ranker(engine, k: int, mode: str):
    def rank(query: str) -> list[str]:
        out = engine.search(query, k=k, mode=mode, no_rerank=True)
        return [h["source"] for h in out["hits"]]
    return rank


def table_ranker(rows, cfg, *, mode="hybrid", aux=True, path_boost=3, rrf_k=60,
                 per_file=1):
    """In-memory ranker over the chunk table with explicit ablation knobs."""
    import heapq
    import numpy as np
    from rag.corpus.schema import CorpusChunk, to_embed_text
    from rag.index.bm25_index import build_bm25, new_searcher
    from rag.index.vector_index import embed_query
    from rag.index.fuse import rrf_fuse

    chunks = [r.to_corpus_chunk() for r in rows]
    sources = [r.source for r in rows]
    titles = [r.title for r in rows]
    index, _ = build_bm25(chunks, sources, titles, path_boost=path_boost,
                          aux=aux, verbose=False)
    searcher = new_searcher(index)
    dense_mat = None
    vecs_path = Path(engine_index_dir(cfg)) / "vectors.f32"
    meta_path = Path(engine_index_dir(cfg)) / "vector_meta.json"
    if mode in ("hybrid", "dense") and vecs_path.exists():
        from rag.index.vector_index import load_meta, load_vectors
        meta = load_meta(meta_path)
        dense_mat = load_vectors(vecs_path, dim=meta["dim"], count=meta["count"])

    def rank(query: str) -> list[str]:
        bm25 = searcher.search(query, top_k=int(cfg.get("bm25_k", 200)))
        dense = []
        if dense_mat is not None:
            q = embed_query(query, model=cfg.get("embed_model", "BAAI/bge-m3"))
            scores = dense_mat @ q
            top = heapq.nlargest(int(cfg.get("dense_k", 200)), range(len(scores)),
                                 key=lambda i: (float(scores[i]), -i))
            dense = [(i, float(scores[i])) for i in top]
        fused = rrf_fuse(bm25, dense, k=rrf_k) if mode == "hybrid" and dense else (
            dense if mode == "dense" else bm25)
        seen: dict[str, int] = {}
        out = []
        for d, _s in fused:
            src = sources[d]
            if seen.get(src, 0) >= per_file:
                continue
            seen[src] = seen.get(src, 0) + 1
            out.append(src)
        return out[:10]
    return rank


def engine_index_dir(cfg) -> str:
    from rag import resolve_path
    return str(resolve_path(cfg.get("index_dir", "index")))


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    ap.add_argument("--gold-set", choices=["base", "ext", "all"], default="all")
    ap.add_argument("--configs", default=None,
                    help="legacy compat: ignored, single engine evaluated per run")
    ap.add_argument("--sweep-aux", action="store_true")
    ap.add_argument("--sweep-path-boost", default=None, help="e.g. 0,1,3,5")
    ap.add_argument("--sweep-rrf-k", default=None, help="e.g. 20,40,60,120")
    ap.add_argument("--per-file", type=int, default=None)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--show-gold", action="store_true")
    args = ap.parse_args()

    gold, gold_name = load_gold(args.gold_set)
    if args.show_gold:
        for q, g in gold:
            print(f"{q!r:70} -> {g}")
        return 0

    from rag.compile import load_rag_config
    from rag.search.engine import SearchEngine
    from rag.index.build import flatten_corpus
    from rag.store import CorpusStore
    from rag import resolve_path

    cfg = load_rag_config()
    if args.rerank:
        cfg["rerank"] = True

    t0 = time.time()
    store = CorpusStore(resolve_path(cfg.get("corpus_dir", "corpus")))
    rows, n_pages = flatten_corpus(store)
    if not rows:
        raise SystemExit("corpus is empty — run rag.py compile first")
    print(f"[eval] {len(rows)} chunks from {n_pages} pages "
          f"(table load {time.time() - t0:.1f}s)", file=sys.stderr)

    engine = SearchEngine(cfg)
    engine.load()

    results = []
    src_id = {r.source: i for i, r in enumerate(rows)}

    def _sources_to_ids(sources):
        # evaluate() indexes its chunks list by returned ids; map sources -> row ids
        return [src_id[s] for s in sources if s in src_id]

    def run(name, rank_fn):
        res = evaluate(name, rows, lambda q: _sources_to_ids(rank_fn(q)),
                       verbose=args.verbose)
        res["gold"] = gold_name
        results.append(res)
        print(f"  {name:<28} MRR={res['MRR']:.3f} hit@1={res['hit@1']:.3f} "
              f"hit@10={res['hit@10']:.3f}")
        return res

    print(f"[eval] gold set: {gold_name} ({len(gold)} queries)")
    if args.sweep_aux:
        for aux in (True, False):
            run(f"table hybrid aux={aux}",
                table_ranker(rows, cfg, mode="hybrid", aux=aux))
        return emit(results, args)
    if args.sweep_path_boost:
        for pb in [int(x) for x in args.sweep_path_boost.split(",")]:
            run(f"table hybrid path_boost={pb}",
                table_ranker(rows, cfg, mode="hybrid", path_boost=pb))
        return emit(results, args)
    if args.sweep_rrf_k:
        for k in [int(x) for x in args.sweep_rrf_k.split(",")]:
            run(f"table hybrid rrf_k={k}",
                table_ranker(rows, cfg, mode="hybrid", rrf_k=k))
        return emit(results, args)
    if args.per_file is not None:
        run(f"table hybrid per_file={args.per_file}",
            table_ranker(rows, cfg, mode="hybrid", per_file=args.per_file))
        return emit(results, args)

    # standard engine path (dense uses the built vectors)
    run(f"engine mode={args.mode} rerank={cfg.get('rerank', False)}",
        engine_ranker(engine, 10, args.mode))
    return emit(results, args)


def emit(results: list[dict], args) -> int:
    path = Path(__file__).resolve().parent / "eval_results.md"
    baseline = "| word / default (old) | word | 0 | 0.875 | 0.833 | 0.917 |"
    lines = [
        "# Retrieval evaluation results",
        "",
        "Same gold set as eval_retrieval.py; new engine = rag.search (BM25 + BGE-M3, RRF).",
        "",
        "| config | MRR | hit@1 | hit@10 |",
        "| --- | --- | --- | --- |",
        "| **old baseline** | **0.875** | **0.833** | **0.917** |",
    ]
    for r in results:
        lines.append(f"| {r['config']} ({r['gold']}) | {r['MRR']} | {r['hit@1']} | "
                     f"{r['hit@10']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] wrote {path}")
    gate = [r for r in results if r["MRR"] >= 0.875 and r["hit@10"] >= 0.917]
    if gate:
        print(f"[eval] GATE PASS: {gate[0]['config']} meets/exceeds the baseline")
    else:
        print("[eval] GATE FAIL: no config reached MRR>=0.875 AND hit@10>=0.917 — "
              "keep the old engine default until it does")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
