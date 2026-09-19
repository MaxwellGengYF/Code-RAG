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
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_lib import GOLD as GOLD_BASE
from eval_lib import evaluate

EXTENDED_GOLD_PATH = Path(__file__).resolve().parent / "eval_gold_extended.json"
PROBE_GOLD_PATH = Path(__file__).resolve().parent / "eval_gold_probe.json"


def generate_extended_gold(n_target: int = 16, seed: int = 13) -> list[dict]:
    """+n_target gold queries drawn from corpus qa fields.

    Each question was written by the corpus LLM about a specific chunk of a
    specific page, so that page is the verified gold. Candidates are
    deduped by question prefix, spread across Manual/ScriptReference, and
    checked against the on-disk mirror.
    """
    from rag import ROOT, resolve_path
    from rag.store import CorpusStore

    banned = ("this chunk", "the chunk", "this section", "the provided",
              "the documentation", "this page", "the page")
    cands = []
    store = CorpusStore(resolve_path("corpus"))
    for rel, data in store.iterate_all():
        for ch in data.get("chunks") or []:
            for qa in ch.get("qa") or []:
                q = (qa.get("q") or "").strip()
                if not (24 < len(q) < 140 and q.endswith("?")):
                    continue
                if any(t in q.lower() for t in banned):
                    continue
                cands.append({"query": q, "gold": [rel]})

    rng = random.Random(seed)
    rng.shuffle(cands)
    seen_pref: set[str] = set()
    out: list[dict] = []
    n_manual = 0
    for c in cands:
        pref = " ".join(c["query"].lower().split()[:5])
        if pref in seen_pref:
            continue
        is_manual = c["gold"][0].startswith("Manual/")
        if is_manual and n_manual >= max(1, n_target // 2):
            continue
        if not (ROOT / c["gold"][0]).exists():
            continue
        seen_pref.add(pref)
        out.append(c)
        n_manual += int(is_manual)
        if len(out) >= n_target:
            break
    EXTENDED_GOLD_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
    print(f"[gold] wrote {len(out)} queries to {EXTENDED_GOLD_PATH}")
    return out


def load_gold(name: str):
    if name == "base":
        return GOLD_BASE, "base-24"
    if name == "ext":
        if not EXTENDED_GOLD_PATH.exists():
            raise SystemExit(f"{EXTENDED_GOLD_PATH} not found — generate it first")
        data = json.loads(EXTENDED_GOLD_PATH.read_text(encoding="utf-8"))
        return [(q["query"], q["gold"]) for q in data], f"ext-{len(data)}"
    if name == "probe":
        # probe-scoped semantic queries: gold pages guaranteed present in the
        # probe corpus, so this measures dense/hybrid value on paraphrase
        # questions rather than being dominated by pages that were not sampled.
        if not PROBE_GOLD_PATH.exists():
            raise SystemExit(f"{PROBE_GOLD_PATH} not found — generate it first")
        data = json.loads(PROBE_GOLD_PATH.read_text(encoding="utf-8"))
        return [(q["query"], q["gold"]) for q in data], f"probe-{len(data)}"
    if name == "all":
        ext, _ = load_gold("ext")
        return GOLD_BASE + ext, f"all-{len(GOLD_BASE) + len(ext)}"
    raise SystemExit(f"unknown gold set {name!r}")


# --------------------------------------------------------------------------------------
# ranking adapters
# --------------------------------------------------------------------------------------


def engine_ranker(engine, k: int, mode: str, *, rerank: bool | None = None):
    """Ranker over the loaded engine.

    ``rerank`` defaults to whatever the config says; pass False to force the
    cross-encoder off. It is honoured explicitly rather than hardcoding
    no_rerank=True — doing that made a "rerank=True" eval run silently identical
    to the rerank-off run, which would have reported a null result as a finding.
    """
    no_rerank = False if rerank is None else (not rerank)

    def rank(query: str) -> list[str]:
        out = engine.search(query, k=k, mode=mode, no_rerank=no_rerank)
        if not no_rerank and engine.cfg.get("rerank"):
            # fail loudly rather than quietly scoring the un-reranked ranking
            assert out.get("reranked"), "rerank was requested but not applied"
        return [h["source"] for h in out["hits"]]
    return rank


def table_ranker(rows, cfg, *, mode="hybrid", aux=True, path_boost=3, rrf_k=60,
                 per_file=1, fusion="rrf", alpha=1.0):
    """In-memory ranker over the chunk table with explicit ablation knobs.

    BM25 is rebuilt from *rows* so aux/path_boost can be ablated without touching
    artefacts. Dense vectors cannot be rebuilt cheaply, so they are read from the
    prebuilt ``vectors.f32`` — which is only valid when its row order matches
    *rows* exactly. That alignment is asserted below (uid-by-uid) because a
    mismatch would silently pair every BM25 doc id with the wrong vector.
    """
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
    idx_dir = Path(engine_index_dir(cfg))
    vecs_path = idx_dir / "vectors.f32"
    meta_path = idx_dir / "vector_meta.json"
    if mode in ("hybrid", "dense") and vecs_path.exists():
        from rag.index.vector_index import load_meta, load_vectors
        meta = load_meta(meta_path)
        count, dim = int(meta["count"]), int(meta["dim"])
        if count != len(rows):
            print(f"[eval] WARNING: vectors.f32 has {count} rows but the corpus "
                  f"flattened to {len(rows)} — index is stale. Dense disabled for "
                  f"this ablation; rebuild with `rag.py compile --steps index "
                  f"--force`.", file=sys.stderr)
        else:
            dense_mat = load_vectors(vecs_path, dim=dim, count=count)
            _assert_vector_alignment(idx_dir, rows)

    def rank(query: str) -> list[str]:
        bm25 = searcher.search(query, top_k=int(cfg.get("bm25_k", 200)))
        dense = []
        if dense_mat is not None:
            q = embed_query(query, model=cfg.get("embed_model", "BAAI/bge-m3"))
            scores = dense_mat @ q
            top = heapq.nlargest(int(cfg.get("dense_k", 200)), range(len(scores)),
                                 key=lambda i: (float(scores[i]), -i))
            dense = [(i, float(scores[i])) for i in top]
        if mode == "dense" or not dense:
            fused = dense if mode == "dense" else bm25
        elif fusion == "linear":
            # the legacy hybrid_search behaviour: min-max score fusion. This is
            # the ablation that shows WHY rank fusion is used -- BM25 scores are
            # unbounded and cosine is bounded, so a linear blend lets whichever
            # scale happens to be wider dominate.
            from rag.index.fuse import linear_fuse
            fused = linear_fuse(bm25, dense, alpha=alpha)
        else:
            fused = rrf_fuse(bm25, dense, k=rrf_k)
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


def _assert_vector_alignment(idx_dir: Path, rows, sample: int = 64) -> None:
    """Verify vectors.f32 row i is the embedding of rows[i].

    vectors.f32 stores no ids, so alignment is positional: it is only valid if
    the corpus has not changed since `compile --steps index` ran. Compare the
    built chunk table's uids against the freshly flattened rows; a mismatch means
    every dense score would be attributed to the wrong chunk.
    """
    import msgspec
    from rag.index.build import ChunkRow

    table_path = idx_dir / "chunks.msgpack"
    if not table_path.exists():
        print("[eval] WARNING: no chunks.msgpack to verify vector alignment; "
              "dense results may be misattributed", file=sys.stderr)
        return
    built = [ChunkRow(**r) for r in msgspec.msgpack.decode(table_path.read_bytes())]
    if len(built) != len(rows):
        raise RuntimeError(
            f"vector/chunk misalignment: index table has {len(built)} rows but the "
            f"corpus flattened to {len(rows)}. Rebuild the index "
            f"(`rag.py compile --steps index --force`) before evaluating dense.")
    step = max(1, len(rows) // sample)
    for i in range(0, len(rows), step):
        if built[i].chunk_uid != rows[i].chunk_uid:
            raise RuntimeError(
                f"vector/chunk misalignment at row {i}: index has "
                f"{built[i].chunk_uid} ({built[i].source}) but corpus has "
                f"{rows[i].chunk_uid} ({rows[i].source}). Rebuild the index.")
    print(f"[eval] vector alignment verified ({len(rows)} rows)", file=sys.stderr)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    ap.add_argument("--gold-set", choices=["base", "ext", "all", "probe"], default="all")
    ap.add_argument("--configs", default=None,
                    help="legacy compat: ignored, single engine evaluated per run")
    ap.add_argument("--sweep-aux", action="store_true")
    ap.add_argument("--sweep-path-boost", default=None, help="e.g. 0,1,3,5")
    ap.add_argument("--sweep-rrf-k", default=None, help="e.g. 20,40,60,120")
    ap.add_argument("--sweep-fusion", action="store_true",
                    help="compare rrf vs linear-alpha fusion on identical scores "
                         "(shows why RRF is the default: BM25 is unbounded, "
                         "cosine is bounded)")
    ap.add_argument("--per-file", type=int, default=None)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--sweep-rerank", action="store_true",
                    help="run the engine with the cross-encoder reranker off, "
                         "then on, in one process (requires rerank config + model)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--show-gold", action="store_true")
    ap.add_argument("--generate-gold", action="store_true",
                    help="regenerate eval_gold_extended.json from corpus qa fields, then exit")
    ap.add_argument("--config", default="rag_config.json",
                    help="rag config to read index_dir/corpus_dir from (use a "
                         "scratch config to evaluate a probe index without "
                         "disturbing the production one)")
    args = ap.parse_args()

    if args.generate_gold:
        generate_extended_gold()
        return 0

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

    cfg = load_rag_config(args.config)
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
                       verbose=args.verbose, gold_set=gold)
        res["gold"] = gold_name
        results.append(res)
        print(f"  {name:<28} MRR={res['MRR']:.3f} hit@1={res['hit@1']:.3f} "
              f"hit@10={res['hit@10']:.3f}")
        return res

    print(f"[eval] gold set: {gold_name} ({len(gold)} queries)")
    if args.sweep_aux:
        for aux in (True, False):
            run(f"table {args.mode} aux={aux}",
                table_ranker(rows, cfg, mode=args.mode, aux=aux))
        return emit(results, args)
    if args.sweep_path_boost:
        for pb in [int(x) for x in args.sweep_path_boost.split(",")]:
            run(f"table {args.mode} path_boost={pb}",
                table_ranker(rows, cfg, mode=args.mode, path_boost=pb))
        return emit(results, args)
    if args.sweep_rrf_k:
        for k in [int(x) for x in args.sweep_rrf_k.split(",")]:
            run(f"table {args.mode} rrf_k={k}",
                table_ranker(rows, cfg, mode=args.mode, rrf_k=k))
        return emit(results, args)
    if args.sweep_fusion:
        # identical BM25 + BGE-M3 scores, different fusion rules
        run("table hybrid fusion=rrf(k=60)",
            table_ranker(rows, cfg, mode="hybrid", fusion="rrf", rrf_k=60))
        for a in (0.3, 0.5, 0.7, 0.9):
            run(f"table hybrid fusion=linear(alpha={a})",
                table_ranker(rows, cfg, mode="hybrid", fusion="linear", alpha=a))
        return emit(results, args)
    if args.per_file is not None:
        run(f"table {args.mode} per_file={args.per_file}",
            table_ranker(rows, cfg, mode=args.mode, per_file=args.per_file))
        return emit(results, args)

    if args.sweep_rerank:
        cfg_for_engine = dict(cfg)
        cfg_for_engine["rerank"] = True  # engine_ranker's rerank= flag controls it
        engine.cfg = cfg_for_engine
        run(f"engine {args.mode} rerank=off",
            engine_ranker(engine, 10, args.mode, rerank=False))
        run(f"engine {args.mode} rerank=on",
            engine_ranker(engine, 10, args.mode, rerank=True))
        return emit(results, args)

    # standard engine path (dense uses the built vectors)
    run(f"engine mode={args.mode} rerank={cfg.get('rerank', False)}",
        engine_ranker(engine, 10, args.mode,
                      rerank=bool(cfg.get("rerank", False))))
    return emit(results, args)


def emit(results: list[dict], args) -> int:
    """Write the raw run output and report the gate verdict.

    Deliberately writes to ``eval_results_latest.md``, NOT ``eval_results.md``:
    the latter is the hand-maintained findings document (ablations, post-mortems,
    caveats) and an earlier version of this function overwrote it wholesale on
    every run. Copy the rows you want to keep into eval_results.md by hand.
    """
    path = Path(__file__).resolve().parent / "eval_results_latest.md"
    lines = [
        "# Retrieval evaluation results (latest raw run)",
        "",
        "Auto-generated by eval_rag.py — raw rows only. The curated findings,",
        "ablations and caveats live in eval_results.md; copy rows across by hand.",
        "",
        f"Run: `eval_rag.py {' '.join(a for a in sys.argv[1:])}`",
        "",
        "| config | gold | MRR | hit@1 | hit@10 |",
        "| --- | --- | --- | --- | --- |",
        "| old baseline (historical record) | base-24 | 0.875 | 0.833 | 0.917 |",
        "| old baseline (re-measured today) | base-24 | 0.833 | 0.750 | 0.917 |",
    ]
    for r in results:
        lines.append(f"| {r['config']} | {r['gold']} | {r['MRR']} | {r['hit@1']} | "
                     f"{r['hit@10']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] wrote {path.name} (curated findings live in eval_results.md)")

    # The gate is judged on the INDEPENDENT base-24 set only. ext/probe queries
    # were harvested from corpus qa fields which are themselves indexed in the
    # aux text, so passing them proves nothing (see eval_results.md).
    gate_rows = [r for r in results if r["gold"].startswith("base")] or results
    if not gate_rows[0]["gold"].startswith("base"):
        print("[eval] GATE: not evaluated — this run used a contaminated gold set "
              f"({gate_rows[0]['gold']}); rerun with --gold-set base")
        return 0
    passed = [r for r in gate_rows if r["MRR"] >= 0.875 and r["hit@10"] >= 0.917]
    if passed:
        print(f"[eval] GATE PASS: {passed[0]['config']} meets/exceeds the baseline "
              f"(MRR>=0.875 AND hit@10>=0.917)")
    else:
        best = max(gate_rows, key=lambda r: r["MRR"])
        print(f"[eval] GATE FAIL: best was {best['config']} at MRR={best['MRR']} "
              f"hit@10={best['hit@10']} — keep the old engine default until it "
              f"passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
