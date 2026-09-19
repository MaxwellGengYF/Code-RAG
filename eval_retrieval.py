"""CLI: sweep retriever configurations against the gold set in eval_lib.py.

    uv run python eval_retrieval.py                        # legacy vs word vs current default
    uv run python eval_retrieval.py --configs legacy,word
    uv run python eval_retrieval.py --verbose              # per-query hit/miss
    uv run python eval_retrieval.py --show-gold
    uv run python eval_retrieval.py --misses               # detail for non-rank-1 queries
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from retrieval import InvertedIndex, Searcher

import eval_lib as L
from hybrid_retrieve import HashEmbedder, get_tokenizer, load_chunks_pickle

CONFIG_PATH = "retriever_config.json"

# name -> (index file, tokenizer, fuzziness, min_should_match, alpha, per_file)
CONFIGS: dict[str, tuple] = {
    "legacy":       ("index.pkl",      "ngram", "AUTO", 0.5, 0.7, 0),
    "nofuzz":       ("index.pkl",      "ngram", 0,      0.5, 0.7, 0),
    "ngram-dedup":  ("index.pkl",      "ngram", 0,      0.5, 0.7, 1),
    "word":         ("index_word.pkl", "word",  0,      0.6, 1.0, 1),
    "word-nodedup": ("index_word.pkl", "word",  0,      0.6, 1.0, 0),
    "word-nopath":  ("index_word_body.pkl", "word", 0, 0.6, 1.0, 1),
    "word+dense":   ("index_word.pkl", "word",  0,      0.6, 0.7, 1),
}

HDR = (f"{'config':<14}{'index':<20}{'tok':<6}{'fz':<6}{'smm':<6}"
       f"{'MRR':>8}{'h@1':>7}{'h@3':>7}{'h@5':>7}{'h@10':>8}{'r@10':>7}")


def load_cfg() -> dict:
    from hybrid_retrieve import load_config
    cfg, _ = load_config(CONFIG_PATH)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="legacy,nofuzz,ngram-dedup,word,word-nodedup,default")
    ap.add_argument("--final-k", type=int, default=10)
    ap.add_argument("--bm25-k", type=int, default=200)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--misses", action="store_true")
    ap.add_argument("--show-gold", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()

    if a.show_gold:
        for q, g in L.GOLD:
            print(f"{q!r}")
            print(f"    -> {g}")
        return 0

    cfg = load_cfg()
    print(f"loading {cfg['chunks_path']} ...", file=sys.stderr, flush=True)
    chunks = load_chunks_pickle(cfg["chunks_path"])
    print(f"{len(chunks)} chunks", file=sys.stderr, flush=True)

    emb: HashEmbedder | None = None
    results = []
    print(HDR)
    print("-" * len(HDR))

    for name in [c.strip() for c in a.configs.split(",") if c.strip()]:
        if name == "default":
            idx_f, tokn = cfg["index_path"], cfg["tokenizer"]
            fz, smm, alpha, pf = cfg["fuzziness"], cfg["min_should_match"], cfg["alpha"], cfg["per_file"]
        elif name in CONFIGS:
            idx_f, tokn, fz, smm, alpha, pf = CONFIGS[name]
        else:
            print(f"unknown config {name!r}", file=sys.stderr)
            continue

        if not Path(idx_f).exists():
            print(f"{name:<14}SKIP (missing {idx_f})", file=sys.stderr)
            continue

        # Load a FRESH index per config: Searcher(fuzziness="AUTO") mutates the shared
        # InvertedIndex (it builds a symmetric-delete index on it), so reusing one object
        # across configs makes the numbers depend on evaluation order.
        i = InvertedIndex()
        i.load(idx_f)
        searcher = Searcher(i, tokenizer=get_tokenizer(tokn),
                            fuzziness=fz, min_should_match=smm)
        if alpha < 1.0 and emb is None:
            emb = HashEmbedder()
        ed = emb if alpha < 1.0 else None

        def run(q, s=searcher, ed=ed, alpha=alpha, pf=pf):
            return L.hybrid_rank(s, chunks, ed, q, bm25_k=a.bm25_k,
                                 final_k=a.final_k, alpha=alpha,
                                 per_file=(pf or None))

        r = L.evaluate(name, chunks, run, verbose=a.verbose)
        results.append({**r, "index": idx_f, "tokenizer": tokn})
        print(f"{name:<14}{Path(idx_f).name:<20}{tokn:<6}{str(fz):<6}{smm:<6}"
              f"{r['MRR']:>8}{r['hit@1']:>7}{r['hit@3']:>7}{r['hit@5']:>7}"
              f"{r['hit@10']:>8}{r['recall@10']:>7}", flush=True)

        if a.misses:
            for q, gold in L.GOLD:
                ids = run(q)
                paths = [Path(chunks[d].source).as_posix() for d in ids]
                rank = next((k + 1 for k, p in enumerate(paths)
                             if any(g in p for g in gold)), None)
                if rank is None or rank > 1:
                    print(f"    [{'MISS' if rank is None else f'r{rank}'}] "
                          f"{q[:52]:<52} -> {[Path(p).name for p in paths[:2]]}")

    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {a.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
