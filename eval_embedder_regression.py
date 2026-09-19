"""SA-7 causal experiment: why did word+dense(hash) regress, and does BGE-M3 fix it?

The old engine measured MRR 0.875 (BM25 alone) -> 0.792 when the offline `hash`
embedder was fused in. This reproduces the mechanism on identical corpus content
(the probe subset), holding BM25 fixed and swapping ONLY the dense embedder:

  bm25-only                 baseline
  bm25 + hash  (linear a)   the old configuration that regressed
  bm25 + bge-m3 (linear a)  same fusion, real semantic embeddings
  bm25 + bge-m3 (rrf)       the shipped configuration

Also reports, per query, how many results each fused list DISPLACES from the
BM25-only top-10 — displacement of correct pages is the actual failure mode, and
it shows whether the embedder adds signal or just noise.

Usage: uv run python .kimix_cache/embedder_regression.py [gold-set]
"""
import heapq
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_lib import evaluate
from eval_rag import engine_index_dir, load_gold, _assert_vector_alignment
from rag.compile import load_rag_config
from rag.index.build import flatten_corpus
from rag.index.bm25_index import build_bm25, new_searcher
from rag.index.fuse import linear_fuse, rrf_fuse
from rag.index.vector_index import embed_query, load_meta, load_vectors
from rag.store import CorpusStore
from rag import resolve_path

GOLD_SET = sys.argv[1] if len(sys.argv) > 1 else "base"


def hash_embed(texts: list[str], dim: int = 1024) -> np.ndarray:
    """The old offline HashEmbedder: character-trigram signature, NOT semantic."""
    import xxhash
    from retrieval import NgramTokenizer

    out = np.zeros((len(texts), dim), dtype=np.float32)
    tok = NgramTokenizer(n=3)
    for i, t in enumerate(texts):
        v = out[i]
        for gram in tok.tokenize(NgramTokenizer.normalize(t)):
            h = xxhash.xxh64(gram.encode("utf-8")).intdigest()
            for j in range(min(8, dim)):
                idx = int((h + j * 0x9E3779B97F4A7C15) % dim)
                v[idx] += 1.0 if (h >> j) & 1 else -1.0
        n = np.linalg.norm(v)
        if n:
            v /= n
    return out


def main():
    cfg = load_rag_config("rag_probe_config.json")
    gold, gold_name = load_gold(GOLD_SET)
    rows, n_pages = flatten_corpus(CorpusStore(resolve_path(cfg["corpus_dir"])))
    print(f"[probe] {len(rows)} chunks / {n_pages} pages | gold={gold_name}")

    chunks = [r.to_corpus_chunk() for r in rows]
    sources = [r.source for r in rows]
    titles = [r.title for r in rows]
    index, _ = build_bm25(chunks, sources, titles, verbose=False)
    searcher = new_searcher(index)

    idx_dir = Path(engine_index_dir(cfg))
    meta = load_meta(idx_dir / "vector_meta.json")
    bge = load_vectors(idx_dir / "vectors.f32", dim=meta["dim"], count=meta["count"])
    _assert_vector_alignment(idx_dir, rows)
    hsh = hash_embed([c.text[:4000] for c in chunks], dim=meta["dim"])
    print(f"[probe] bge-m3 {bge.shape} vs hash {hsh.shape}")

    # evaluate() resolves a returned id via chunks[i].source, so any row index
    # belonging to that source works.
    src_id = {}
    for i, s in enumerate(sources):
        src_id.setdefault(s, i)
    BM25_K, DENSE_K, PER_FILE = 200, 200, 1

    def dedupe(fused):
        seen, out = {}, []
        for d, _s in fused:
            src = sources[d]
            if seen.get(src, 0) >= PER_FILE:
                continue
            seen[src] = seen.get(src, 0) + 1
            out.append(src)
        return out

    def rank_bm25(q):
        return [src_id[s] for s in dedupe(searcher.search(q, top_k=BM25_K))[:10]]

    def dense_hits(mat, q_emb):
        scores = mat @ q_emb
        top = heapq.nlargest(DENSE_K, range(len(scores)),
                            key=lambda i: (float(scores[i]), -i))
        return [(i, float(scores[i])) for i in top]

    def rank_fused(q, mat, fuse):
        bm25 = searcher.search(q, top_k=BM25_K)
        dense = dense_hits(mat, fuse["embed"](q))
        fused = fuse["fn"](bm25, dense)
        return [src_id[s] for s in dedupe(fused)[:10]]

    def q_emb_bge(q):
        return embed_query(q, model=cfg.get("embed_model", "BAAI/bge-m3"))

    def q_emb_hash(q):
        return hash_embed([q], dim=meta["dim"])[0]

    configs = {
        "bm25 only": lambda q: rank_bm25(q),
        "bm25+hash  linear a=0.7 (OLD)": lambda q: rank_fused(
            q, hsh, {"fn": lambda b, d: linear_fuse(b, d, alpha=0.7),
                     "embed": q_emb_hash}),
        "bm25+bge   linear a=0.7": lambda q: rank_fused(
            q, bge, {"fn": lambda b, d: linear_fuse(b, d, alpha=0.7),
                     "embed": q_emb_bge}),
        "bm25+bge   rrf k=60 (SHIPPED)": lambda q: rank_fused(
            q, bge, {"fn": lambda b, d: rrf_fuse(b, d, k=60),
                     "embed": q_emb_bge}),
    }

    results = {}
    for name, fn in configs.items():
        res = evaluate(name, rows, fn, gold_set=gold)
        results[name] = res
        print(f"  {name:<32} MRR={res['MRR']:.3f} hit@1={res['hit@1']:.3f} "
              f"hit@10={res['hit@10']:.3f}")

    # displacement analysis: how much does each dense config move the BM25 top-10?
    print("\n[displacement] pages pushed out of the BM25-only top-10, and whether "
          "the gold page survived:")
    for name, fn in list(configs.items())[1:]:
        moved, gold_lost, gold_gained = 0, 0, 0
        for q, golds in gold:
            base_ids = rank_bm25(q)
            new_ids = fn(q)
            base_src = {rows[i].source for i in base_ids}
            new_src = {rows[i].source for i in new_ids}
            moved += len(base_src ^ new_src) // 2
            base_ok = any(any(g in s for g in golds) for s in base_src)
            new_ok = any(any(g in s for g in golds) for s in new_src)
            if base_ok and not new_ok:
                gold_lost += 1
            if new_ok and not base_ok:
                gold_gained += 1
        print(f"  {name:<32} avg displaced={moved/len(gold):.2f} "
              f"gold lost={gold_lost} gold gained={gold_gained}")
    return results


if __name__ == "__main__":
    main()
