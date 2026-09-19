"""Fusion: RRF (default) + legacy linear-alpha (ablation only).

BM25 scores are unbounded while cosine is bounded, so ranks fuse far more
robustly than scores: ``rrf(d) = sum_i 1 / (k + rank_i(d))`` with k=60.

All functions take two rank lists — ``bm25`` and ``dense`` — each
``[(doc_id, score)]`` in DESCENDING score order, and return ``[(doc_id, fused)]``
descending. Missing-from-one-list candidates still contribute via the other list.
"""
from __future__ import annotations

RRF_K = 60


def rrf_fuse(
    bm25: list[tuple[int, float]],
    dense: list[tuple[int, float]],
    *,
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """Reciprocal-rank fusion of two descending rank lists."""
    scores: dict[int, float] = {}
    for rank, (doc_id, _s) in enumerate(bm25):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, (doc_id, _s) in enumerate(dense):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def _minmax(scores: list[tuple[int, float]]) -> dict[int, float]:
    vals = [s for _d, s in scores]
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {d: 1.0 for d, _s in scores}
    return {d: (s - lo) / (hi - lo) for d, s in scores}


def linear_fuse(
    bm25: list[tuple[int, float]],
    dense: list[tuple[int, float]],
    *,
    alpha: float = 1.0,
) -> list[tuple[int, float]]:
    """Legacy min-max linear fusion (the old hybrid_search behaviour). Ablation only."""
    bnorm = _minmax(bm25)
    dnorm = _minmax(dense)
    fused: dict[int, float] = {}
    for d, _s in bm25:
        fused[d] = alpha * bnorm[d] + (1.0 - alpha) * dnorm.get(d, 0.0)
    for d, _s in dense:
        if d not in fused:
            fused[d] = alpha * bnorm.get(d, 0.0) + (1.0 - alpha) * dnorm[d]
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
