"""Optional cross-encoder rerank (BAAI/bge-reranker-v2-m3). Default OFF.

Lazy import + lazy model load: the reranker is only pulled in when the config
enables it, so BM25-only users never pay the download.
"""
from __future__ import annotations

_MODEL = None
_MODEL_NAME: str | None = None


def _get_model(model: str = "BAAI/bge-reranker-v2-m3"):
    global _MODEL, _MODEL_NAME
    if _MODEL is None or _MODEL_NAME != model:
        from sentence_transformers import CrossEncoder

        _MODEL = CrossEncoder(model, device="cpu")
        _MODEL_NAME = model
    return _MODEL


def rerank_pairs(pairs: list[tuple[str, str]], *, model: str) -> list[float]:
    """Relevance scores for (query, passage) pairs."""
    m = _get_model(model)
    return [float(s) for s in m.predict(list(pairs))]
