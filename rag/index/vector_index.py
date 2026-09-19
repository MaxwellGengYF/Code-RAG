"""Dense index (BGE-M3) over CLEAN chunk text only — vectors.f32 + vector_meta.json.

Aux fields are synthetic; they must never pollute the vector space (the old
word+dense(hash) regression, MRR 0.875 -> 0.792, is the cautionary tale), so
embedding input is ``to_embed_text`` = title + heading_path + text.

BGE-M3 query convention: instruction prefix on the query, no prefix on passages.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag.corpus.schema import CorpusChunk, to_embed_text

BGE_QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

_MODEL = None
_MODEL_NAME: str | None = None


def ensure_embed_model(model: str = "BAAI/bge-m3"):
    """Download/load the embed model (used by the deps step; raises on failure)."""
    global _MODEL, _MODEL_NAME
    from sentence_transformers import SentenceTransformer

    if _MODEL is None or _MODEL_NAME != model:
        _MODEL = SentenceTransformer(model, device="cpu")
        _MODEL_NAME = model
    return _MODEL


def embed_texts(texts: list[str], *, batch_size: int = 32,
                model: str = "BAAI/bge-m3", show_progress: bool = False) -> np.ndarray:
    """(n, dim) L2-normalized float32 matrix."""
    m = ensure_embed_model(model)
    emb = m.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return np.asarray(emb, dtype=np.float32)


def embed_query(query: str, *, model: str = "BAAI/bge-m3") -> np.ndarray:
    m = ensure_embed_model(model)
    emb = m.encode([BGE_QUERY_INSTRUCTION + query], convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)[0]


def build_vectors(
    chunks: list[CorpusChunk],
    titles: list[str],
    *,
    model: str = "BAAI/bge-m3",
    batch_size: int = 32,
) -> np.ndarray:
    texts = [to_embed_text(c, titles[i]) for i, c in enumerate(chunks)]
    return embed_texts(texts, batch_size=batch_size, model=model,
                       show_progress=True)


def save_vectors(arr: np.ndarray, path: str | Path) -> None:
    np.asarray(arr, dtype=np.float32).tofile(str(path))


def load_vectors(path: str | Path, *, dim: int, count: int) -> np.memmap:
    return np.memmap(str(path), dtype=np.float32, mode="r", shape=(count, dim))


def save_meta(path: str | Path, *, model: str, dim: int, count: int,
              extra: dict | None = None) -> None:
    meta = {"model": model, "dim": dim, "count": count,
            "normalized": True, **(extra or {})}
    Path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_meta(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
