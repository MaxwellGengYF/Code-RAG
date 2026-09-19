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


def build_vectors_resumable(
    chunks: list[CorpusChunk],
    titles: list[str],
    *,
    out_path: str | Path,
    model: str = "BAAI/bge-m3",
    batch_size: int = 32,
    dim: int = 1024,
    stamp: str = "",
    progress_every: int = 50,
    log=print,
) -> np.ndarray:
    """Embed every chunk to *out_path*, resuming after an interruption.

    Embedding the full corpus takes hours on CPU, so an in-memory build would
    restart from zero whenever the process died. Instead the matrix is written as
    a float32 memmap in row order and flushed every batch: rows are positional and
    the chunk order is deterministic, so the first N valid rows on disk are
    exactly the first N chunks and a rerun continues at row N.

    ``stamp`` identifies the chunk table the rows belong to (e.g. its sha1). It is
    written to ``<out_path>.stamp``; a partial file whose stamp differs is
    DISCARDED rather than resumed, because continuing it would splice embeddings
    of a stale chunk ordering onto the new one — vectors carry no ids, so such a
    mismatch is otherwise silent and would corrupt every dense score.

    A partial file is distinguished from a complete one by size: a complete
    matrix is exactly ``len(chunks) * dim * 4`` bytes.
    """
    out_path = Path(out_path)
    stamp_path = Path(str(out_path) + ".stamp")
    n = len(chunks)
    if n == 0:
        raise ValueError("build_vectors_resumable: no chunks to embed")

    expected_bytes = n * dim * 4
    start = 0
    if out_path.exists():
        have = out_path.stat().st_size
        prev_stamp = (stamp_path.read_text(encoding="utf-8").strip()
                      if stamp_path.exists() else "")
        if stamp and prev_stamp != stamp:
            log(f"[vectors] discarding partial {out_path.name}: it was embedded "
                f"for a different chunk table (stamp {prev_stamp[:8]!r} != "
                f"{stamp[:8]!r}); resuming it would misalign every row")
            out_path.unlink()
            stamp_path.unlink(missing_ok=True)
        elif have >= expected_bytes:
            log(f"[vectors] {out_path.name} already complete ({n} rows); reusing")
            return np.memmap(out_path, dtype=np.float32, mode="r", shape=(n, dim))
        else:
            start = have // (dim * 4)  # only whole rows count
            if start:
                log(f"[vectors] resuming at row {start}/{n} "
                    f"({start / n:.0%} already embedded)")

    # grow/extend the file to its final size so the memmap is addressable
    with open(out_path, "ab") as fh:
        fh.truncate(expected_bytes)
    if stamp:
        stamp_path.write_text(stamp, encoding="utf-8")
    mat = np.memmap(out_path, dtype=np.float32, mode="r+", shape=(n, dim))

    m = ensure_embed_model(model)
    texts = [to_embed_text(c, titles[i]) for i, c in enumerate(chunks)]
    for lo in range(start, n, batch_size):
        hi = min(lo + batch_size, n)
        emb = m.encode(texts[lo:hi], batch_size=batch_size,
                       normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False)
        mat[lo:hi] = np.asarray(emb, dtype=np.float32)
        mat.flush()  # durable: this is what makes the resume safe
        if progress_every and (hi - start) % (batch_size * progress_every) < batch_size:
            log(f"[vectors] {hi}/{n} ({hi / n:.0%})")
    mat.flush()
    return mat


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
