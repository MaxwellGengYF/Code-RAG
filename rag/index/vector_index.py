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
_MODEL_DEVICE: str | None = None


def select_embed_device() -> str:
    """cuda when a GPU is usable, else cpu. The model dtype follows the device:
    fp16 halves GPU memory and roughly doubles throughput (BGE-M3 on a 4080
    embeds the full corpus in minutes, not hours); fp32 stays the CPU default."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def embed_device() -> str:
    """Device the loaded model runs on ("" before the first load)."""
    return _MODEL_DEVICE or ""


def ensure_embed_model(model: str = "BAAI/bge-m3"):
    """Download/load the embed model (used by the deps step; raises on failure)."""
    global _MODEL, _MODEL_NAME, _MODEL_DEVICE
    import torch
    from sentence_transformers import SentenceTransformer

    if _MODEL is None or _MODEL_NAME != model:
        device = select_embed_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        _MODEL = SentenceTransformer(model, device=device,
                                     model_kwargs={"dtype": dtype})
        _MODEL_NAME = model
        _MODEL_DEVICE = device
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
    restart from zero whenever the process died. Rows are positional and the chunk
    order is deterministic, so the first N embedded rows on disk are exactly the
    first N chunks and a rerun can continue at row N.

    Progress is tracked in a sidecar (``<out_path>.progress``: the chunk-table
    stamp plus the number of COMPLETE rows), never inferred from file size. The
    file is pre-allocated to its full length so the memmap is addressable, which
    means size says nothing about how much was actually embedded — an interrupted
    run leaves a full-size file whose tail is zero-filled. Trusting size here
    would silently ship thousands of zero vectors and corrupt every dense score.

    A partial file whose stamp differs from *stamp* is DISCARDED rather than
    resumed: vectors carry no ids, so splicing rows embedded for a different chunk
    ordering onto the new one would misalign every row, invisibly.
    """
    out_path = Path(out_path)
    stamp_path = Path(str(out_path) + ".stamp")
    progress_path = Path(str(out_path) + ".progress")
    n = len(chunks)
    if n == 0:
        raise ValueError("build_vectors_resumable: no chunks to embed")

    expected_bytes = n * dim * 4
    start = _resume_row(out_path, stamp_path, progress_path, stamp=stamp,
                        n=n, dim=dim, expected_bytes=expected_bytes, log=log)

    # grow/extend the file to its final size so the memmap is addressable
    with open(out_path, "ab") as fh:
        fh.truncate(expected_bytes)
    if stamp:
        stamp_path.write_text(stamp, encoding="utf-8")
    mat = np.memmap(out_path, dtype=np.float32, mode="r+", shape=(n, dim))

    if start >= n:
        log(f"[vectors] {out_path.name} already complete ({n} rows); reusing")
        return mat

    m = ensure_embed_model(model)
    texts = [to_embed_text(c, titles[i]) for i, c in enumerate(chunks)]
    for lo in range(start, n, batch_size):
        hi = min(lo + batch_size, n)
        emb = m.encode(texts[lo:hi], batch_size=batch_size,
                       normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False)
        mat[lo:hi] = np.asarray(emb, dtype=np.float32)
        mat.flush()  # durable: this is what makes the resume safe
        # record progress only AFTER the rows are on disk, so a kill between the
        # two leaves the counter low (re-embedding a few rows) rather than high
        # (claiming zero vectors as real embeddings).
        progress_path.write_text(f"{stamp}\n{hi}\n", encoding="utf-8")
        if progress_every and (hi - start) % (batch_size * progress_every) < batch_size:
            log(f"[vectors] {hi}/{n} ({hi / n:.0%})")
    mat.flush()
    progress_path.write_text(f"{stamp}\n{n}\n", encoding="utf-8")
    return mat


def _resume_row(out_path: Path, stamp_path: Path, progress_path: Path, *,
                stamp: str, n: int, dim: int, expected_bytes: int, log) -> int:
    """Row to resume from; 0 when the existing file cannot be trusted."""
    if not out_path.exists():
        return 0
    have = out_path.stat().st_size

    # A pre-allocation that was never embedded into has no progress sidecar.
    done_rows = -1
    if progress_path.exists():
        try:
            parts = progress_path.read_text(encoding="utf-8").split()
            prev_stamp = parts[0] if parts else ""
            done_rows = int(parts[1]) if len(parts) > 1 else -1
            if stamp and prev_stamp != stamp:
                log(f"[vectors] discarding partial {out_path.name}: embedded for a "
                    f"different chunk table (stamp {prev_stamp[:8]!r} != "
                    f"{stamp[:8]!r}); resuming would misalign every row")
                _remove_vector_artifacts(out_path, stamp_path, progress_path)
                return 0
        except (ValueError, OSError):
            done_rows = -1

    if done_rows < 0:
        # No trustworthy progress record. Only a legacy full-size file WITHOUT a
        # progress sidecar could be complete, and we cannot verify that, so treat
        # it as unusable rather than risk shipping zero vectors.
        if have >= expected_bytes:
            log(f"[vectors] discarding full-size {out_path.name} with no progress "
                f"record: cannot tell embedded rows from pre-allocated zeros")
        _remove_vector_artifacts(out_path, stamp_path, progress_path)
        return 0

    if done_rows >= n and have >= expected_bytes:
        return n
    if done_rows * dim * 4 > have:  # claims more than the file holds
        log(f"[vectors] progress record ({done_rows} rows) exceeds file size; "
            f"rebuilding")
        _remove_vector_artifacts(out_path, stamp_path, progress_path)
        return 0
    if done_rows:
        log(f"[vectors] resuming at row {done_rows}/{n} "
            f"({done_rows / n:.0%} already embedded)")
    return max(done_rows, 0)


def _remove_vector_artifacts(*paths: Path) -> None:
    for p in paths:
        try:
            p.unlink()
        except OSError:
            pass


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
