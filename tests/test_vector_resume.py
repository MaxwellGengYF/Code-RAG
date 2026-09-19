"""Resumable dense-embedding tests (network-free: fake embed model).

The full dense build takes hours on CPU, so it must survive interruption. These
tests cover the resume math and, critically, the stamp guard that prevents
resuming a partial file whose rows were embedded for a DIFFERENT chunk table —
vectors carry no ids, so such a mismatch would silently corrupt every score.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rag.corpus.schema import CorpusChunk
import rag.index.vector_index as vi


class FakeModel:
    """Deterministic 'embeddings': a vector derived from the text itself."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts, batch_size=32, normalize_embeddings=True,
               convert_to_numpy=True, show_progress_bar=False):
        self.calls.append(list(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t[:self.dim]):
                out[i, j] = float(ord(ch)) / 1000.0
            out[i, -1] = float(len(t))
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return out / n


def make_chunks(n: int, prefix: str = "chunk") -> tuple[list[CorpusChunk], list[str]]:
    chunks = [CorpusChunk(chunk_uid=f"{prefix}{i}", heading_path=["H"],
                          text=f"text body number {i} for testing")
              for i in range(n)]
    titles = [f"Title {i}" for i in range(n)]
    return chunks, titles


@pytest.fixture()
def fake_model(monkeypatch):
    model = FakeModel(dim=8)
    monkeypatch.setattr(vi, "ensure_embed_model", lambda name: model)
    return model


def test_builds_all_rows(fake_model, tmp_path):
    chunks, titles = make_chunks(20)
    out = tmp_path / "vectors.f32"
    mat = vi.build_vectors_resumable(
        chunks, titles, out_path=out, dim=8, batch_size=4, stamp="sha-abc",
        progress_every=0, log=lambda *_: None)
    assert mat.shape == (20, 8)
    assert out.stat().st_size == 20 * 8 * 4
    # rows are L2-normalized
    norms = np.linalg.norm(np.asarray(mat), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert (tmp_path / "vectors.f32.stamp").read_text() == "sha-abc"


def test_resume_skips_completed_rows(fake_model, tmp_path):
    """Interrupting mid-build then rerunning must not re-embed finished rows."""
    chunks, titles = make_chunks(20)
    out = tmp_path / "vectors.f32"

    # first pass: stop after 8 rows by faking a truncated file
    vi.build_vectors_resumable(chunks[:8], titles[:8], out_path=out, dim=8,
                               batch_size=4, stamp="sha-abc", progress_every=0,
                               log=lambda *_: None)
    assert out.stat().st_size == 8 * 8 * 4
    before = len(fake_model.calls)

    # now build the full 20 with the same stamp: rows 0-7 must be reused
    mat = vi.build_vectors_resumable(chunks, titles, out_path=out, dim=8,
                                     batch_size=4, stamp="sha-abc",
                                     progress_every=0, log=lambda *_: None)
    assert mat.shape == (20, 8)
    embedded = [t for call in fake_model.calls[before:] for t in call]
    assert len(embedded) == 12, f"expected only the 12 remaining rows, got {len(embedded)}"
    assert "text body number 0 for testing" not in " ".join(embedded)


def test_resume_reports_progress(fake_model, tmp_path):
    chunks, titles = make_chunks(12)
    out = tmp_path / "vectors.f32"
    vi.build_vectors_resumable(chunks[:4], titles[:4], out_path=out, dim=8,
                               batch_size=4, stamp="s", progress_every=0,
                               log=lambda *_: None)
    msgs = []
    vi.build_vectors_resumable(chunks, titles, out_path=out, dim=8,
                               batch_size=4, stamp="s", progress_every=0,
                               log=msgs.append)
    assert any("resuming at row 4/12" in m for m in msgs), msgs


def test_complete_file_is_reused_without_embedding(fake_model, tmp_path):
    chunks, titles = make_chunks(8)
    out = tmp_path / "vectors.f32"
    vi.build_vectors_resumable(chunks, titles, out_path=out, dim=8,
                               batch_size=4, stamp="s", progress_every=0,
                               log=lambda *_: None)
    calls_before = len(fake_model.calls)
    mat = vi.build_vectors_resumable(chunks, titles, out_path=out, dim=8,
                                     batch_size=4, stamp="s", progress_every=0,
                                     log=lambda *_: None)
    assert mat.shape == (8, 8)
    assert len(fake_model.calls) == calls_before, "complete file must not re-embed"


def test_stamp_mismatch_discards_partial(fake_model, tmp_path):
    """A partial file from a DIFFERENT chunk table must be rebuilt, not resumed.

    Resuming it would splice stale rows onto new ones; since vectors carry no ids
    the corruption would be invisible at query time.
    """
    chunks_a, titles_a = make_chunks(10, prefix="a")
    chunks_b, titles_b = make_chunks(10, prefix="b")
    out = tmp_path / "vectors.f32"

    # partially embed table A
    vi.build_vectors_resumable(chunks_a[:6], titles_a[:6], out_path=out, dim=8,
                               batch_size=2, stamp="sha-A", progress_every=0,
                               log=lambda *_: None)
    assert out.stat().st_size == 6 * 8 * 4

    msgs = []
    calls_before = len(fake_model.calls)
    mat = vi.build_vectors_resumable(chunks_b, titles_b, out_path=out, dim=8,
                                     batch_size=2, stamp="sha-B",
                                     progress_every=0, log=msgs.append)
    assert any("discarding partial" in m for m in msgs), msgs
    assert mat.shape == (10, 8)
    # all 10 rows of table B were embedded from scratch
    embedded = [t for call in fake_model.calls[calls_before:] for t in call]
    assert len(embedded) == 10
    assert (tmp_path / "vectors.f32.stamp").read_text() == "sha-B"


def test_missing_stamp_treated_as_mismatch(fake_model, tmp_path):
    """No stamp file = unknown provenance = rebuild rather than trust it."""
    chunks, titles = make_chunks(6)
    out = tmp_path / "vectors.f32"
    vi.build_vectors_resumable(chunks[:3], titles[:3], out_path=out, dim=8,
                               batch_size=3, stamp="sha-old", progress_every=0,
                               log=lambda *_: None)
    stamp_path = Path(str(out) + ".stamp")
    assert stamp_path.exists()
    stamp_path.unlink()  # simulate a stamp lost to an older build / manual edit
    msgs = []
    vi.build_vectors_resumable(chunks, titles, out_path=out, dim=8,
                               batch_size=3, stamp="sha-new", progress_every=0,
                               log=msgs.append)
    assert any("discarding partial" in m for m in msgs), msgs


def test_embed_text_excludes_aux_fields(tmp_path):
    """Regression guard for the 0.875->0.792 lesson: dense input must stay clean.

    Synthetic aux text (summary/keywords/synonyms/qa) is allowed to boost BM25
    lexical recall but must never enter the vector space.
    """
    from rag.corpus.schema import QA, to_embed_text
    ch = CorpusChunk(chunk_uid="u", heading_path=["Rigidbody", "Properties"],
                     text="The velocity of the rigidbody.",
                     summary="SYNTHETIC SUMMARY TOKENS",
                     keywords=["zzzkeywordzzz"], synonyms=["zzzsynonymzzz"],
                     qa=[QA(q="zzzquestionzzz", a="zzzanswerzzz")])
    emb_text = to_embed_text(ch, title="Rigidbody")
    assert "The velocity of the rigidbody." in emb_text
    for banned in ("SYNTHETIC", "zzzkeywordzzz", "zzzsynonymzzz",
                   "zzzquestionzzz", "zzzanswerzzz"):
        assert banned not in emb_text, f"aux field {banned!r} leaked into embed text"


def test_empty_chunks_rejected(tmp_path):
    with pytest.raises(ValueError):
        vi.build_vectors_resumable([], [], out_path=tmp_path / "v.f32", dim=8)
