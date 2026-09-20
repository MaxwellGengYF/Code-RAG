"""Which engine should serve a query right now?

The RAG engine's index is built from ``corpus/``, which is generated page by page
(a full 44k-page build takes many hours). While that build is incomplete, the
legacy ``chunks.pkl``/``index_word.pkl`` artefacts — which cover the whole mirror
— retrieve strictly more pages. Silently delegating to a partial index would look
like a regression in coverage rather than one in ranking, so callers ask here and
get an explicit, reportable decision.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from rag import resolve_path
from rag.config import config_dir, data_path

#: fraction of mirrored pages that must have corpus coverage before the RAG index
#: is considered a drop-in replacement for the full-corpus legacy index
COMPLETENESS_FRACTION = 0.95


def count_html_pages(dirs: list[str], base: Path | None = None) -> int:
    """Fast *.html count under *dirs* (no hashing, unlike FileManager.scan).
    Relative dirs anchor at *base* (the config's directory by default)."""
    total = 0
    root = base or config_dir({})
    for d in dirs:
        dbase = root / d
        if not dbase.is_dir():
            continue
        for _dirpath, _dirnames, filenames in os.walk(dbase):
            total += sum(1 for f in filenames if f.endswith(".html"))
    return total


def rag_index_status(cfg: dict) -> dict:
    """Describe the RAG index: built? complete? how many chunks/pages?"""
    base = config_dir(cfg)
    index_dir = data_path(cfg, "index_dir", "index", base=base)
    corpus_dir = data_path(cfg, "corpus_dir", "corpus", base=base)
    manifest_path = index_dir / "manifest.json"
    out = {
        "built": False, "complete": False, "n_chunks": 0, "n_pages": 0,
        "html_pages": 0, "coverage": 0.0, "has_dense": False, "reason": "",
    }
    if not manifest_path.exists():
        out["reason"] = f"no {manifest_path.name} — index not built"
        return out
    try:
        im = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        out["reason"] = f"unreadable index manifest: {exc}"
        return out

    out["built"] = True
    out["n_chunks"] = int(im.get("n_chunks", 0))
    out["n_pages"] = int(im.get("n_pages", 0))
    out["has_dense"] = (index_dir / "vectors.f32").exists()
    html_pages = count_html_pages(cfg.get("dirs", []), base)
    out["html_pages"] = html_pages
    if html_pages:
        out["coverage"] = out["n_pages"] / html_pages
    if not out["n_chunks"]:
        out["reason"] = "index has 0 chunks"
    elif out["coverage"] >= COMPLETENESS_FRACTION:
        out["complete"] = True
        out["reason"] = (f"coverage {out['coverage']:.1%} of {html_pages} mirrored "
                         f"pages")
    else:
        out["reason"] = (f"corpus covers only {out['coverage']:.1%} of {html_pages} "
                         f"mirrored pages (needs >= {COMPLETENESS_FRACTION:.0%})")
    return out


def legacy_index_status() -> dict:
    """Whether the legacy artefacts (chunks.pkl + index_word.pkl) are present.

    Needed because auto-selection falls back to the legacy engine: on a fresh
    checkout NEITHER engine is built, and claiming "LEGACY would serve this
    query" would be wrong — the honest answer is that nothing has been built yet.
    """
    chunks = resolve_path("chunks.pkl")
    index = resolve_path("index_word.pkl")
    present = chunks.exists() and index.exists()
    return {
        "built": present,
        "missing": [p.name for p in (chunks, index) if not p.exists()],
    }


def choose_engine(cfg: dict, *, prefer: str | None = None) -> tuple[str, dict]:
    """Return ("rag"|"legacy", status) for a search request.

    ``prefer`` forces the choice ("rag"/"legacy") when the caller passes an
    explicit flag; otherwise the RAG engine is used only once its index is both
    built and complete. When neither engine is available the answer is still
    "legacy" (so the caller's own not-found handling runs and prints the rebuild
    hint), but ``status["nothing_built"]`` says so plainly.
    """
    status = rag_index_status(cfg)
    legacy = legacy_index_status()
    status["legacy"] = legacy
    status["nothing_built"] = not status["built"] and not legacy["built"]
    if prefer == "rag":
        return "rag", status
    if prefer == "legacy":
        return "legacy", status
    return ("rag" if status["complete"] else "legacy"), status
