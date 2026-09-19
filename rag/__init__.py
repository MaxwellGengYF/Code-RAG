"""LLM-built RAG corpus + hybrid retrieval for the Unity Manual mirror.

Two-command system:
    rag.py compile  -- LLM-generated corpus -> indexes -> deps (md5-incremental)
    rag.py search   -- query -> ranked results (BM25 + dense, RRF-fused)

Subpackages:
    rag.llm     -- vendored, tool-free LLM backend (openai_legacy / openai_responses /
                   anthropic / kimi providers driven by a provider-config JSON)
    rag.corpus  -- page extraction (HTML -> markdown) + LLM corpus generation
    rag.store   -- md5 file manager + corpus store (incremental regeneration memory)
    rag.index   -- BM25 + dense index builders, RRF fusion
    rag.search  -- search engine + rerank + CLI support
    rag.cli     -- compile / search / status commands
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("UNITY_DOCS_ROOT", "")).resolve() \
    if os.environ.get("UNITY_DOCS_ROOT") else Path(__file__).resolve().parent.parent


def resolve_path(p: str | Path, base: Path | None = None) -> Path:
    """Absolute path for *p*; relative paths anchor at the repo root."""
    p = Path(p)
    return p if p.is_absolute() else ((base or ROOT) / p).resolve()


def rel_source(path: str | Path) -> str:
    """Posix path relative to ROOT when possible (stable, display-friendly source ids)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return p.as_posix()


__version__ = "0.1.0"
