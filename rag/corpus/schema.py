"""Corpus schema: per-page JSON contract (msgspec structs) + index-text helpers.

Versioning contract (SA-3 file manager): ``gen_key`` =
sha1(PROMPT_VERSION | model | EXTRACTOR_VERSION | SCHEMA_VERSION). Bump
SCHEMA_VERSION on any incompatible change to the structs below.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

import msgspec

from rag.corpus.prompts import PROMPT_VERSION

SCHEMA_VERSION = "v1"
EXTRACTOR_VERSION = "v1"  # bump when rag/corpus/extract.py rendering changes

_WS_RE = re.compile(r"\s+")
# The markdown renderer inserts spaces around punctuation where HTML uses inline
# spans (e.g. the heading "WheelFrictionCurve .extremumSlip"). Models copy the
# "natural" dotted form, so the verbatim assertion compares with spacing around
# punctuation removed on BOTH sides.
_PUNCT_WS_RE = re.compile(r"\s*([.,;:()\[\]{}])\s*")
#: models routinely retype curly quotes as straight ones; the verbatim assertion
#: is typography-insensitive on both sides for exactly this reason
_QUOTE_TRANSLATION = str.maketrans(
    {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
     "\u00ab": '"', "\u00bb": '"'})


def norm_ws(text: str) -> str:
    """Whitespace-collapsed text."""
    return _WS_RE.sub(" ", text).strip()


def norm_verbatim(text: str) -> str:
    """Whitespace-collapsed + punctuation-spacing/typography-insensitive form.

    Used for the chunk-text-must-be-a-substring assertion: it tolerates the
    renderer's span-induced spacing and quote style without tolerating paraphrase.
    """
    return _PUNCT_WS_RE.sub(r"\1", norm_ws(text)).translate(_QUOTE_TRANSLATION)


class QA(msgspec.Struct, kw_only=True):
    q: str = ""
    a: str = ""


class CorpusChunk(msgspec.Struct, kw_only=True):
    """One retrieval chunk. ``chunk_uid`` keys BOTH index layers (dual-write)."""

    chunk_uid: str
    heading_path: list[str] = []
    text: str = ""
    summary: str = ""
    keywords: list[str] = []
    synonyms: list[str] = []
    qa: list[QA] = []


class PageCorpus(msgspec.Struct, kw_only=True):
    """Per-page corpus file — the unit of incremental regeneration."""

    source: str  # ROOT-relative posix
    title: str
    html_md5: str
    gen_key: str
    generated_at: str  # ISO8601
    schema_version: str = SCHEMA_VERSION
    chunks: list[CorpusChunk] = []

    def to_json_dict(self) -> dict:
        return msgspec.to_builtins(self)

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "PageCorpus":
        return msgspec.json.decode(data, type=cls)


class LLMChunk(msgspec.Struct, kw_only=True):
    """What the LLM actually emits (no chunk_uid — we assign it)."""

    heading_path: list[str] = []
    text: str = ""
    summary: str = ""
    keywords: list[str] = []
    synonyms: list[str] = []
    qa: list[QA] = []


class LLMPageOutput(msgspec.Struct, kw_only=True):
    """Top-level LLM JSON envelope."""

    chunks: list[LLMChunk] = []


def make_chunk_uid(source: str, idx: int) -> str:
    """Stable chunk id: sha1(source#idx), 16 hex chars.

    Deterministic across rebuilds as long as the per-page chunk ORDER is stable
    (it is: index order from generation, fallback chunker is deterministic).
    """
    return hashlib.sha1(f"{source}#{idx}".encode("utf-8")).hexdigest()[:16]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_index_text(chunk: CorpusChunk, title: str = "", aux: bool = True) -> str:
    """BM25 index text: clean text PLUS synthetic aux fields.

    Aux text (summary/keywords/synonyms/qa) boosts lexical recall; retrieval
    hits still map back to the clean original chunk.
    """
    parts: list[str] = []
    if title:
        parts.append(title)
    parts.extend(chunk.heading_path)
    parts.append(chunk.text)
    if aux:
        if chunk.summary:
            parts.append(chunk.summary)
        parts.extend(chunk.keywords)
        parts.extend(chunk.synonyms)
        parts.extend(q.q for q in chunk.qa if q.q)
    return "\n".join(p for p in parts if p)


def to_embed_text(chunk: CorpusChunk, title: str = "") -> str:
    """Dense index text: CLEAN text only (title + heading path + text).

    Aux fields are synthetic — they must never pollute the vector space
    (the old word+dense(hash) regression MRR 0.875 -> 0.792 is the lesson).
    """
    parts: list[str] = []
    if title:
        parts.append(title)
    parts.extend(chunk.heading_path)
    parts.append(chunk.text)
    return "\n".join(p for p in parts if p)


def snippet(chunk: CorpusChunk, query_terms: list[str] | None = None,
            width: int = 500) -> tuple[str, bool]:
    """Clean-text snippet centered on the first query-term hit.

    Returns (snippet, truncated).
    """
    text = chunk.text
    if query_terms:
        low = text.lower()
        pos = -1
        for term in query_terms:
            pos = low.find(term.lower())
            if pos >= 0:
                break
        if pos >= 0:
            half = width // 2
            lo = max(0, pos - half)
            hi = min(len(text), lo + width)
            lo = max(0, hi - width)
            snip = text[lo:hi].strip()
            return snip, (lo > 0 or hi < len(text))
    return text[:width].strip(), len(text) > width
