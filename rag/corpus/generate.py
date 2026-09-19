"""LLM corpus generation for one page: strict JSON -> validation -> repair -> fallback.

Never hard-fails: every page yields a usable PageCorpus. Failures are reported
via :class:`PageGenStats` for the caller to log to failures.jsonl.

Pipeline per page:
  1. LLM call (system + user prompt only — no tools).
  2. Strict JSON extraction (strip fences, take the outermost object).
  3. msgspec validation + page-level invariants, including the verbatim
     assertion: whitespace-collapsed chunk text must be a substring of the
     whitespace-collapsed source markdown.
  4. On validation errors: ONE repair retry with the error list echoed back.
  5. On repair failure / API failure: heuristic fallback (a port of the
     original fixed-size ``chunk_text``) with empty aux fields.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import msgspec

from rag.llm.base import LLMClient, with_retry
from rag.corpus.extract import PageInput
from rag.corpus import prompts
from rag.corpus.schema import (
    LLMPageOutput,
    CorpusChunk,
    PageCorpus,
    make_chunk_uid,
    norm_verbatim,
    now_iso,
)

MAX_RETRIES = 4  # transient API retries per attempt (429/5xx/timeout)
MAX_QA_PER_CHUNK = 3
MAX_CHUNKS_PER_PAGE = 40  # sanity cap against runaway generations


@dataclass
class PageGenStats:
    source: str
    attempts: int = 0  # LLM calls made (0 for pure fallback)
    first_try_valid: bool = False
    repaired: bool = False
    fallback: bool = False
    api_failed: bool = False  # at least one attempt died on an API error
    dropped_chunks: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.first_try_valid or self.repaired


# --------------------------------------------------------------------------------------
# strict JSON extraction
# --------------------------------------------------------------------------------------


def extract_json_object(text: str) -> dict:
    """Return the first balanced {...} object in *text* as a parsed dict.

    Tolerates markdown fences and leading/trailing commentary. Raises
    ValueError when no parseable object exists.
    """
    cleaned = text.strip()
    # strip a whole-output ```json ... ``` wrapper if present (anchored, so inner
    # code fences like ```csharp inside chunk text are never mistaken for it)
    if cleaned.startswith("```"):
        fence = re.match(r"^```(?:json)?\s*(.*?)```\s*$", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("no JSON object found in model output")
    # decode incrementally to find the end of the first balanced object
    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object")
    return obj


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------


def validate_llm_output(obj: dict, page: PageInput, max_chunk_chars: int) -> tuple[list, list[str]]:
    """Decode + validate. Returns (llm_chunks, error_list). Errors → repair."""
    errors: list[str] = []
    try:
        out = msgspec.convert(obj, type=LLMPageOutput)
    except msgspec.ValidationError as exc:
        return [], [f"schema decode failed: {exc}"]
    if not out.chunks:
        return [], ["no chunks in output"]
    if len(out.chunks) > MAX_CHUNKS_PER_PAGE:
        errors.append(f"{len(out.chunks)} chunks exceeds cap {MAX_CHUNKS_PER_PAGE}")
    page_norm = norm_verbatim(page.markdown)
    valid_chunks = []
    for i, ch in enumerate(out.chunks):
        if not ch.text or not ch.text.strip():
            errors.append(f"chunk {i}: empty text")
            continue
        if len(ch.text) > max_chunk_chars * 2:
            errors.append(f"chunk {i}: {len(ch.text)} chars, over 2x the {max_chunk_chars} limit")
        if norm_verbatim(ch.text) not in page_norm:
            errors.append(f"chunk {i}: text is not a verbatim excerpt of the source page")
            continue
        if len(ch.qa) > MAX_QA_PER_CHUNK:
            ch.qa = ch.qa[:MAX_QA_PER_CHUNK]
        valid_chunks.append(ch)
    if not valid_chunks:
        errors.append("no valid chunks survived validation")
    return valid_chunks, errors


# --------------------------------------------------------------------------------------
# heuristic fallback (port of hybrid_retrieve.chunk_text)
# --------------------------------------------------------------------------------------


def chunk_text(text: str, max_len: int = 800, overlap: int = 120) -> list[str]:
    """Simple paragraph-based chunking with overlap (verbatim by construction)."""
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 1 > max_len and current:
            chunks.append(current.strip())
            current = current[-overlap:] if overlap < len(current) else ""
        current += ("\n" if current else "") + p
    if current:
        chunks.append(current.strip())
    return chunks or [text[:max_len]]


def _fallback_corpus(page: PageInput, gen_key: str) -> PageCorpus:
    chunks = [
        CorpusChunk(chunk_uid=make_chunk_uid(page.source, i),
                    heading_path=[page.title] if page.title else [],
                    text=c)
        for i, c in enumerate(chunk_text(page.markdown))
    ]
    return PageCorpus(source=page.source, title=page.title, html_md5="",
                      gen_key=gen_key, generated_at=now_iso(), chunks=chunks)


# --------------------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------------------


async def generate_page_corpus(
    client: LLMClient,
    page: PageInput,
    *,
    gen_key: str,
    html_md5: str,
    max_chunk_chars: int = 1200,
    retries: int = MAX_RETRIES,
) -> tuple[PageCorpus, PageGenStats]:
    """Generate the corpus for one page; always returns a usable PageCorpus."""
    stats = PageGenStats(source=page.source)
    sys_prompt = prompts.system_prompt(max_chunk_chars)
    user_prompt = prompts.build_user_prompt(
        page.source, page.title, page.markdown, page.char_len,
        truncated=page.char_len > len(page.markdown))

    async def call(prompt_text: str):
        result = await with_retry(
            lambda: client.generate(sys_prompt, prompt_text), retries=retries,
            on_retry=lambda exc, n: stats.errors.append(f"retry {n}: {exc}"))
        stats.attempts += 1
        stats.input_tokens += result.input_tokens
        stats.output_tokens += result.output_tokens
        return result

    # Track the best attempt: valid chunks are kept even when some siblings failed,
    # so a single bad chunk never demotes the whole page to heuristic fallback.
    best: tuple[list, int] | None = None  # (chunks, attempt)

    for attempt in (1, 2):
        try:
            result = await call(user_prompt)
        except Exception as exc:  # API failed even after retries
            stats.api_failed = True
            stats.errors.append(f"api error (attempt {attempt}): {exc}")
            if attempt == 1:
                # give the provider one more shot (it may have been a blip);
                # if it fails again we fall through to the heuristic fallback
                continue
            break
        try:
            obj = extract_json_object(result.text)
            chunks, errors = validate_llm_output(obj, page, max_chunk_chars)
        except ValueError as exc:
            chunks, errors = [], [str(exc)]
        stats.errors.extend(f"attempt {attempt}: {e}" for e in errors[:8])
        stats.dropped_chunks += len(errors)
        if chunks and (best is None or len(chunks) > len(best[0])):
            best = (chunks, attempt)
        if chunks and not errors:
            break  # fully valid — no need for the repair attempt
        if attempt == 1:
            user_prompt = prompts.build_repair_prompt(user_prompt, errors[:8])

    if best is not None:
        chunks, attempt = best
        stats.first_try_valid = attempt == 1
        stats.repaired = attempt == 2
        corpus = PageCorpus(
            source=page.source, title=page.title, html_md5=html_md5,
            gen_key=gen_key, generated_at=now_iso(),
            chunks=[CorpusChunk(
                chunk_uid=make_chunk_uid(page.source, i),
                heading_path=ch.heading_path or ([page.title] if page.title else []),
                text=ch.text, summary=ch.summary, keywords=ch.keywords,
                synonyms=ch.synonyms, qa=ch.qa,
            ) for i, ch in enumerate(chunks)],
        )
        return corpus, stats

    stats.fallback = True
    corpus = _fallback_corpus(page, gen_key)
    corpus.html_md5 = html_md5
    return corpus, stats
