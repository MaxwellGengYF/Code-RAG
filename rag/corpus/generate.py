"""LLM corpus generation for one page: strict JSON -> json_repair salvage -> self-repair sessions.

ONE PAGE = ONE SESSION. This module is called once per page (see
``run_corpus_compile``), and every call it makes is a fresh single-turn session:
the client protocol keeps no tools, no multi-turn history and no provider-side
session state (rag/llm/base.py), so nothing — prompt context, repair chain,
token accounting, or failure state — ever crosses a page boundary. The
self-repair sessions below are contained in the same per-page unit: they fix
THIS page's JSON and are discarded with it.

Pipeline per page:
  1. LLM call (system + user prompt only — no tools).
  2. JSON extraction by an escalating recovery ladder: strict decode, then
     truncated-closer repair (local models sometimes emit EOS right after the
     last chunk object, dropping the trailing ``]}``), then ``json_repair``
     (trailing commas, unterminated strings, stray closers — the
     hallucinated-JSON salvage).
  3. msgspec validation + page-level invariants, including the verbatim
     assertion: whitespace-collapsed chunk text must be a substring of the
     whitespace-collapsed source markdown.
  4. On parse/validation errors: up to ``MAX_JSON_REPAIR_SESSIONS`` fresh LLM
     sessions (each call is a new session — this backend keeps no history),
     each echoing the error list back so the model fixes its own JSON.
  5. Still unrepairable after all sessions: raise :class:`CorpusGenerationError`
     (hard fail).
  6. API failure (provider down even after the transient-retry schedule):
     heuristic fallback (a port of the original fixed-size ``chunk_text``) with
     empty aux fields, flagged ``needs_regen`` by the compile loop. Retriable
     failures (429 / 5xx / timeout / connection) wait out
     :data:`rag.llm.base.RETRY_DELAYS` — 2s, 4s, 1min, 10min, 1h, 2h, 4h —
     before the page is demoted, so a rate limit or an exhausted quota window
     costs latency, not corpus quality. A Ctrl-C during one of those long waits
     raises :class:`rag.llm.base.RetryAborted`, which is NOT an API failure:
     the page is dropped unfinished (nothing written) by the compile loop.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Sequence

import json_repair
import msgspec

from rag.llm.base import (
    RETRY_DELAYS,
    LLMClient,
    RetryAborted,
    fmt_delay,
    with_retry,
)
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

MAX_RETRIES = len(RETRY_DELAYS)  # retries per LLM call, one per schedule entry:
# HTTP 429 / 5xx / timeout / connection reset wait 2s, 4s, 1min, 10min, 1h, 2h,
# 4h before the page is demoted to the heuristic fallback (rag.llm.base)
MAX_QA_PER_CHUNK = 3
MAX_CHUNKS_PER_PAGE = 40  # sanity cap against runaway generations
#: fresh LLM sessions allowed for the model to fix its own broken JSON after
#: the initial generation; exceeding this raises CorpusGenerationError
MAX_JSON_REPAIR_SESSIONS = 3
#: retry waits at or above this announce themselves on stderr (the run must never
#: look hung just because the provider asked us to wait ten minutes)
SLOW_RETRY_NOTICE_S = 60.0


class CorpusGenerationError(Exception):
    """A page's LLM output stayed broken after every recovery step.

    Raised only for persistently unrepairable MODEL OUTPUT (bad JSON that
    survives ``json_repair`` plus every self-repair session). Provider/API
    failures are a different failure mode and demote to the heuristic
    fallback instead.

    ``stats`` carries the spent generation stats (attempts, tokens, errors)
    so callers degrading the page to a fallback keep honest accounting.
    """

    def __init__(self, message: str, *, stats: PageGenStats | None = None):
        super().__init__(message)
        self.stats = stats


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


def _close_truncated_containers(s: str) -> str | None:
    """Repair *s* when it is truncated only in its trailing container closers.

    Local models (observed with Qwen3.5-9B non-thinking) sometimes emit their
    EOS token right after the last chunk object, dropping the final ``]}`` —
    the JSON is complete in content but unbalanced. A string-aware stack scan
    appends exactly the missing closers. Returns None when the text has any
    other problem (mismatched closer, ends inside a string, already balanced):
    those are not truncations and must keep failing.
    """
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return None  # mismatched closer — not a pure truncation
    if in_str or not stack:
        return None
    return s + "".join(reversed(stack))


def extract_json_object(text: str) -> dict:
    """Return the first balanced {...} object in *text* as a parsed dict.

    Escalating recovery ladder for hallucinated output:
      1. strict decode — markdown fences and leading/trailing commentary are
         tolerated, nothing else;
      2. truncated-closer repair — output cut early only in its trailing
         ``]}`` (exact, never alters content);
      3. ``json_repair`` — trailing commas, unterminated strings, mismatched
         closers, and similar breakage a hallucinating model produces.

    Raises ValueError when no step yields a JSON object.
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
    candidate = cleaned[start:]
    try:
        obj, _end = decoder.raw_decode(candidate)
    except json.JSONDecodeError as exc:
        repaired = _close_truncated_containers(candidate)
        if repaired is not None:
            try:
                obj, _end = decoder.raw_decode(repaired)
            except json.JSONDecodeError:
                pass  # not repairable this way — fall through to json_repair
            else:
                if isinstance(obj, dict):
                    return obj
        # json_repair never raises by contract, but stay defensive: a repair
        # failure here means the output is declared broken, full stop.
        try:
            obj = json_repair.repair_json(candidate, return_objects=True)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            return obj
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
    retry_delays: Sequence[float] | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> tuple[PageCorpus, PageGenStats]:
    """Generate the corpus for one page.

    Returns a usable PageCorpus on success (first try or after self-repair
    sessions) and on API failure (heuristic fallback). Raises
    :class:`CorpusGenerationError` when the model's JSON stays broken after
    ``MAX_JSON_REPAIR_SESSIONS`` self-repair sessions, and propagates
    :class:`RetryAborted` (an abort asked for during a long backoff wait — the
    page is dropped, never degraded).

    ``retries``/``retry_delays`` control the transient-failure schedule (default
    :data:`RETRY_DELAYS`); ``should_abort`` is polled while waiting it out.
    """
    stats = PageGenStats(source=page.source)
    sys_prompt = prompts.system_prompt(max_chunk_chars)
    user_prompt = prompts.build_user_prompt(
        page.source, page.title, page.markdown, page.char_len,
        truncated=page.char_len > len(page.markdown))

    def note_retry(exc: BaseException, n: int, delay: float) -> None:
        """Record a retry on the page's stats, and announce long backoffs.

        A one-minute-plus silence would look like a hung run, so from
        ``SLOW_RETRY_NOTICE_S`` on the wait is printed. Short waits (2s/4s) stay
        quiet — they are ordinary throttling and the page usually succeeds.
        """
        stats.errors.append(f"retry {n}: {exc}")
        if delay >= SLOW_RETRY_NOTICE_S:
            print(f"  [retry] {client.model_name}: {exc} — waiting "
                  f"{fmt_delay(delay)} before retry {n}", file=sys.stderr)

    async def call(prompt_text: str):
        result = await with_retry(
            lambda: client.generate(sys_prompt, prompt_text),
            delays=retry_delays, retries=retries, should_abort=should_abort,
            on_retry=note_retry)
        stats.attempts += 1
        stats.input_tokens += result.input_tokens
        stats.output_tokens += result.output_tokens
        return result

    def parse_and_validate(text: str) -> tuple[list, list[str]]:
        """Extraction ladder + validation; never raises."""
        try:
            obj = extract_json_object(text)
            chunks, errors = validate_llm_output(obj, page, max_chunk_chars)
        except ValueError as exc:
            chunks, errors = [], [str(exc)]
        return chunks, errors

    # --- initial generation: one extra shot on an API blip, then fallback ---
    for api_shot in (1, 2):
        try:
            result = await call(user_prompt)
            break
        except RetryAborted:
            raise  # Ctrl-C during a backoff wait: drop the page, degrade nothing
        except Exception as exc:  # API failed even after transient retries
            stats.api_failed = True
            stats.errors.append(f"api error (shot {api_shot}): {exc}")
    else:
        # Provider is down — not the model's fault. Degrade to heuristic
        # chunks (flagged needs_regen by the compile loop for a later retry).
        stats.fallback = True
        corpus = _fallback_corpus(page, gen_key)
        corpus.html_md5 = html_md5
        return corpus, stats

    # --- initial output + up to MAX_JSON_REPAIR_SESSIONS fresh sessions ---
    # Each session is a brand-new generate() call (this backend keeps no
    # history) whose repair prompt echoes the parse/validation errors back, so
    # the model fixes its own JSON. Track the best output: valid chunks are
    # kept even when some siblings failed, so a single bad chunk never
    # demotes the whole page.
    best: tuple[list, int] | None = None  # (chunks, attempt)
    attempt = 0
    sessions_used = 0
    while True:
        attempt += 1
        chunks, errors = parse_and_validate(result.text)
        stats.errors.extend(f"attempt {attempt}: {e}" for e in errors[:8])
        stats.dropped_chunks += len(errors)
        if chunks and (best is None or len(chunks) > len(best[0])):
            best = (chunks, attempt)
        if chunks and not errors:
            break  # fully valid — stop spending sessions
        if sessions_used >= MAX_JSON_REPAIR_SESSIONS:
            break  # out of sessions — the caller decides below
        sessions_used += 1
        try:
            result = await call(
                prompts.build_repair_prompt(user_prompt, errors[:8]))
        except RetryAborted:
            raise  # Ctrl-C during a backoff wait: drop the page
        except Exception as exc:  # provider died mid-repair
            stats.api_failed = True
            stats.errors.append(f"api error (repair session {sessions_used}): {exc}")
            break

    if best is not None:
        chunks, attempt = best
        stats.first_try_valid = attempt == 1
        stats.repaired = attempt > 1
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

    if stats.api_failed:
        # Provider died mid-repair and nothing usable was salvaged — degrade
        # to heuristic chunks (retried next run via needs_regen).
        stats.fallback = True
        corpus = _fallback_corpus(page, gen_key)
        corpus.html_md5 = html_md5
        return corpus, stats

    raise CorpusGenerationError(
        f"{page.source}: LLM output unrepairable after "
        f"{sessions_used} self-repair session(s) ({stats.attempts} "
        f"generation(s)): " + " | ".join(stats.errors[-4:]),
        stats=stats)
