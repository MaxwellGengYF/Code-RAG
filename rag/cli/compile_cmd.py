"""The ``compile`` command: LLM corpus generation with md5-incremental scheduling.

Flow: scan → diff vs manifest → (per-page gen_key mismatch forces regen) → async
LLM generation with per-provider worker semaphores → per-page corpus files
(atomic) → checkpointed manifest flushes → failures.jsonl → final report.

Never aborts the run on per-page failures: a page that fails every LLM attempt
falls back to heuristic chunking (see rag.corpus.generate) and is logged.

Multi-provider sharding: pages are assigned to providers deterministically
(md5(rel) % n_providers), which doubles throughput when two gateways are
available and keeps reruns stable (a page always maps to the same provider).

Ctrl-C quits cleanly instead of dumping an asyncio traceback: the first interrupt
drains the pages in flight (each one is checkpointed), the run reports how far it
got, and the CLI exits 130 (:data:`INTERRUPT_EXIT_CODE`); a second interrupt
aborts the in-flight calls immediately.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from rag.corpus import (
    EXTRACTOR_VERSION,
    SCHEMA_VERSION,
    CorpusGenerationError,
    PageCorpus,
    extract_page,
    generate_page_corpus,
)
from rag.config import data_path
from rag.corpus.prompts import PROMPT_VERSION
from rag.llm.base import LLMClient
from rag.store import CorpusStore, FileManager, ManifestDiff

log = logging.getLogger(__name__)

#: The compile UNIT is ONE PAGE — always. One page is one LLM session (the
#: tool-free clients keep no history, see rag/llm/base.py), so the unit of work,
#: the unit of accounting and the unit of LLM state all coincide: nothing is
#: shared across pages, and a killed run can lose at most the pages its workers
#: had in flight. Both granularities below therefore default to 1:
#:   * CHECKPOINT_EVERY — the incremental manifest flush: after EVERY finished
#:     page, so resume never rewinds past a page whose LLM tokens were already
#:     paid for (the old value, 25, could throw away a whole batch);
#:   * PROGRESS_EVERY — the progress line: after EVERY finished page, so the bar
#:     also moves on work lists smaller than the old 50-page print stride.
#: Both stay overridable per config ("checkpoint_every" / "progress_every") for
#: builds that would rather trade resume granularity for fewer manifest writes;
#: >1 batches, 0 disables that mechanism (the final report always prints).
CHECKPOINT_EVERY = 1
PROGRESS_EVERY = 1
#: Per-page checkpointing rewrites the manifest once per page; re-parsing the
#: whole manifest each time would be the dominant cost on a 44k-page build
#: (~10 MB x 44k), so the merge works off an in-memory view that is re-read from
#: disk every N checkpoints. That keeps a concurrent run's entries visible while
#: making the steady-state flush a dict merge + one atomic write.
MANIFEST_RELOAD_EVERY = 100
#: Token-estimator constants, CALIBRATED against real gateway usage counters
#: (measured on a 24-page sample: estimate was +217% off before this).
#:
#: The system prompt is NOT billed: these gateways apply prompt caching, and the
#: corpus build sends an identical system prompt on every request, so it is cached
#: after the first. Counting it (~620 tokens/page) tripled the estimate. Only the
#: per-page markdown is charged, at ~3.07 chars/token for this corpus (English
#: prose + C# identifiers + markdown tables).
CHARS_PER_TOKEN = 3.07
#: output/input ratio, measured 2.04 (no-thinking); thinking adds reasoning tokens
EST_OUTPUT_INPUT_RATIO_THINKING = 2.2
EST_OUTPUT_INPUT_RATIO_NO_THINKING = 2.04
#: fixed per-request token overhead (roles, formatting)
EST_REQUEST_OVERHEAD_TOKENS = 16


@dataclass
class CompileReport:
    pages_total: int = 0       # pages in the scan
    pages_planned: int = 0     # pages selected for regeneration
    pages_done: int = 0
    chunks: int = 0
    first_try: int = 0
    repaired: int = 0
    fallback: int = 0
    pruned: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: list[dict] = field(default_factory=list)
    wall_s: float = 0.0
    dry_run: bool = False
    est_input_tokens: int = 0
    est_output_tokens: int = 0
    #: pages whose corpus is a heuristic fallback and should be retried later
    needs_regen: set[str] = field(default_factory=set)
    #: pages the final manifest claims as processed (set by finalize)
    pages_claimed: int = 0
    #: set when the run stopped early because every provider circuit was open
    aborted_all_providers_down: bool = False
    #: set when the run stopped because the user pressed Ctrl-C (see
    #: :func:`install_interrupt_stop`): pages already finished are on disk, the
    #: rest were left untouched for a later resume.
    interrupted: bool = False

    def to_dict(self) -> dict:
        out = dict(self.__dict__)
        out["needs_regen"] = sorted(out.get("needs_regen", ()))
        return out
#: Exit status of a run the user stopped with Ctrl-C (POSIX convention: 128+2).
INTERRUPT_EXIT_CODE = 130
#: Written with os.write from the signal handler — terse, and safe to emit from
#: inside a handler (no buffered-stream lock to deadlock on).
INTERRUPT_NOTICE = (
    "\n  [interrupt] Ctrl-C — stopping: the pages in flight are still finished, "
    "saved and checkpointed.\n"
    "  [interrupt] press Ctrl-C again to abandon them and quit immediately.\n"
)


def install_interrupt_stop(aborted: dict, interrupted: dict) -> object | None:
    """Make the FIRST Ctrl-C stop the corpus run cleanly instead of unwinding it.

    ``aborted["v"]`` makes the workers stop taking new pages; the pages already in
    flight are still finished, written and checkpointed, so a graceful interrupt
    costs ZERO pages (the manifest flush is per page). ``interrupted["v"]`` records
    why the run ended, so the report and the CLI can say so.

    A SECOND Ctrl-C falls through to the usual behaviour (``KeyboardInterrupt``)
    for a user who will not wait for the in-flight LLM calls.

    Returns the handler to hand back to :func:`restore_interrupt_stop`, or ``None``
    when SIGINT cannot be hooked here (not the main thread, or a platform without
    SIGINT) — in which case Ctrl-C keeps its default meaning, and asyncio's own
    SIGINT path (cancel the main task) applies.
    """
    if not hasattr(signal, "SIGINT"):
        return None

    def on_sigint(signum, frame):
        # Runs in the main thread's signal context while the loop is parked in a
        # task step or in select(); keep it to flag-setting + one raw write.
        if interrupted["v"]:
            raise KeyboardInterrupt  # second Ctrl-C: abort now
        interrupted["v"] = True
        aborted["v"] = True
        try:
            os.write(2, INTERRUPT_NOTICE.encode("utf-8"))
        except OSError:
            pass

    try:
        return signal.signal(signal.SIGINT, on_sigint)
    except (ValueError, OSError):
        return None


def restore_interrupt_stop(previous: object | None) -> None:
    """Put the previous SIGINT handler back (no-op when none was installed)."""
    if previous is None:
        return
    try:
        signal.signal(signal.SIGINT, previous)
    except (ValueError, OSError, TypeError):
        pass


def interrupt_message(report: CompileReport | None = None, *,
                      hard: bool = False) -> str:
    """One friendly line for a run the user stopped with Ctrl-C.

    ``hard`` marks the abort that did NOT wait for the in-flight pages (a second
    Ctrl-C, or a cancellation): asyncio re-raises ``KeyboardInterrupt`` straight
    out of the event loop (see ``asyncio.tasks``), so on that path the partial
    report is gone and the caller passes ``None``. Either way every FINISHED page
    is on disk — that is the part the user needs to trust.
    """
    where = ("Rerun the same command to resume — every finished page is "
             "checkpointed and no half-written page is ever claimed.")
    if report is None or not report.pages_planned:
        # Nothing to quote: this is the hard path, where asyncio re-raised the
        # KeyboardInterrupt past the code that owns the report.
        return (f"[interrupt] stopped by Ctrl-C (the pages in flight were "
                f"abandoned). {where}")
    how = ("the pages in flight were abandoned" if hard else
           "every in-flight page was finished and saved")
    left = max(0, report.pages_planned - report.pages_done)
    return (f"[interrupt] stopped by Ctrl-C after {report.pages_done:,} of "
            f"{report.pages_planned:,} pages ({how}); {left:,} pages still "
            f"pending. {where}")


# --------------------------------------------------------------------------------------
# generation keys + provider sharding
# --------------------------------------------------------------------------------------


def gen_key_for_model(model: str) -> tuple[str, dict]:
    parts = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "extractor_version": EXTRACTOR_VERSION,
        "schema_version": SCHEMA_VERSION,
    }
    return FileManager.gen_key(**parts), parts


def shard_index(rel: str, n_shards: int) -> int:
    """Deterministic provider assignment for one page."""
    return int(hashlib.md5(rel.encode("utf-8")).hexdigest(), 16) % n_shards


class ProviderShard:
    """One provider's client + its generation key."""

    def __init__(self, client: LLMClient, model: str):
        self.client = client
        self.model = model
        self.gen_key, self.gen_parts = gen_key_for_model(model)


def make_expected_gen_key(shards: Sequence[ProviderShard]) -> Callable[[str], str]:
    keys = [s.gen_key for s in shards]
    return lambda rel: keys[shard_index(rel, len(keys))]


def valid_gen_keys(shards: Sequence[ProviderShard]) -> set[str]:
    """Every gen_key the current fleet can legitimately produce.

    Used by :func:`plan_work` to decide whether a page's stored corpus is still
    valid. Set membership (rather than an exact per-shard match) is what makes
    cross-provider failover safe: a page served by a backup provider after its
    primary died keeps a valid key and is not pointlessly regenerated.
    """
    return {s.gen_key for s in shards}


def set_gen_key(shards: Sequence[ProviderShard]) -> str:
    """Manifest-level marker for a multi-provider build (informational only;
    per-page validity is tracked in ``page_gen_keys``)."""
    return "|".join(sorted({s.gen_key for s in shards}))


# --------------------------------------------------------------------------------------
# provider pool: circuit breaker + cross-provider failover
# --------------------------------------------------------------------------------------

#: consecutive hard API failures before a provider is tripped open
CIRCUIT_THRESHOLD = 12
#: how long a tripped provider stays out of rotation before a probe (seconds).
#: Deliberately short: 429 throttling recovers in seconds, and even a kimi-style
#: 5-hour quota window is better re-probed every few minutes (one wasted request
#: per probe) than assumed dead for the rest of the run.
CIRCUIT_COOLDOWN = 180.0
#: when EVERY circuit is open (shared quota window exhausted), wait this long for
#: recovery before aborting the run. Waiting is strictly better than degrading:
#: aborting leaves the remaining pages untouched for a later run, whereas
#: proceeding would overwrite ~37k pages with aux-less heuristic chunks and then
#: exit 0, so the resume loop would report COMPILE COMPLETE and never retry.
ALL_DOWN_WAIT_BUDGET = 5400.0


class ProviderPool:
    """Per-provider circuit breakers with automatic failover.

    Rationale (learned the hard way on a 44k-page build): several of these
    gateways die mid-run — exhausted 5-hour quota windows (kimi: 403
    access_terminated), revoked model access (403 AccessDenied.Unpurchased).
    Without failover every page assigned to a dead provider silently degrades to
    the aux-less heuristic fallback while still being marked complete.

    So: after CIRCUIT_THRESHOLD consecutive hard API failures a provider is
    tripped open; its pages are served by the next healthy provider (which
    records ITS gen_key — the per-page key registry keeps that honest). After
    CIRCUIT_COOLDOWN one request is let through as a probe; success closes the
    circuit again.

    Note the pool deliberately ignores shard_index: assignment is dynamic.
    Determinism of *content* comes from the page markdown, not from which model
    chunked it; determinism of *incrementality* comes from page_gen_keys.
    """

    def __init__(self, shards: Sequence[ProviderShard], *,
                 threshold: int = CIRCUIT_THRESHOLD,
                 cooldown: float = CIRCUIT_COOLDOWN):
        self.shards = list(shards)
        self.threshold = threshold
        self.cooldown = cooldown
        self.failures = [0] * len(shards)   # consecutive hard API failures
        self.opened_at: list[float | None] = [None] * len(shards)
        self.trips = [0] * len(shards)     # times tripped open (diagnostics)
        self.served = [0] * len(shards)    # pages actually generated (diagnostics)

    def is_open(self, i: int, now: float) -> bool:
        """True when provider *i* must be skipped right now."""
        if self.opened_at[i] is None:
            return False
        if now - self.opened_at[i] >= self.cooldown:
            return False  # half-open: allow a probe request through
        return True

    def healthy_indices(self, now: float | None = None) -> list[int]:
        now = time.time() if now is None else now
        out = [i for i in range(len(self.shards)) if not self.is_open(i, now)]
        return out

    def pick(self, preferred: int, now: float | None = None) -> int | None:
        """Provider to use: *preferred* when healthy, else the next healthy one.

        Returns None when every provider is tripped (caller should then fall
        back — and the page is left flagged needs_regen for a later run).
        """
        now = time.time() if now is None else now
        if not self.is_open(preferred, now):
            return preferred
        for i in self.healthy_indices(now):
            return i
        return None

    def order(self, preferred: int, now: float | None = None) -> list[int]:
        """Candidate providers to try in order: *preferred* first (when healthy),
        then every other healthy one.

        Empty when all circuits are open and still cooling down — the caller must
        then fall back instead of burning requests on providers known to be down.
        """
        now = time.time() if now is None else now
        out: list[int] = []
        if not self.is_open(preferred, now):
            out.append(preferred)
        for i in self.healthy_indices(now):
            if i != preferred:
                out.append(i)
        return out

    def record_success(self, i: int) -> None:
        if self.opened_at[i] is not None:
            log.info("provider %s circuit CLOSED again after probe",
                     self.shards[i].model)
        self.failures[i] = 0
        self.opened_at[i] = None
        self.served[i] += 1

    def record_api_failure(self, i: int, now: float | None = None) -> None:
        self.failures[i] += 1
        if self.opened_at[i] is None and self.failures[i] >= self.threshold:
            self.opened_at[i] = time.time() if now is None else now
            self.trips[i] += 1
            log.warning("provider %s circuit OPEN after %d consecutive API "
                        "failures; its pages fail over to healthy providers for "
                        "%.0fs", self.shards[i].model, self.failures[i],
                        self.cooldown)

    def summary(self) -> str:
        parts = []
        for i, s in enumerate(self.shards):
            state = "OPEN" if self.opened_at[i] is not None else "ok"
            parts.append(f"{s.model}: served={self.served[i]} trips={self.trips[i]} "
                         f"state={state}")
        return " | ".join(parts)

    def all_open(self, now: float | None = None) -> bool:
        """True when no provider can serve a request right now."""
        return not self.healthy_indices(now)

    def next_probe_in(self, now: float | None = None) -> float:
        """Seconds until the soonest tripped provider may be probed again."""
        now = time.time() if now is None else now
        ready = [self.opened_at[i] + self.cooldown for i in range(len(self.shards))
                 if self.opened_at[i] is not None]
        if not ready:
            return 0.0
        return max(0.0, min(ready) - now)


# --------------------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------------------


def plan_work(
    fm: FileManager,
    store: CorpusStore,
    diff: ManifestDiff,
    *,
    valid_gen_keys: set[str] | None = None,
    force: bool = False,
    regen: bool = False,
    only: str | None = None,
    max_files: int | None = None,
) -> list[str]:
    """Pages needing (re)generation, sorted for determinism.

    A page is requeued when any of:
      * it is new or its html changed (the md5 diff);
      * its corpus file is missing (stat-only check);
      * its recorded gen_key is not among *valid_gen_keys* — i.e. it was built
        with a stale prompt/extractor/schema version, or by a provider no longer
        in the fleet. Set membership (not an exact per-shard match) is what makes
        cross-provider failover safe: a page served by a backup provider after
        its primary died is NOT pointlessly regenerated.
      * it is flagged ``needs_regen`` in the manifest (heuristic fallback).

    Bumping PROMPT_VERSION / EXTRACTOR_VERSION / SCHEMA_VERSION, or changing the
    provider fleet's models, changes every valid key → full regeneration.
    """
    if only:
        rel = only.replace("\\", "/").lstrip("./")
        if rel not in fm.scan():
            raise SystemExit(f"--only {only!r}: not an html page under {fm.dirs}")
        return [rel]
    candidates = sorted(diff.added + diff.changed + diff.unchanged) \
        if (force or regen) else sorted(diff.added + diff.changed)
    if force or regen:
        return candidates[:max_files] if max_files else candidates

    manifest = fm.load_manifest()
    page_gen_keys = manifest.get("page_gen_keys", {})
    needs_regen = set(manifest.get("needs_regen", []))

    work = list(candidates)
    for rel in diff.unchanged:
        if rel in needs_regen:
            work.append(rel)
        elif store.missing(rel):
            work.append(rel)
        elif valid_gen_keys is not None and page_gen_keys.get(rel) not in valid_gen_keys:
            work.append(rel)
    work = sorted(set(work))
    return work[:max_files] if max_files else work


# --------------------------------------------------------------------------------------
# clean rebuild (--clean)
# --------------------------------------------------------------------------------------


def wipe_generated_dirs(cfg: dict, *, base_dir: Path) -> list[Path]:
    """Delete ALL generated artefacts so the next build starts from nothing.

    Removes the configured ``corpus_dir`` (per-page corpus files, manifest.json,
    failures.jsonl) and ``index_dir`` (chunk table, BM25 + dense indexes, index
    manifest). After the wipe every scanned page is "added" and every index is
    rebuilt from scratch — that is exactly what ``compile --clean`` wants: no
    stale generated content can survive a config/provider change.

    SAFETY: a generated-dir setting of "." — or an absolute path pointing at
    the config or working directory — would delete the document mirror or the
    repo itself, so any dir that IS the base dir, the CWD, or the filesystem
    root is refused rather than removed. Missing dirs are skipped (nothing to
    delete).

    Returns the directories actually removed (empty when nothing existed).
    """
    import shutil

    cwd = Path.cwd().resolve()
    base = Path(base_dir).resolve()
    # Resolve and validate BOTH dirs before deleting ANY of them: a config with
    # a safe corpus_dir but a malicious index_dir must not wipe the corpus first
    # and only then refuse (the guards are only trustworthy when they all ran).
    targets: list[Path] = []
    for key, default in (("corpus_dir", "corpus"), ("index_dir", "index")):
        d = data_path(cfg, key, default, base=base).resolve()
        if d == base or d == cwd or d.parent == d:
            raise SystemExit(
                f"[clean] REFUSING to delete {d}: {key} must name a dedicated "
                f"generated-artefact directory, not the config directory, the "
                f"current working directory, or a filesystem root")
        if d not in targets:
            targets.append(d)
    wiped: list[Path] = []
    for d in targets:
        if d.exists():
            shutil.rmtree(d)
            wiped.append(d)
    return wiped


# --------------------------------------------------------------------------------------
# cost estimation
# --------------------------------------------------------------------------------------


def estimate_tokens(pages: list, sys_prompt: str, *,
                    thinking: bool = False,
                    system_prompt_billed: bool = False) -> tuple[int, int]:
    """(input, output) token estimates for a list of PageInputs.

    The system prompt is EXCLUDED by default: every request in a corpus build
    carries the identical prompt, so provider prompt caching bills it once at most.
    Counting it per page inflated the estimate ~3x (measured +217% error). Pass
    ``system_prompt_billed=True`` for a provider without caching.

    Sizes on ``len(page.markdown)`` — the CAPPED text actually sent — not
    ``page.char_len``, which is the pre-cap length and would overestimate on the
    long pages that hit the 24k input cap.
    """
    sys_chars = len(sys_prompt) if system_prompt_billed else 0
    est_in = 0
    for p in pages:
        sent = len(getattr(p, "markdown", "") or "") or p.char_len
        est_in += int((sys_chars + sent) / CHARS_PER_TOKEN
                      + EST_REQUEST_OVERHEAD_TOKENS)
    ratio = (EST_OUTPUT_INPUT_RATIO_THINKING if thinking
             else EST_OUTPUT_INPUT_RATIO_NO_THINKING)
    return est_in, int(est_in * ratio)


def report_cost(est_in: int, est_out: int, price_in: float | None,
                price_out: float | None) -> str:
    lines = [f"estimated tokens: input={est_in:,} output={est_out:,}",
             f"  (input counts page text only at {CHARS_PER_TOKEN} chars/token; the "
             f"identical system prompt is prompt-cached and not billed per page. "
             f"output = input x {EST_OUTPUT_INPUT_RATIO_NO_THINKING}). Calibrated "
             f"against real gateway usage; expect +-20%, not exact."]
    if price_in and price_out:
        cost = (est_in * price_in + est_out * price_out) / 1e6
        lines.append(f"estimated cost: {cost:.2f} (price_in={price_in}/M, "
                     f"price_out={price_out}/M)")
    else:
        lines.append("cost: pass --price-in/--price-out (per 1M tokens) to monetize")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# per-page progress (the compile unit is one page)
# --------------------------------------------------------------------------------------


def fmt_duration(seconds: float) -> str:
    """Compact duration: ``42s`` / ``5m03s`` / ``2h14m``."""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def progress_line(done: int, total: int, elapsed_s: float, chunks: int,
                  source: str, *, sessions: int = 1, fallback: int = 0) -> str:
    """One line per FINISHED PAGE.

    The compile unit is a single page, so this fires once per page (no modulo
    stride): the percentage a caller watches is the real per-page completion,
    and a work list smaller than the old 50-page print stride still shows
    progress. ``sessions`` is that page's LLM call count (1 = one clean session,
    >1 = self-repair sessions), which is why the line doubles as the "one page,
    one session" audit trail. Rate + ETA make a 44k-page run observable without
    waiting for a milestone.
    """
    pct = (100.0 * done / total) if total else 100.0
    rate = (done / elapsed_s) if elapsed_s > 0 else 0.0
    eta = ((total - done) / rate) if rate > 0 and total > done else 0.0
    fb = f" fallback={fallback}" if fallback else ""
    return (f"  [{done}/{total} {pct:5.1f}%] {rate:.2f} pages/min "
            f"eta {fmt_duration(eta)} | chunks={chunks} sess={sessions}{fb} | "
            f"{source}")


# --------------------------------------------------------------------------------------
# the compile run
# --------------------------------------------------------------------------------------


async def run_corpus_compile(
    shards: Sequence[ProviderShard],
    cfg: dict,
    *,
    work: list[str],
    scanned: dict[str, str],
    fm: FileManager,
    workers_per_provider: int = 4,
    failures_path: Path | None = None,
    progress: bool = True,
    checkpoint_every: int = CHECKPOINT_EVERY,
    progress_every: int = PROGRESS_EVERY,
    circuit_threshold: int = CIRCUIT_THRESHOLD,
    circuit_cooldown: float = CIRCUIT_COOLDOWN,
    wait_budget_s: float = ALL_DOWN_WAIT_BUDGET,
) -> CompileReport:
    """Generate the corpus for *work*.

    THE UNIT IS ONE PAGE, end to end: one queue item, one ``generate_page_corpus``
    call chain (one LLM session, plus its own self-repair sessions), one corpus
    file, one manifest checkpoint, one progress line. No page ever shares LLM
    state, bookkeeping or a checkpoint window with another (default
    ``checkpoint_every``/``progress_every`` are 1; raise them per config to batch
    the manifest writes on very large builds, or set either to 0 to disable that
    mechanism (a disabled per-page flush still flushes once at the end).

    Ctrl-C is not an error: the first interrupt stops the workers from taking new
    pages, lets the pages in flight finish and be checkpointed, and returns a
    report with ``interrupted`` set. A second one unwinds the run immediately;
    asyncio re-raises it out of the event loop (it is not catchable inside the
    coroutine), so the CLI's own KeyboardInterrupt handler reports it.
    """
    store = CorpusStore(fm.corpus_dir)
    report = CompileReport(pages_total=len(scanned), pages_planned=len(work))
    t0 = time.time()
    processed: dict[str, str] = {}   # rel -> gen_key actually used
    needs_regen: set[str] = set()    # heuristic-fallback pages (retry next run)
    pool = ProviderPool(shards, threshold=circuit_threshold,
                        cooldown=circuit_cooldown)
    sems = [asyncio.Semaphore(workers_per_provider) for _ in shards]
    done_counter = 0
    lock = asyncio.Lock()
    aborted = {"v": False}
    #: set by the Ctrl-C handler (see install_interrupt_stop): "stop taking new
    #: pages", as opposed to ``aborted`` which the workers raise themselves.
    interrupted = {"v": False}
    #: in-memory manifest view for the per-page checkpoint merge + how many
    #: checkpoints it is ahead of disk (see checkpoint()).
    manifest_cache: dict | None = None
    cache_age = 0
    max_chunk_chars = cfg.get("max_chunk_chars", 1200)
    max_input_chars = cfg.get("max_input_chars", 24_000)

    if failures_path is None:
        failures_path = fm.corpus_dir / "failures.jsonl"

    def write_failure(rec: dict) -> None:
        with open(failures_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def checkpoint() -> None:
        """Merge this run's progress into the manifest (atomic tmp+rename).

        Only actually-processed pages are claimed, and entries from earlier
        partial runs survive — so an interrupted run resumes exactly where it
        stopped instead of silently marking unprocessed pages done.

        Runs once per finished page (``checkpoint_every`` = 1), which is what
        makes a kill cost at most the in-flight pages. To keep that affordable
        the merge reads the manifest from disk only every
        ``MANIFEST_RELOAD_EVERY`` flushes and otherwise works off the in-memory
        view it wrote last; the reload keeps entries from a concurrent run
        visible. Call it in a thread (``await asyncio.to_thread(checkpoint)``)
        so the manifest write never blocks the workers' event loop.
        """
        nonlocal manifest_cache, cache_age
        if manifest_cache is None or cache_age >= MANIFEST_RELOAD_EVERY:
            manifest_cache = fm.load_manifest()
            cache_age = 0
        old = manifest_cache
        # Carry the previous fallback set forward, add this run's fallbacks, then
        # drop every page this run regenerated successfully (it is in `processed`
        # but not in the local `needs_regen` set).
        still_flagged = (set(old.get("needs_regen", [])) | needs_regen) - (
            set(processed) - needs_regen)
        files = {**old.get("files", {}), **{r: scanned[r] for r in processed}}
        page_keys = {**old.get("page_gen_keys", {}), **processed}
        fm.save_manifest(
            files,
            set_gen_key(shards), dict(shards[0].gen_parts),
            page_gen_keys=page_keys,
            needs_regen=still_flagged,
        )
        manifest_cache = {"files": files, "page_gen_keys": page_keys,
                          "needs_regen": sorted(still_flagged)}
        cache_age += 1

    async def one(rel: str) -> bool:
        """Generate one page. Returns False when it could not be served and the
        caller should stop the run (every provider down past the wait budget)."""
        nonlocal done_counter
        page = await asyncio.to_thread(extract_page, fm.root / rel, root=fm.root,
                                       max_chars=max_input_chars)
        preferred = shard_index(rel, len(shards))

        if page is None:
            # Nothing an LLM can do: record an empty corpus, no fallback flag.
            shard = shards[preferred]
            corpus, stats = _empty_page_corpus(rel, scanned[rel], shard.gen_key)
        else:
            corpus = None
            stats = None
            # Walk the healthy candidate list; on a hard API failure move to the
            # next provider (its gen_key is recorded, which keeps the per-page
            # key registry honest under failover).
            candidates = pool.order(preferred)
            if not candidates:
                # Every circuit is open — typically the shared quota window is
                # exhausted (observed: token-plan 5-hour limit, kimi 5-hour limit).
                # Burning the work list into heuristic fallback chunks would write
                # ~37k aux-less pages and then report success, so WAIT for a
                # provider to come back instead. Bounded by wait_budget_s.
                waited = 0.0
                while (not candidates and waited < wait_budget_s
                       and not interrupted["v"]):
                    nap = min(max(pool.next_probe_in(), 15.0), 120.0)
                    if progress and int(waited) % 300 < int(nap):
                        print(f"  [paused] all provider circuits open "
                              f"({pool.summary()}); waiting {nap:.0f}s for quota "
                              f"recovery — {waited:.0f}s/{wait_budget_s:.0f}s budget",
                              file=sys.stderr)
                    await asyncio.sleep(nap)
                    waited += nap
                    candidates = pool.order(preferred)
                if not candidates:
                    if interrupted["v"]:
                        # Ctrl-C while paused waiting for quota: drop this page and
                        # let the worker pool unwind (pages already finished are on
                        # disk; the run does NOT report "quota exhausted").
                        return False
                    if progress:
                        print(f"  [abort] provider wait budget exhausted "
                              f"({wait_budget_s:.0f}s); stopping the run with "
                              f"{len(work) - report.pages_done} pages unprocessed. "
                              f"Rerun when quota resets — nothing was degraded.",
                              file=sys.stderr)
                    report.aborted_all_providers_down = True
                    return False
            for idx in candidates:
                shard = shards[idx]
                async with sems[idx]:
                    try:
                        corpus, stats = await generate_page_corpus(
                            shard.client, page, gen_key=shard.gen_key,
                            html_md5=scanned[rel], max_chunk_chars=max_chunk_chars)
                    except CorpusGenerationError as exc:
                        # The model's JSON survived json_repair AND every
                        # self-repair session. The API itself is healthy (no
                        # failover — a sibling provider would face the same
                        # page), so degrade THIS page to heuristic chunks and
                        # flag it for a later run; one pathological page must
                        # never wedge a 44k-page build.
                        corpus, stats = _unrepairable_output(
                            page, scanned[rel], shard.gen_key, exc)
                if stats.api_failed:
                    pool.record_api_failure(idx)
                    corpus = stats = None
                    continue
                pool.record_success(idx)
                break
            if corpus is None:  # candidates exhausted by fresh API failures
                corpus, stats = _all_providers_down(page, scanned[rel],
                                                    shards[preferred].gen_key)

        used_key = corpus.gen_key

        # Heuristic-fallback pages (API death or unparseable output) are still
        # written so search works, but flagged so a later run retries them.
        # Pages with no extractable content are permanently empty — flagging
        # those would requeue them forever, so they are excluded.
        is_fallback = (page is not None and stats.fallback
                       and not stats.first_try_valid and not stats.repaired)
        if is_fallback:
            corpus.needs_regen = True
            needs_regen.add(rel)

        store.save(rel, corpus.to_json_dict())
        async with lock:
            processed[rel] = used_key
            report.pages_done += 1
            report.chunks += len(corpus.chunks)
            report.input_tokens += stats.input_tokens
            report.output_tokens += stats.output_tokens
            report.first_try += int(stats.first_try_valid)
            report.repaired += int(stats.repaired and not stats.first_try_valid)
            report.fallback += int(is_fallback)
            if stats.fallback or stats.errors:
                report.failures.append({"source": rel, "errors": stats.errors[:6]})
                await asyncio.to_thread(write_failure,
                                        {"source": rel, "errors": stats.errors})
            done_counter += 1
            # Incremental flush: once per finished page by default, so the
            # manifest on disk is never more than the in-flight pages behind the
            # work the run has already paid for. Off-loop: the atomic rewrite
            # must not stall the other workers' LLM requests.
            if checkpoint_every > 0 and done_counter % checkpoint_every == 0:
                await asyncio.to_thread(checkpoint)
            # Progress: once per finished page by default (no stride), so the
            # percentage is the true per-page completion. `stats.attempts` is
            # this page's session count — 1 for a clean page. progress_every
            # > 1 batches the line, 0 silences it (the final report still runs).
            if progress and progress_every > 0 and (
                    progress_every == 1
                    or report.pages_done % progress_every == 0):
                print(progress_line(report.pages_done, len(work),
                                    time.time() - t0, report.chunks, rel,
                                    sessions=stats.attempts,
                                    fallback=report.fallback),
                      file=sys.stderr)
        return True

    # Bounded worker pool: keeps memory flat on a 44k-page work list instead of
    # materialising one coroutine per page up front.
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    for rel in work:
        queue.put_nowait(rel)
    n_workers = max(1, workers_per_provider * len(shards))

    async def worker() -> None:
        while not aborted["v"]:
            try:
                rel = queue.get_nowait()
            except asyncio.QueueEmpty:
                return  # queue is pre-filled; empty means the run is over
            try:
                if await one(rel) is False:
                    # every provider is down past the wait budget: stop the whole
                    # run so the remaining pages stay untouched for a later retry
                    aborted["v"] = True
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # never let one page kill the run
                log.error("page %s failed hard: %s", rel, exc)
                async with lock:
                    report.failures.append({"source": rel,
                                            "errors": [f"hard error: {exc}"]})

    # Ctrl-C is handled HERE rather than left to asyncio.run: the FIRST interrupt
    # makes the workers stop picking up new pages, so the run reaches its normal
    # end below with every in-flight page finished and checkpointed (zero pages
    # lost). A SECOND one reaches the loop as an unwinding KeyboardInterrupt,
    # which asyncio deliberately re-raises straight out of the event loop
    # (``asyncio.tasks``: ``except (KeyboardInterrupt, SystemExit): raise``) — so
    # it CANNOT be caught here; rag.compile's own handler turns it into the
    # friendly exit. The ``finally`` below still flushes on that path: the loop
    # tears the tasks down through Runner.close() and this coroutine's cleanup
    # runs there.
    prev_sigint = install_interrupt_stop(aborted, interrupted)
    try:
        await asyncio.gather(*(worker() for _ in range(n_workers)))
    finally:
        restore_interrupt_stop(prev_sigint)
        # Final flush. With the per-page default this is a no-op in content terms
        # (every finished page was already checkpointed) and exists for the batched
        # configurations, for pages recorded after the last modulo boundary, and
        # for an abort — a ``checkpoint_every`` > 1 configuration would otherwise
        # lose its last batch.
        await asyncio.to_thread(checkpoint)
    report.wall_s = time.time() - t0
    report.needs_regen = set(needs_regen)
    report.interrupted = interrupted["v"]
    if progress:
        print(f"[corpus] providers: {pool.summary()}", file=sys.stderr)
        if needs_regen:
            print(f"[corpus] {len(needs_regen)} pages fell back to heuristic "
                  f"chunks and are flagged needs_regen — rerun compile to retry "
                  f"them (they are still indexed meanwhile)", file=sys.stderr)
    return report


def _all_providers_down(page, html_md5: str, gen_key: str):
    """Heuristic corpus for a page no provider could serve (all circuits open).

    Flagged needs_regen so the next run retries it with real LLM chunks.
    """
    from rag.corpus.generate import PageGenStats, _fallback_corpus

    stats = PageGenStats(source=page.source)
    stats.fallback = True
    stats.api_failed = True
    stats.errors.append("fallback: all providers unavailable (circuits open)")
    corpus = _fallback_corpus(page, gen_key)
    corpus.html_md5 = html_md5
    return corpus, stats


def _unrepairable_output(page, html_md5: str, gen_key: str,
                         exc: CorpusGenerationError) -> tuple[PageCorpus, Any]:
    """Heuristic corpus for a page whose model output survived every recovery
    step (strict decode, truncated-closer repair, json_repair, and all
    self-repair sessions).

    Same contract as the API-failure fallback: flagged needs_regen so a later
    run retries the page (a model or prompt update may un-stick it). The
    generation stats attached to the exception (attempts, tokens, error list)
    are preserved so the report reflects the spent generations.
    """
    from rag.corpus.generate import PageGenStats, _fallback_corpus

    stats = exc.stats if exc.stats is not None else PageGenStats(source=page.source)
    stats.fallback = True
    stats.errors.append(f"fallback: {exc}")
    corpus = _fallback_corpus(page, gen_key)
    corpus.html_md5 = html_md5
    return corpus, stats


def _empty_page_corpus(rel: str, html_md5: str, gen_key: str) -> tuple[PageCorpus, Any]:
    """Pages with no extractable content still get a (chunk-less) corpus record."""
    from rag.corpus.generate import PageGenStats
    from rag.corpus.schema import now_iso
    stats = PageGenStats(source=rel)
    stats.fallback = True
    stats.errors.append("no extractable content")
    return PageCorpus(source=rel, title="", html_md5=html_md5, gen_key=gen_key,
                      generated_at=now_iso(), chunks=[]), stats


def finalize(fm: FileManager, diff: ManifestDiff, files_state: dict[str, str],
             page_gen_keys: dict[str, str], gen_key: str, gen_parts: dict,
             report: CompileReport, *,
             needs_regen: set[str] | None = None) -> None:
    """Prune removed pages and write the final manifest.

    Two honesty rules, both of which make the manifest self-healing:

    1. A page is only claimed as processed when its corpus file actually exists.
       Callers pass the full scan, so a ``--max-files 100`` run would otherwise
       mark all 43k pages done and the rest would never be generated.
    2. The heuristic-fallback flag set is taken from the manifest already on
       disk: :func:`run_corpus_compile` checkpoints the correctly merged value
       (previous flags, plus this run's fallbacks, minus pages regenerated
       successfully). Re-deriving it from ``page_gen_keys`` would be wrong
       because callers pass a merged dict covering every page ever processed.
    """
    fm.prune(diff.removed)
    report.pruned = len(diff.removed)

    store = CorpusStore(fm.corpus_dir)
    # The filesystem is the only ground truth for "was this page generated".
    # An earlier version also accepted `rel in old_files`, which let phantom
    # entries inherited from a previous broken run survive forever — the manifest
    # kept claiming all 43,938 pages while ~18k corpus files existed. Filter on
    # existence alone so finalize is self-healing.
    honest_files = {rel: md5 for rel, md5 in files_state.items()
                    if store.exists(rel)}
    honest_keys = {rel: k for rel, k in page_gen_keys.items() if rel in honest_files}

    flagged = (set(fm.load_manifest().get("needs_regen", []))
               if needs_regen is None else set(needs_regen))
    # drop flags for pages we no longer claim (e.g. pruned)
    flagged &= set(honest_files)
    fm.save_manifest(honest_files, gen_key, gen_parts,
                     page_gen_keys=honest_keys, needs_regen=flagged)
    report.pages_claimed = len(honest_files)


def print_report(report: CompileReport, out=sys.stdout) -> None:
    print(f"\n== compile report ==", file=out)
    print(f"pages total={report.pages_total} planned={report.pages_planned} "
          f"done={report.pages_done}", file=out)
    print(f"chunks written={report.chunks}", file=out)
    print(f"first-try valid={report.first_try} repaired={report.repaired} "
          f"fallback={report.fallback}", file=out)
    print(f"tokens: input={report.input_tokens:,} output={report.output_tokens:,}",
          file=out)
    print(f"pruned corpus files={report.pruned}", file=out)
    print(f"wall time: {report.wall_s:.0f}s", file=out)
    if report.needs_regen:
        print(f"needs_regen: {len(report.needs_regen)} pages fell back to "
              f"heuristic chunks — rerun compile to retry them", file=out)
    if report.aborted_all_providers_down:
        print(f"ABORTED: every provider circuit stayed open past the wait budget "
              f"(quota exhausted). {report.pages_planned - report.pages_done} pages "
              f"were left untouched rather than degraded to heuristic chunks — "
              f"rerun when the quota window resets.", file=out)
    if report.interrupted:
        print(interrupt_message(report), file=out)
    if report.failures:
        print(f"pages with errors/fallback: {len(report.failures)} (see "
              f"corpus/failures.jsonl)", file=out)
