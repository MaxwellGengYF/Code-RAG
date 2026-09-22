"""Tool-free LLM client protocol, result type, error types, and retry helper.

This backend is deliberately minimal: a single-turn (system prompt + user prompt)
generation with NO tools, NO multi-turn history, NO provider session state. That is
all the RAG compile step needs, and it keeps the vendored providers small enough to
audit. Wire-format behaviour (extra_body thinking gating, Moonshot defaults, ...)
follows kosong's chat providers, trimmed accordingly.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol, Sequence


@dataclass
class GenerationResult:
    """One completed generation."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking: str | None = None  # reasoning content when the model produced it


class LLMClient(Protocol):
    """The only interface the compile pipeline depends on."""

    @property
    def model_name(self) -> str: ...

    async def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult: ...


# --------------------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for all vendored-backend errors."""


class APIStatusError(LLMError):
    """A non-2xx API response. ``status_code`` distinguishes retriable (429/5xx)."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class APITimeoutError(LLMError):
    """Request timed out."""


class APIConnectionError(LLMError):
    """Network-level failure (DNS, refused, reset, ...)."""


class RateLimitError(APIStatusError):
    """HTTP 429."""

    def __init__(self, message: str):
        super().__init__(message, status_code=429)


class RetryAborted(LLMError):
    """A retry backoff was cut short by a caller-requested abort (Ctrl-C).

    NOT a provider failure: nothing was sent and nothing has been written, so the
    caller drops the in-flight work instead of degrading it to a heuristic
    fallback. Raised by :func:`with_retry` when the ``should_abort`` callback
    reports an abort while it is waiting out the schedule — which is what keeps a
    multi-hour backoff responsive to Ctrl-C.
    """


# --------------------------------------------------------------------------------------
# retry
# --------------------------------------------------------------------------------------

#: status codes / error kinds worth retrying
_RETRIABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: The retry schedule for transient provider failures — HTTP 429 rate limits,
#: 5xx, request timeouts, connection resets: one entry per retry, in seconds,
#: ``2 s -> 4 s -> 1 min -> 10 min -> 1 h -> 2 h -> 4 h``.
#:
#: Why this shape: the first two steps absorb ordinary throttling (these gateways
#: start 429-ing past ~4 concurrent requests per key and recover in seconds), while
#: the long tail rides out an exhausted ROLLING QUOTA WINDOW — the providers this
#: project talks to reset on 5-hour windows, and a run that waits keeps its pages
#: instead of grinding tens of thousands of them into aux-less heuristic chunks.
#: Override per config with ``"retry_delays": [seconds, ...]``.
RETRY_DELAYS: tuple[float, ...] = (2.0, 4.0, 60.0, 600.0, 3600.0, 7200.0, 14400.0)

#: how often a long backoff re-checks ``should_abort`` (seconds), so Ctrl-C is
#: still honoured within a few seconds during a multi-hour wait
_ABORT_POLL_S = 5.0


def fmt_delay(seconds: float) -> str:
    """Human scale for a backoff wait: 2s / 4s / 1min / 10min / 1h / 2h / 4h."""
    if seconds < 60:
        return f"{seconds:g}s"
    if seconds < 3600:
        return f"{seconds / 60:g}min"
    return f"{seconds / 3600:g}h"


def is_retriable(exc: BaseException) -> bool:
    """Whether *exc* is a transient failure that deserves a retry.

    Covers rate limiting (429), server-side blips (408/409/425/5xx) and
    transport failures (timeout/connection). Permanent 4xx (401 auth, 403 quota
    or revoked access, 404) are deliberately NOT retriable here: the compile
    pipeline's circuit breaker and provider failover own those.
    """
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRIABLE_STATUSES
    return False


def retry_schedule(
    delays: Sequence[float] | None = None,
    retries: int | None = None,
) -> tuple[float, ...]:
    """Normalize a retry schedule into the per-retry wait list (seconds).

    ``delays`` is the schedule and defaults to :data:`RETRY_DELAYS`; ``retries``
    only CAPS how many of its waits are used (``0`` disables retrying, ``None``
    means all of them) — the schedule's length stays the hard maximum, so a
    shortened ``retry_delays`` config is never padded back out. Raises
    ``ValueError`` on an empty/negative schedule so a bad ``retry_delays`` config
    fails loudly at startup instead of once per page.
    """
    seq = tuple(float(d) for d in (RETRY_DELAYS if delays is None else delays))
    if not seq:
        raise ValueError("is empty")
    if any(d < 0 for d in seq):
        raise ValueError(f"entries must be >= 0 seconds, got {list(seq)}")
    if retries is None:
        return seq
    return seq[:max(0, retries)]


async def _sleep_interruptibly(delay: float,
                               should_abort: Callable[[], bool] | None) -> None:
    """Sleep *delay* seconds, raising :class:`RetryAborted` if an abort lands.

    The wait is slept in ``_ABORT_POLL_S`` slices so a 4-hour backoff still
    reacts to Ctrl-C within seconds instead of hours. ``CancelledError`` is a
    ``BaseException`` and is deliberately left to propagate (an external cancel
    must stay a cancellation, not become a retry abort).
    """
    if should_abort is None:
        await asyncio.sleep(delay)
        return
    remaining = delay
    while True:
        if should_abort():
            raise RetryAborted(
                f"retry backoff interrupted with {remaining:.0f}s still to wait")
        if remaining <= 0:
            return
        slice_s = min(remaining, _ABORT_POLL_S)
        await asyncio.sleep(slice_s)
        remaining -= slice_s


async def with_retry(
    fn: Callable[[], Awaitable[GenerationResult]],
    *,
    delays: Sequence[float] | None = None,
    retries: int | None = None,
    should_abort: Callable[[], bool] | None = None,
    on_retry: Callable[[BaseException, int, float], None] | None = None,
) -> GenerationResult:
    """Run *fn*, waiting out transient API failures on a fixed schedule.

    ``delays`` is the per-retry wait list in seconds (default
    :data:`RETRY_DELAYS`: 2s -> 4s -> 1min -> 10min -> 1h -> 2h -> 4h); ``retries``
    caps how many of those waits are used (``0`` disables retrying). Non-retriable
    errors raise immediately, and once the whole schedule is spent the last error
    is re-raised so the caller's circuit breaker / fallback takes over.

    ``on_retry(exc, n, delay)`` is called just before each wait (``n`` is the
    1-based retry number). ``should_abort`` turns a long wait into
    :class:`RetryAborted` (see :func:`_sleep_interruptibly`).
    """
    schedule = retry_schedule(delays, retries)
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:
            if not is_retriable(exc) or attempt >= len(schedule):
                raise
            delay = schedule[attempt]
            if on_retry is not None:
                on_retry(exc, attempt + 1, delay)
            await _sleep_interruptibly(delay, should_abort)
            attempt += 1
