"""Tool-free LLM client protocol, result type, error types, and retry helper.

This backend is deliberately minimal: a single-turn (system prompt + user prompt)
generation with NO tools, NO multi-turn history, NO provider session state. That is
all the RAG compile step needs, and it keeps the vendored providers small enough to
audit. Wire-format behaviour (extra_body thinking gating, Moonshot defaults, ...)
follows kosong's chat providers, trimmed accordingly.
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol


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


# --------------------------------------------------------------------------------------
# retry
# --------------------------------------------------------------------------------------

#: status codes / error kinds worth retrying
_RETRIABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def is_retriable(exc: BaseException) -> bool:
    """Whether *exc* is a transient failure that deserves a retry."""
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRIABLE_STATUSES
    return False


async def with_retry(
    fn: Callable[[], Awaitable[GenerationResult]],
    *,
    retries: int = 4,
    backoff: float = 1.0,
    max_backoff: float = 30.0,
    on_retry: Callable[[BaseException, int], None] | None = None,
) -> GenerationResult:
    """Run *fn* with exponential-backoff retry on transient errors.

    ``retries`` is the number of RETRIES after the first attempt (so up to
    ``retries + 1`` attempts total). Non-retriable errors raise immediately.
    """
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:
            if not is_retriable(exc) or attempt >= retries:
                raise
            delay = min(max_backoff, backoff * (2 ** attempt)) * (0.5 + random.random())
            if on_retry is not None:
                on_retry(exc, attempt + 1)
            await asyncio.sleep(delay)
            attempt += 1
