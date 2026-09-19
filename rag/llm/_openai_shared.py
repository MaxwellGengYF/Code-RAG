"""Shared helpers for the OpenAI-SDK-based providers (openai_legacy / kimi)."""
from __future__ import annotations

from typing import Any

import httpx
from openai import (
    APIConnectionError as _OpenAIAPIConnectionError,
)
from openai import (
    APIStatusError as _OpenAIAPIStatusError,
)
from openai import (
    APITimeoutError as _OpenAIAPITimeoutError,
)
from openai import (
    AuthenticationError as _OpenAIAuthenticationError,
)
from openai import (
    OpenAIError,
    RateLimitError as _OpenAIRateLimitError,
)

from .base import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

#: output-token budgets are clamped to this (kosong._MAX_OUTPUT_TOKENS) so an
#: over-large ``max_tokens`` from the config layer (e.g. the context size) never
#: triggers a 400 from the API's per-model output limit.
MAX_SAFE_OUTPUT_TOKENS = 384_000


def clamp_max_tokens(kwargs: dict[str, Any]) -> None:
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        raw = kwargs.get(key)
        if raw is not None and raw > MAX_SAFE_OUTPUT_TOKENS:
            kwargs[key] = MAX_SAFE_OUTPUT_TOKENS


def convert_openai_error(exc: BaseException) -> BaseException:
    """Map an openai-SDK exception onto the vendored error hierarchy."""
    if isinstance(exc, _OpenAIRateLimitError):
        return RateLimitError(str(exc))
    if isinstance(exc, _OpenAIAPITimeoutError):
        return APITimeoutError(str(exc))
    if isinstance(exc, _OpenAIAPIConnectionError):
        return APIConnectionError(str(exc))
    if isinstance(exc, _OpenAIAPIStatusError):
        err = APIStatusError(str(exc), status_code=exc.status_code)
        if isinstance(exc, _OpenAIAuthenticationError):
            # 401 must never be retried
            return APIStatusError(str(exc), status_code=401)
        return err
    if isinstance(exc, OpenAIError):
        return APIStatusError(str(exc))
    if isinstance(exc, httpx.HTTPError):
        return APIConnectionError(str(exc))
    return exc


def effort_to_extra_body_level(effort: str | None) -> str:
    """Map a thinking-effort value to the three-level extra_body string."""
    if effort in (None, "off"):
        return "no_think"
    if effort in ("low", "minimal"):
        return "low"
    if effort == "medium":
        return "medium"
    if effort == "high":
        return "high"
    return effort  # non-standard (max, xhigh, ...) passes through verbatim


def build_thinking_extra_body(enabled: bool, effort: str | None) -> dict[str, Any]:
    """Moonshot-style / qwen-compatible gateway extra_body gating (kosong parity)."""
    level = effort_to_extra_body_level(effort if enabled else None)
    return {
        "thinking": {"type": "enabled" if enabled else "disabled"},
        "reasoning": {"effort": level},
        "chat_template_kwargs": {"reasoning_effort": level},
    }
