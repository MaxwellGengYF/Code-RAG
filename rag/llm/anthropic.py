"""Tool-free Anthropic Messages API client (trimmed vend of kosong Anthropic).

Kept: ``messages.create`` with the ``system`` param, thinking config derived from
``thinking_effort``, ``max_tokens`` from the provider config, token usage.
Stripped: tools, tool_choice, betas, image parts (page input is text-only),
cache-control metadata, streaming.
"""
from __future__ import annotations

from typing import Any

import httpx
from anthropic import (
    APIConnectionError as _AnthropicAPIConnectionError,
)
from anthropic import (
    APIStatusError as _AnthropicAPIStatusError,
)
from anthropic import (
    APITimeoutError as _AnthropicAPITimeoutError,
)
from anthropic import (
    AsyncAnthropic,
    RateLimitError as _AnthropicRateLimitError,
)

from .base import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    GenerationResult,
    RateLimitError,
)
from .config import ProviderConfig

#: Anthropic models top out at 128k output tokens, and the Python SDK refuses
#: non-streaming requests whose expected duration exceeds 10 minutes
#: (128_000 * 10 / 60 = 21_333 tokens).
_MAX_OUTPUT_TOKENS = 128_000
_MAX_OUTPUT_TOKENS_NONSTREAMING = 21_333

#: thinking-effort -> thinking budget_tokens for the manual (non-adaptive) pathway
_EFFORT_BUDGETS = {
    "low": 1024,
    "medium": 4096,
    "high": 16384,
    "xhigh": 32768,
    "max": 32768,
}


def _convert_anthropic_error(exc: BaseException) -> BaseException:
    if isinstance(exc, _AnthropicRateLimitError):
        return RateLimitError(str(exc))
    if isinstance(exc, _AnthropicAPITimeoutError):
        return APITimeoutError(str(exc))
    if isinstance(exc, _AnthropicAPIConnectionError):
        return APIConnectionError(str(exc))
    if isinstance(exc, _AnthropicAPIStatusError):
        return APIStatusError(str(exc), status_code=exc.status_code)
    if isinstance(exc, httpx.HTTPError):
        return APIConnectionError(str(exc))
    return exc


class AnthropicClient:
    """Single-turn messages.create client."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int = 0,
        **client_kwargs: Any,
    ):
        self._config = config
        kwargs: dict[str, Any] = dict(client_kwargs)
        if http_client is not None:
            kwargs["http_client"] = http_client
        if config.timeout:
            kwargs["timeout"] = config.timeout
        self._client = AsyncAnthropic(
            api_key=config.api_key, base_url=config.base_url,
            max_retries=max_retries, **kwargs,
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    async def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        cfg = self._config
        max_tokens = min(cfg.max_tokens or 8192, _MAX_OUTPUT_TOKENS,
                         _MAX_OUTPUT_TOKENS_NONSTREAMING)

        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if system_prompt:
            # Anthropic takes the system prompt as a top-level param, not a message.
            kwargs["system"] = system_prompt

        effort = cfg.thinking_effort if cfg.thinking_enabled else None
        if effort and effort != "off":
            budget = _EFFORT_BUDGETS.get(effort, 4096)
            budget = min(budget, max(1024, max_tokens - 1))
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as exc:
            raise _convert_anthropic_error(exc) from exc

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in response.content:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "thinking":
                thinking_parts.append(block.thinking)
        usage = response.usage
        return GenerationResult(
            text="".join(text_parts),
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            thinking="\n".join(thinking_parts) or None,
        )
