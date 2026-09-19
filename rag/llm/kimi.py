"""Tool-free Kimi (Moonshot) chat.completions client (trimmed vend of kosong Kimi).

Kept: OpenAI-compatible chat.completions, Moonshot defaults
(``https://api.moonshot.ai/v1``), ``extra_body.thinking`` gating, ``prompt_cache_key``,
``max_completion_tokens`` normalization, temperature 1.0/0.6 by thinking state.
Stripped: tools, files API, video parts, tool-call-id normalization, streaming.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from openai import AsyncOpenAI

from .base import GenerationResult, LLMError
from ._openai_shared import clamp_max_tokens, convert_openai_error
from .config import ProviderConfig

_DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"


class KimiClient:
    """Single-turn chat.completions client for the Kimi/Moonshot API."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int = 0,
        **client_kwargs: Any,
    ):
        api_key = config.api_key or os.getenv("KIMI_API_KEY")
        if api_key is None:
            raise LLMError(
                "Kimi provider requires 'api_key' in the provider config or the "
                "KIMI_API_KEY environment variable"
            )
        base_url = config.base_url or os.getenv("KIMI_BASE_URL", _DEFAULT_BASE_URL)
        self._config = config
        self._base_url = base_url
        kwargs: dict[str, Any] = dict(client_kwargs)
        if http_client is not None:
            kwargs["http_client"] = http_client
        if config.timeout:
            kwargs["timeout"] = config.timeout
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, max_retries=max_retries, **kwargs,
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    async def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        cfg = self._config
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: dict[str, Any] = {}
        # Kimi prefers max_completion_tokens for reasoning models; max_tokens shares the
        # budget with reasoning_content and a small value can yield a 200 with no content.
        if cfg.max_tokens:
            kwargs["max_completion_tokens"] = cfg.max_tokens
        clamp_max_tokens(kwargs)

        effort = cfg.thinking_effort if cfg.thinking_enabled else None
        thinking_enabled = effort is not None and effort != "off"
        kwargs["extra_body"] = {
            "thinking": {
                "type": "enabled" if thinking_enabled else "disabled",
                **({"effort": effort} if thinking_enabled and effort else {}),
            }
        }
        # Moonshot-recommended sampling temperatures by thinking state.
        kwargs["temperature"] = 1.0 if thinking_enabled else 0.6
        # Stable prompt cache key: same system prompt across calls.
        kwargs["prompt_cache_key"] = f"rag-corpus-v1:{cfg.model}"

        try:
            response = await self._client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                stream=False,
                **kwargs,
            )
        except Exception as exc:
            raise convert_openai_error(exc) from exc

        choice = response.choices[0]
        text = choice.message.content or ""
        thinking = getattr(choice.message, "reasoning_content", None) or None
        usage = response.usage
        return GenerationResult(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            thinking=thinking,
        )
