"""Tool-free OpenAI Chat Completions client (trimmed vend of kosong OpenAILegacy).

Kept: chat.completions, system+user roles, ``reasoning_content`` extra-key support,
Moonshot-style ``extra_body`` thinking/reasoning gating (opt-in), output-token clamping.
Stripped: tools, ``Message``/``ToolCall`` machinery, tool-call-id normalization,
streaming (compile-time batch work does not need incremental display).
"""
from __future__ import annotations

from typing import Any

import httpx
from openai import AsyncOpenAI

from .base import GenerationResult
from ._openai_shared import (
    build_thinking_extra_body,
    clamp_max_tokens,
    convert_openai_error,
)
from .config import ProviderConfig

#: efforts the standard OpenAI SDK ``reasoning_effort`` field accepts; anything else
#: (e.g. "max") is forwarded via extra_body to bypass SDK-side validation.
_SDK_EFFORTS = frozenset({"minimal", "low", "medium", "high"})


class OpenAILegacyClient:
    """Single-turn chat.completions client for OpenAI-compatible APIs."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        reasoning_key: str | None = None,
        extra_body_thinking: bool | None = None,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int = 0,
        **client_kwargs: Any,
    ):
        self._config = config
        # Moonshot's own API speaks the standard reasoning_effort/reasoning_content wire
        # format; the extra_body keys are for other OpenAI-compatible gateways.
        is_moonshot = config.model.startswith("kimi-") or (
            config.base_url is not None and "moonshot" in config.base_url.lower()
        )
        self._extra_body_thinking = (not is_moonshot) if extra_body_thinking is None else extra_body_thinking
        self._reasoning_key = reasoning_key
        kwargs: dict[str, Any] = dict(client_kwargs)
        if http_client is not None:
            kwargs["http_client"] = http_client
        if config.timeout:
            kwargs["timeout"] = config.timeout
        self._client = AsyncOpenAI(
            api_key=config.api_key, base_url=config.base_url,
            max_retries=max_retries, **kwargs,
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    async def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        cfg = self._config
        messages: list[dict[str, Any]] = []
        if system_prompt:
            # `system` (not `developer`) for max OpenAI-compatible acceptance (kosong parity).
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: dict[str, Any] = {}
        if cfg.max_tokens:
            kwargs["max_tokens"] = cfg.max_tokens
        clamp_max_tokens(kwargs)
        kwargs.pop("max_output_tokens", None)  # Responses-API param, not accepted here

        effort = cfg.thinking_effort if cfg.thinking_enabled else None
        thinking_enabled = effort is not None and effort != "off"

        if self._extra_body_thinking:
            kwargs["extra_body"] = build_thinking_extra_body(thinking_enabled, effort)
        # Bypass SDK validation for non-standard effort values ("max", ...) while still
        # sending them in the body.
        if effort is not None and effort not in _SDK_EFFORTS:
            extra_body = kwargs.setdefault("extra_body", {})
            extra_body["reasoning_effort"] = effort
        if effort in _SDK_EFFORTS:
            kwargs["reasoning_effort"] = effort  # type: ignore[dict-item]

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
        msg = choice.message
        text = msg.content or ""
        thinking: str | None = None
        if self._reasoning_key:
            extra: Any = getattr(msg, self._reasoning_key, None)
            if isinstance(extra, str) and extra:
                thinking = extra
        elif getattr(msg, "reasoning_content", None):
            thinking = msg.reasoning_content  # type: ignore[union-attr]
        usage = response.usage
        return GenerationResult(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            thinking=thinking,
        )
