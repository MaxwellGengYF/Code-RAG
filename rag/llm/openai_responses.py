"""Tool-free OpenAI Responses API client (trimmed vend of kosong OpenAIResponses).

Kept: ``responses.create`` with ``store=False`` (no provider session state),
``developer`` role for OpenAI models, reasoning effort via extra_body, usage fields.
Stripped: tools / function-call output handling, streaming, input persistence.
"""
from __future__ import annotations

from typing import Any

import httpx
from openai import AsyncOpenAI

from .base import GenerationResult
from ._openai_shared import clamp_max_tokens, convert_openai_error
from .config import ProviderConfig

#: model families that require the ``developer`` role instead of ``system``
_DEVELOPER_ROLE_PREFIXES = ("o1", "o3", "o4", "gpt-5", "codex", "computer-use")


class OpenAIResponsesClient:
    """Single-turn responses.create client. ``store=False`` always (plan contract)."""

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
        self._client = AsyncOpenAI(
            api_key=config.api_key, base_url=config.base_url,
            max_retries=max_retries, **kwargs,
        )

    @property
    def model_name(self) -> str:
        return self._config.model

    async def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        cfg = self._config
        role = "developer" if cfg.model.lower().startswith(_DEVELOPER_ROLE_PREFIXES) else "system"
        instructions = system_prompt or None
        # Responses API: keep everything in ``input`` so no server-side session is
        # created; ``store=False`` below guarantees nothing is retained either way.
        input_messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]

        kwargs: dict[str, Any] = {"store": False}
        if cfg.max_tokens:
            kwargs["max_output_tokens"] = cfg.max_tokens
        clamp_max_tokens(kwargs)

        extra_body: dict[str, Any] = {}
        effort = cfg.thinking_effort if cfg.thinking_enabled else None
        if effort:
            extra_body["reasoning"] = {"effort": effort}
        if extra_body:
            kwargs["extra_body"] = extra_body

        try:
            response = await self._client.responses.create(
                model=cfg.model,
                instructions=instructions,
                input=input_messages,
                **kwargs,
            )
        except Exception as exc:
            raise convert_openai_error(exc) from exc

        text = ""
        thinking: str | None = None
        for item in getattr(response, "output", None) or []:
            item_type = getattr(item, "type", "")
            if item_type == "message":
                for part in getattr(item, "content", None) or []:
                    if getattr(part, "type", "") == "output_text":
                        text += part.text or ""
            elif item_type == "reasoning":
                summary = getattr(item, "summary", None) or []
                thinking = "\n".join(
                    p.text for p in summary if getattr(p, "type", "") == "summary_text"
                ) or thinking
        usage = response.usage
        return GenerationResult(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            thinking=thinking,
        )
