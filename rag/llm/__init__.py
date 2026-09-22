"""Vendored, tool-free LLM backend.

``create_llm`` builds a client from a provider-config JSON (qwen_flash.json /
k27.json format) or an already-parsed :class:`ProviderConfig`.
"""
from __future__ import annotations

from pathlib import Path

from .base import (
    RETRY_DELAYS,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    GenerationResult,
    LLMClient,
    LLMError,
    RateLimitError,
    RetryAborted,
    fmt_delay,
    is_retriable,
    retry_schedule,
    with_retry,
)
from .config import ProviderConfig

__all__ = [
    "APIConnectionError", "APIStatusError", "APITimeoutError", "GenerationResult",
    "LLMClient", "LLMError", "ProviderConfig", "RETRY_DELAYS", "RateLimitError",
    "RetryAborted", "create_llm", "fmt_delay", "is_retriable", "retry_schedule",
    "with_retry",
]

_FACTORIES = {}


def _register() -> None:
    from .anthropic import AnthropicClient
    from .kimi import KimiClient
    from .llama import LlamaClient
    from .openai_legacy import OpenAILegacyClient
    from .openai_responses import OpenAIResponsesClient

    _FACTORIES.update({
        "openai_legacy": OpenAILegacyClient,
        "openai_responses": OpenAIResponsesClient,
        "anthropic": AnthropicClient,
        "kimi": KimiClient,
        "llama": LlamaClient,
    })


def create_llm(
    config: str | Path | ProviderConfig,
    *,
    http_client=None,
    **client_kwargs,
) -> LLMClient:
    """Build an :class:`LLMClient` from a provider config path or object."""
    if not _FACTORIES:
        _register()
    cfg = config if isinstance(config, ProviderConfig) else ProviderConfig.from_file(config)
    factory = _FACTORIES.get(cfg.type)
    if factory is None:  # unreachable: ProviderConfig validates the type
        raise LLMError(f"no client implementation for provider type {cfg.type!r}")
    kwargs = dict(client_kwargs)
    if http_client is not None:
        kwargs["http_client"] = http_client
    return factory(cfg, **kwargs)
