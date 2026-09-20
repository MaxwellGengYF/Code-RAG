"""Provider-config JSON parsing (the qwen_flash.json / k27.json format).

Recognized keys (verified against real config files)::

    {
      "model": "...",                  # required
      "max_context_size": 1000000,     # informational
      "capabilities": ["thinking", "image_in"],
      "url": "<base_url>",             # mapped to base_url
      "type": "anthropic|kimi|openai_legacy|openai_responses",
      "api_key": "sk-...",
      "max_tokens": 131072,
      "thinking_effort": "low|medium|high|max|...",
      "env": {},                       # applied to os.environ (values must be strings)
      "services": {...}               # ignored (agent services, not generation)
      // unknown keys are ignored with a logged warning
    }
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import resolve_path

log = logging.getLogger(__name__)

#: accepted ``type`` values -> canonical provider names
_TYPE_ALIASES = {
    "openai_legacy": "openai_legacy",
    "openai": "openai_legacy",
    "openai_responses": "openai_responses",
    "responses": "openai_responses",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "kimi": "kimi",
    "moonshot": "kimi",
    "llama": "llama",
    "llama_cpp": "llama",
    "llama.cpp": "llama",
    "llama-server": "llama",
}

_KNOWN_KEYS = {
    "model", "max_context_size", "capabilities", "url", "base_url", "type",
    "api_key", "api_key_env", "max_tokens", "thinking_effort", "env",
    "services", "name", "description", "timeout",
    # llama.cpp (local server) provider extras, consumed via ProviderConfig.raw
    "server_bin", "server_cmd", "model_path", "host", "port", "ngl", "ctx_size",
    "extra_args", "extra_body", "start_timeout",
}


@dataclass
class ProviderConfig:
    """Normalized provider configuration."""

    model: str
    type: str  # canonical: openai_legacy | openai_responses | anthropic | kimi
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int | None = None
    max_context_size: int | None = None
    thinking_effort: str | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)
    timeout: float | None = None  # seconds, None = SDK default
    env: dict[str, str] = field(default_factory=dict)
    path: Path | None = None  # where this config was loaded from
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def thinking_enabled(self) -> bool:
        return "thinking" in self.capabilities

    @classmethod
    def from_file(cls, path: str | Path) -> "ProviderConfig":
        # Relative provider paths anchor at the repo root, so documented
        # commands work from any CWD.
        p = resolve_path(path)
        if not p.exists():
            raise FileNotFoundError(f"provider config not found: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid provider config JSON {p}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"provider config {p} must be a JSON object")
        cfg = cls.from_dict(data)
        cfg.path = p
        return cfg

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderConfig":
        model = data.get("model")
        if not model or not isinstance(model, str):
            raise ValueError("provider config: 'model' (string) is required")

        raw_type = str(data.get("type", "")).strip().lower()
        if not raw_type:
            raise ValueError(
                "provider config: 'type' is required, one of "
                + ", ".join(sorted(set(_TYPE_ALIASES.values())))
            )
        canonical = _TYPE_ALIASES.get(raw_type)
        if canonical is None:
            raise ValueError(
                f"provider config: unknown 'type' {raw_type!r}; expected one of "
                + ", ".join(sorted(_TYPE_ALIASES))
            )

        for key in data:
            if key not in _KNOWN_KEYS:
                log.warning("provider config: ignoring unknown key %r", key)

        env = data.get("env") or {}
        if not isinstance(env, dict):
            raise ValueError("provider config: 'env' must be an object")
        env = {str(k): str(v) for k, v in env.items()}
        # Apply env vars (setdefault: an already-set real environment wins).
        for k, v in env.items():
            os.environ.setdefault(k, v)

        api_key = data.get("api_key")
        if api_key is None and data.get("api_key_env"):
            api_key = os.environ.get(str(data["api_key_env"]))

        capabilities = data.get("capabilities") or []
        if not isinstance(capabilities, list):
            raise ValueError("provider config: 'capabilities' must be a list")

        return cls(
            model=model,
            type=canonical,
            base_url=data.get("base_url") or data.get("url"),
            api_key=api_key,
            max_tokens=data.get("max_tokens"),
            max_context_size=data.get("max_context_size"),
            thinking_effort=data.get("thinking_effort"),
            capabilities=frozenset(str(c) for c in capabilities),
            timeout=data.get("timeout"),
            env=env,
            raw=dict(data),
        )
