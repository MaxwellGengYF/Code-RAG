"""Provider-config parsing tests (network-free)."""
from __future__ import annotations

import pytest

from rag import ROOT
from rag.llm.config import ProviderConfig

QWEN_FLASH = {
    "model": "qwen3.8-flash",
    "max_context_size": 1000000,
    "capabilities": ["thinking"],
    "url": "https://token-plan.cn-beijing.maas.aliyuncs.com/apps/anthropic",
    "type": "anthropic",
    "env": {},
    "api_key": "sk-test",
    "max_tokens": 131072,
    "thinking_effort": "low",
    "services": {"search": {"base_url": "https://example.invalid/v1/search",
                            "api_key": "sk-x"}},
}

K27 = {
    "model": "kimi-for-coding",
    "max_context_size": 1000000,
    "capabilities": ["thinking", "image_in"],
    "url": "https://api.kimi.com/coding/v1",
    "type": "kimi",
    "max_tokens": 384000,
    "show_thinking_stream": True,
    "thinking_effort": "max",
    "api_key": "sk-test2",
}


def test_parse_anthropic_type(tmp_path):
    p = tmp_path / "qwen_flash.json"
    p.write_text(__import__("json").dumps(QWEN_FLASH), encoding="utf-8")
    cfg = ProviderConfig.from_file(p)
    assert cfg.model == "qwen3.8-flash"
    assert cfg.type == "anthropic"
    assert cfg.base_url == "https://token-plan.cn-beijing.maas.aliyuncs.com/apps/anthropic"
    assert cfg.api_key == "sk-test"
    assert cfg.max_tokens == 131072
    assert cfg.thinking_enabled
    assert cfg.thinking_effort == "low"
    assert cfg.path == p


def test_parse_kimi_type_ignores_unknown_keys(tmp_path):
    p = tmp_path / "k27.json"
    p.write_text(__import__("json").dumps(K27), encoding="utf-8")
    cfg = ProviderConfig.from_file(p)
    assert cfg.type == "kimi"
    assert cfg.thinking_enabled
    assert "image_in" in cfg.capabilities
    assert cfg.thinking_effort == "max"  # non-standard effort passes through


def test_unknown_type_rejected():
    with pytest.raises(ValueError, match="unknown 'type'"):
        ProviderConfig.from_dict({"model": "m", "type": "teapot"})


def test_missing_model_rejected():
    with pytest.raises(ValueError, match="'model'"):
        ProviderConfig.from_dict({"type": "kimi"})


def test_missing_type_rejected():
    with pytest.raises(ValueError, match="'type'"):
        ProviderConfig.from_dict({"model": "m"})


def test_env_application(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_TEST_ENV_VAR", raising=False)
    p = tmp_path / "env.json"
    p.write_text(__import__("json").dumps({
        "model": "m", "type": "openai_legacy", "env": {"RAG_TEST_ENV_VAR": "hello"},
    }), encoding="utf-8")
    cfg = ProviderConfig.from_file(p)
    assert cfg.env == {"RAG_TEST_ENV_VAR": "hello"}
    import os
    assert os.environ["RAG_TEST_ENV_VAR"] == "hello"


def test_from_file_relative_path_anchors_at_root():
    # The llama_cpp/provider-qwen35-local.json provider config must
    # resolve no matter which CWD the process was started from.
    cfg = ProviderConfig.from_file("llama_cpp/provider-qwen35-local.json")
    assert cfg.path == ROOT / "llama_cpp" / "provider-qwen35-local.json"
    assert cfg.type == "llama"


def test_from_file_absolute_path_untouched(tmp_path):
    p = tmp_path / "abs.json"
    p.write_text('{"model": "m", "type": "llama"}', encoding="utf-8")
    cfg = ProviderConfig.from_file(p)
    assert cfg.path == p


def test_type_aliases():
    for alias, canonical in [("openai", "openai_legacy"), ("claude", "anthropic"),
                             ("moonshot", "kimi"), ("responses", "openai_responses")]:
        cfg = ProviderConfig.from_dict({"model": "m", "type": alias})
        assert cfg.type == canonical
