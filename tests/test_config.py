"""Tests for the unified config loader (rag.config) and generic prompts.

The config file is BOTH the provider config and the RAG config: provider keys,
corpus locations (dirs/corpus_dir/index_dir) and engine tuning live in one JSON,
defaulting to ./config.json in the CWD when --config is not given.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.config import (BASE_DIR_KEY, DEFAULT_CONFIG_NAME, RagConfig,
                        config_dir, data_path, load_config, load_settings,
                        resolve_paths)


def _write(path: Path, obj: dict) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


PROVIDER = {"model": "m1", "type": "openai_legacy", "url": "http://x/v1",
            "api_key": "sk"}


def test_combined_file_yields_settings_and_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write(tmp_path / DEFAULT_CONFIG_NAME, {
        **PROVIDER,
        "dirs": ["docs"], "corpus_dir": "corpus", "index_dir": "index",
        "mode": "bm25", "rrf_k": 42,
    })
    rc = load_config(None)  # no --config -> ./config.json
    assert rc.path == cfg.resolve()
    assert rc.base_dir == tmp_path.resolve()
    assert rc.settings["dirs"] == ["docs"]
    assert rc.settings["rrf_k"] == 42
    assert len(rc.providers) == 1
    p = rc.providers[0]
    assert p.model == "m1" and p.type == "openai_legacy" and p.base_url == "http://x/v1"
    # provider keys are not duplicated into settings warnings; rag keys stay
    assert "model" in rc.settings and "mode" in rc.settings


def test_no_config_file_gives_empty_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = load_config(None)
    assert rc.path is None
    assert rc.settings == {BASE_DIR_KEY: str(tmp_path.resolve())}
    assert rc.providers == []


def test_explicit_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nope.json"))


def test_relative_output_dirs_anchor_at_config_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "proj"
    d.mkdir()
    _write(d / "cfg.json", {"dirs": ["docs"], "corpus_dir": "out/corpus",
                            "index_dir": "out/index"})
    rc = load_config(str(d / "cfg.json"))
    assert rc.settings["corpus_dir"] == str(d / "out" / "corpus")
    assert rc.settings["index_dir"] == str(d / "out" / "index")
    assert config_dir(rc.settings) == d.resolve()


def test_providers_list_adds_shards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / DEFAULT_CONFIG_NAME, {
        **PROVIDER,
        "dirs": ["docs"],
        "providers": [
            {"model": "m2", "type": "anthropic", "api_key": "sk2"},
            {"model": "m3", "type": "kimi", "api_key": "sk3"},
        ],
    })
    rc = load_config(None)
    assert [p.model for p in rc.providers] == ["m1", "m2", "m3"]


def test_first_config_wins_settings_but_all_providers_merge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _write(tmp_path / "a.json", {**PROVIDER, "dirs": ["docs"],
                                     "mode": "bm25", "rrf_k": 60})
    b = _write(tmp_path / "b.json", {"model": "m9", "type": "kimi",
                                     "api_key": "sk9", "mode": "dense"})
    rc = load_config([str(a), str(b)])
    assert rc.settings["mode"] == "bm25"      # first file wins
    assert rc.settings["rrf_k"] == 60
    assert [p.model for p in rc.providers] == ["m1", "m9"]


def test_provider_only_file_is_valid_for_corpus_step(tmp_path, monkeypatch):
    """A bare provider JSON (no rag keys) works as a --config shard."""
    monkeypatch.chdir(tmp_path)
    p = _write(tmp_path / "prov.json", PROVIDER)
    rc = load_config(str(p))
    assert len(rc.providers) == 1
    assert "dirs" not in rc.settings


def test_settings_only_file_has_no_providers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / DEFAULT_CONFIG_NAME,
           {"dirs": ["docs"], "corpus_dir": "corpus"})
    rc = load_config(None)
    assert rc.providers == []
    assert load_settings(None)["dirs"] == ["docs"]


def test_rag_keys_do_not_warn_as_unknown_provider_keys(tmp_path, recwarn):
    """The combined file must not log 'ignoring unknown key' for rag settings."""
    import logging
    logger = logging.getLogger("rag.llm.config")
    _write(tmp_path / "c.json", {**PROVIDER, "dirs": ["docs"],
                                 "corpus_dir": "c", "rrf_k": 60,
                                 "accept_legacy_models": ["old-model"]})
    with _capture_logs(logger) as records:
        load_config(str(tmp_path / "c.json"))
    assert not any("ignoring unknown key" in r.getMessage() for r in records)


class _capture_logs:
    def __init__(self, logger):
        self.logger = logger
        self.records = []

    def __enter__(self):
        import logging
        self.handler = logging.Handler()
        self.handler.emit = self.records.append
        self.logger.addHandler(self.handler)
        self.old_level = self.logger.level
        self.logger.setLevel(logging.WARNING)
        return self.records

    def __exit__(self, *a):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.old_level)


def test_data_path_resolution():
    cfg = {"_base_dir": "c:/base"}
    assert data_path(cfg, "corpus_dir", "corpus") == Path("c:/base/corpus")
    assert data_path(cfg, "index_dir", "index") == Path("c:/base/index")
    assert data_path({"corpus_dir": "c:/abs/c"}, "corpus_dir", "corpus") == Path("c:/abs/c")


def test_ragconfig_dict_facade():
    rc = RagConfig(settings={"a": 1}, providers=[])
    assert rc.get("a") == 1 and rc.get("b", 2) == 2
    assert rc["a"] == 1 and "a" in rc and "b" not in rc


def test_resolve_paths_default_lookup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_paths(None) == []
    f = _write(tmp_path / DEFAULT_CONFIG_NAME, {})
    assert resolve_paths(None) == [f.resolve()]
    assert resolve_paths(str(f)) == [f.resolve()]


# --------------------------------------------------------------------------------------
# generic prompts: no deployment-specific vocabulary, concise
# --------------------------------------------------------------------------------------

def test_prompts_are_generic():
    from rag.corpus import prompts
    sp = prompts.system_prompt(1200)
    low = sp.lower()
    for banned in ("unity", "rigidbody", "csharp", "manual", "scripting api",
                   "wheelfrictioncurve", "script alias"):
        assert banned not in low, f"prompt mentions {banned!r}"
    up = prompts.build_user_prompt("docs/x.html", "T", "# md", 10, False)
    assert "unity" not in up.lower()
    rp = prompts.build_repair_prompt(up, ["err"])
    assert "SAME document" in rp
    # the JSON shape contract is unchanged
    assert '"chunks"' in sp and '"heading_path"' in sp and '"qa"' in sp


def test_prompt_version_unchanged_by_rewording():
    """The generic rewording kept v2 semantics: existing corpus stays valid."""
    from rag.corpus.prompts import PROMPT_VERSION
    assert PROMPT_VERSION == "v2"


# --------------------------------------------------------------------------------------
# CLI: --config plumbing (compile requires a provider; default ./config.json)
# --------------------------------------------------------------------------------------

def test_cli_compile_requires_provider(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from rag.__main__ import main
    (tmp_path / "docs").mkdir()
    _write(tmp_path / DEFAULT_CONFIG_NAME, {"dirs": ["docs"]})
    with pytest.raises(SystemExit) as exc:
        main(["compile", "--steps", "corpus"])
    assert "provider" in str(exc.value).lower()


def test_cli_compile_missing_dirs_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from rag.__main__ import main
    _write(tmp_path / DEFAULT_CONFIG_NAME, PROVIDER)
    with pytest.raises(SystemExit) as exc:
        main(["compile", "--steps", "corpus", "--dry-run"])
    assert "dirs" in str(exc.value)


def test_cli_status_uses_default_config(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    from rag.__main__ import main
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.html").write_text("<html><body><p>hi</p></body></html>",
                                 encoding="utf-8")
    _write(tmp_path / DEFAULT_CONFIG_NAME, {"dirs": ["docs"]})
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "1 html pages" in out
