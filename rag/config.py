"""Unified RAG config: ONE JSON file = provider + corpus settings + engine tuning.

The same file carries three kinds of keys, all optional unless noted:

* **provider keys** — ``model`` (required for the compile corpus step), ``type``,
  ``url``/``base_url``, ``api_key``, ``max_tokens``, ``capabilities``,
  ``thinking_effort``, ``env``, ``timeout`` — parsed into
  :class:`rag.llm.config.ProviderConfig` (see that module for the full list).
* **corpus locations** — ``dirs``: list of input directories to scan (HTML
  documents, relative to the config file; required for the corpus step),
  ``corpus_dir``: generated corpus output (default ``"corpus"``),
  ``index_dir``: generated index output (default ``"index"``).
* **engine tuning** — ``mode``, ``rrf_k``, ``bm25_k``, ``dense_k``, ``final_k``,
  ``path_boost``, ``bm25_aux``, ``embed_model``, ``rerank``, ``compile_workers``,
  ``accept_legacy_models``, ``llm_timeout``, ``provider_wait_budget_s``, ...

Extra provider shards for multi-provider sharding go in a top-level
``"providers"`` list of inline provider objects::

    {
      "model": "qwen3.8-flash", "type": "openai_legacy", "url": "...",
      "api_key": "...", "dirs": ["docs"], "corpus_dir": "corpus",
      "providers": [ {"model": "deepseek-v4-flash", ...} ]
    }

Relative ``corpus_dir`` / ``index_dir`` anchor at the config file's directory,
so a config plus its data directories form a self-contained, relocatable unit.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from rag.llm.config import ProviderConfig, _KNOWN_KEYS

log = logging.getLogger(__name__)

#: default config file name, looked up in the current working directory
DEFAULT_CONFIG_NAME = "config.json"

#: settings key under which the loader records the config's base directory
#: (used to anchor input dirs); private so it never collides with user keys
BASE_DIR_KEY = "_base_dir"


@dataclass
class RagConfig:
    """A loaded config: merged settings + provider shards + path anchoring."""
    settings: dict[str, Any]
    providers: list[ProviderConfig] = field(default_factory=list)
    path: Path | None = None       # the primary config file (first of the list)
    base_dir: Path = field(default_factory=lambda: Path.cwd().resolve())

    # dict-like facade so plain-dict consumers (SearchEngine, build_indexes, ...)
    # keep working unchanged
    def get(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.settings[key]

    def __contains__(self, key: str) -> bool:
        return key in self.settings


def _provider_from(data: dict[str, Any], *, path: Path | None) -> ProviderConfig | None:
    """Build a ProviderConfig from a combined config dict; None when it holds no
    provider (settings-only file). Unknown keys are dropped here — they are RAG
    settings, not provider keys, so no 'ignoring unknown key' warnings."""
    if not data.get("model"):
        return None
    prov_data = {k: v for k, v in data.items() if k in _KNOWN_KEYS}
    cfg = ProviderConfig.from_dict(prov_data)
    cfg.path = path
    return cfg


def _parse_one(path: Path) -> tuple[dict[str, Any], list[ProviderConfig]]:
    """One config file -> (settings dict, provider shards from it)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid config JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a JSON object")
    providers: list[ProviderConfig] = []
    top = _provider_from(data, path=path)
    if top is not None:
        providers.append(top)
    extra = data.get("providers") or []
    if not isinstance(extra, list):
        raise ValueError(f"config {path}: 'providers' must be a list of provider objects")
    for entry in extra:
        if not isinstance(entry, dict):
            raise ValueError(f"config {path}: 'providers' entries must be JSON objects")
        p = _provider_from(entry, path=path)
        if p is None:
            raise ValueError(f"config {path}: 'providers' entry is missing 'model'")
        providers.append(p)
    settings = {k: v for k, v in data.items() if k != "providers"}
    return settings, providers


def _as_paths(paths: str | Path | Iterable[str | Path] | None) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def resolve_paths(paths: str | Path | Iterable[str | Path] | None) -> list[Path]:
    """Resolve the config file list per CLI rules: explicit paths as given
    (CWD-relative or absolute); None -> ``config.json`` in the current working
    directory when it exists, else an empty list."""
    given = _as_paths(paths)
    if given:
        out: list[Path] = []
        for p in given:
            if not p.exists():
                raise FileNotFoundError(f"config not found: {p}")
            out.append(p.resolve())
        return out
    default = Path.cwd() / DEFAULT_CONFIG_NAME
    return [default.resolve()] if default.exists() else []


def load_config(paths: str | Path | Iterable[str | Path] | None = None) -> RagConfig:
    """Load and merge config file(s).

    Merging: settings merge with the FIRST file winning key conflicts (pass the
    primary config first; later files are usually extra provider shards).
    Provider shards concatenate in order. ``corpus_dir`` / ``index_dir`` are
    absolutized against the primary file's directory (falling back to the CWD
    when no config file exists) and the base dir is recorded under
    :data:`BASE_DIR_KEY` so consumers anchor input dirs the same way.
    """
    files = resolve_paths(paths)
    settings: dict[str, Any] = {}
    providers: list[ProviderConfig] = []
    for f in files:
        s, p = _parse_one(f)
        for k, v in s.items():
            settings.setdefault(k, v)   # first file wins
        providers.extend(p)

    base_dir = files[0].parent if files else Path.cwd().resolve()
    for key in ("corpus_dir", "index_dir"):
        v = settings.get(key)
        if isinstance(v, str) and v and not Path(v).is_absolute():
            settings[key] = str((base_dir / v).resolve())
    settings[BASE_DIR_KEY] = str(base_dir)
    return RagConfig(settings=settings, providers=providers,
                     path=files[0] if files else None, base_dir=base_dir)


def load_settings(paths: str | Path | Iterable[str | Path] | None = None) -> dict[str, Any]:
    """Just the merged settings dict (no providers needed — search/status/repl)."""
    return load_config(paths).settings


def config_dir(cfg: dict[str, Any]) -> Path:
    """The base dir recorded by :func:`load_config` (repo-root fallback for
    hand-built dicts that never went through the loader)."""
    from rag import ROOT
    return Path(cfg.get(BASE_DIR_KEY) or ROOT)


def data_path(cfg: dict[str, Any], key: str, default_name: str,
              base: Path | None = None) -> Path:
    """Absolute path for a dir setting (``corpus_dir`` / ``index_dir`` style):
    a set relative value anchors at the config dir (or *base*); unset falls
    back to ``<base>/<default_name>``."""
    b = base or config_dir(cfg)
    v = cfg.get(key)
    p = Path(v) if v else b / default_name
    return p if p.is_absolute() else (b / p).resolve()
