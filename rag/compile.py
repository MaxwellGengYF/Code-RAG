"""Stage runner for ``rag.py compile``: deps -> corpus -> index.

Each stage is idempotent and independently skippable via ``--steps``:
  deps    -- verify pip requirements + install/embed the dense model (BGE-M3)
  corpus  -- LLM corpus generation (md5-incremental; the expensive step)
  index   -- flatten corpus -> BM25 + dense indexes (delegates to rag.index.build)
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from rag import resolve_path
from rag.cli.compile_cmd import (
    CompileReport,
    compute_gen_key,
    estimate_tokens,
    finalize,
    plan_work,
    print_report,
    report_cost,
    run_corpus_compile,
)
from rag.corpus.prompts import system_prompt
from rag.llm import ProviderConfig, create_llm
from rag.store import CorpusStore, FileManager

log = logging.getLogger(__name__)

ALL_STEPS = ("deps", "corpus", "index")


def load_rag_config(path: str = "rag_config.json") -> dict[str, Any]:
    import json
    p = resolve_path(path)
    cfg: dict[str, Any] = {}
    if p.exists():
        cfg = json.loads(p.read_text(encoding="utf-8"))
    return cfg


def run_deps(cfg: dict) -> int:
    """Verify imports and (if requested) pre-download the dense embed model."""
    import importlib
    for mod in ("openai", "anthropic", "httpx", "msgspec", "numpy",
                "sentence_transformers", "bs4", "lxml"):
        try:
            importlib.import_module(mod)
            print(f"  [deps] ok: {mod}")
        except ImportError:
            print(f"  [deps] MISSING: {mod} — run: uv sync --extra dev", file=sys.stderr)
            return 1
    embed_model = cfg.get("embed_model", "BAAI/bge-m3")
    try:
        from rag.index.vector_index import ensure_embed_model
        ensure_embed_model(embed_model)
        print(f"  [deps] embed model ready: {embed_model}")
    except Exception as exc:
        print(f"  [deps] embed model '{embed_model}' unavailable: {exc}\n"
              f"         dense index will fail; check HF connectivity or set "
              f"embed_model in rag_config.json", file=sys.stderr)
        return 1
    return 0


async def run_corpus_step(
    cfg: dict,
    provider: str,
    *,
    workers: int,
    max_files: int | None,
    force: bool,
    regen: bool,
    only: str | None,
    dry_run: bool,
    no_thinking: bool = False,
    price_in: float | None,
    price_out: float | None,
) -> CompileReport:
    root = resolve_path(".")
    corpus_dir = resolve_path(cfg.get("corpus_dir", "corpus"))
    dirs = cfg.get("dirs", ["Manual", "ScriptReference"])
    fm = FileManager(root=root, dirs=dirs, corpus_dir=corpus_dir)
    store = CorpusStore(corpus_dir)

    pcfg = ProviderConfig.from_file(provider)
    if no_thinking:
        # thinking roughly triples per-page latency on reasoning gateways; corpus
        # chunking does not need it. Disabled by stripping the capability so the
        # provider clients send thinking.type=disabled.
        pcfg = ProviderConfig(
            model=pcfg.model, type=pcfg.type, base_url=pcfg.base_url,
            api_key=pcfg.api_key, max_tokens=pcfg.max_tokens,
            max_context_size=pcfg.max_context_size, thinking_effort=None,
            capabilities=frozenset(), timeout=pcfg.timeout, env=pcfg.env,
            path=pcfg.path, raw=pcfg.raw)
        print("[corpus] thinking disabled (--no-thinking)", file=sys.stderr)
    gen_key, gen_parts = compute_gen_key(pcfg.model)
    scanned = fm.scan()
    diff = fm.diff(scanned)
    manifest = fm.load_manifest()
    old_gen_key = fm.current_gen_key() or manifest.get("gen_key", "")
    gen_key_changed = bool(old_gen_key) and old_gen_key != gen_key
    if gen_key_changed:
        print(f"[corpus] gen_key changed ({old_gen_key} -> {gen_key}); "
              f"all pages will be regenerated", file=sys.stderr)

    work = plan_work(fm, store, diff, force=force, regen=regen, only=only,
                     max_files=max_files, gen_key_changed=gen_key_changed)
    print(f"[corpus] scan: {len(scanned)} pages | added={len(diff.added)} "
          f"changed={len(diff.changed)} removed={len(diff.removed)} "
          f"unchanged={len(diff.unchanged)} | to process: {len(work)}", file=sys.stderr)

    report = CompileReport(pages_total=len(scanned), pages_planned=len(work),
                           dry_run=dry_run)
    if not work:
        print("[corpus] nothing to do — corpus is up to date", file=sys.stderr)
        if diff.removed:
            finalize(fm, diff, scanned, gen_key, gen_parts, report)
        return report

    sys_p = system_prompt(cfg.get("max_chunk_chars", 1200))
    if dry_run:
        t0 = __import__("time").time()
        pages = []
        from rag.corpus import extract_page
        for rel in work:
            p = extract_page(fm.root / rel, root=fm.root,
                             max_chars=cfg.get("max_input_chars", 24_000))
            if p:
                pages.append(p)
        est_in, est_out = estimate_tokens(pages, sys_p, thinking=not no_thinking)
        report.est_input_tokens, report.est_output_tokens = est_in, est_out
        print(f"[dry-run] {len(pages)} pages to regenerate "
              f"(extraction took {__import__('time').time() - t0:.0f}s)")
        print(report_cost(est_in, est_out, price_in, price_out))
        return report

    client = create_llm(pcfg)
    report = await run_corpus_compile(
        client, cfg, work=work, scanned=scanned, fm=fm, gen_key=gen_key,
        gen_parts=gen_parts, workers=workers,
    )
    finalize(fm, diff, scanned, gen_key, gen_parts, report)
    print_report(report)
    return report


def run_index_step(cfg: dict) -> int:
    from rag.index.build import build_indexes
    return build_indexes(cfg)


def run_compile(
    *,
    provider: str | None,
    steps: list[str],
    workers: int = 8,
    max_files: int | None = None,
    force: bool = False,
    regen: bool = False,
    only: str | None = None,
    dry_run: bool = False,
    no_thinking: bool = False,
    price_in: float | None = None,
    price_out: float | None = None,
    config_path: str = "rag_config.json",
) -> int:
    cfg = load_rag_config(config_path)
    if not steps:
        steps = list(ALL_STEPS)
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        raise SystemExit(f"unknown --steps {unknown}; expected any of {ALL_STEPS}")
    workers = workers or cfg.get("compile_workers", 8)

    if "deps" in steps:
        print("[compile] step: deps", file=sys.stderr)
        if run_deps(cfg):
            return 1
    if "corpus" in steps:
        if not provider:
            raise SystemExit("--provider is required for the corpus step")
        print("[compile] step: corpus", file=sys.stderr)
        asyncio.run(run_corpus_step(
            cfg, provider, workers=workers, max_files=max_files, force=force,
            regen=regen, only=only, dry_run=dry_run, no_thinking=no_thinking,
            price_in=price_in, price_out=price_out))
    if "index" in steps and not dry_run:
        print("[compile] step: index", file=sys.stderr)
        return run_index_step(cfg)
    return 0
