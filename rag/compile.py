"""Stage runner for ``python -m rag compile``: deps -> corpus -> index.

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

from rag.cli.compile_cmd import (
    ALL_DOWN_WAIT_BUDGET,
    CompileReport,
    ProviderShard,
    estimate_tokens,
    finalize,
    gen_key_for_model,
    valid_gen_keys,
    plan_work,
    print_report,
    report_cost,
    run_corpus_compile,
    set_gen_key,
    wipe_generated_dirs,
)
from rag.config import RagConfig, config_dir, data_path, load_config
from rag.corpus.prompts import system_prompt
from rag.llm import ProviderConfig, create_llm
from rag.store import CorpusStore, FileManager

log = logging.getLogger(__name__)

ALL_STEPS = ("deps", "corpus", "index")

# Required for BM25 search + corpus generation: installed by a plain `uv sync`
# (plus --extra dev for the test suite).
CORE_IMPORTS = ("openai", "anthropic", "httpx", "msgspec", "numpy", "bs4",
                "lxml", "json_repair")
# The opt-in `local` extra: multi-GB CUDA torch wheels, so deliberately NOT a
# default dependency. Only the dense index / dense+hybrid search / reranker need
# it; a BM25-only checkout builds and searches fine without it.
LOCAL_IMPORTS = ("torch", "sentence_transformers")
LOCAL_EXTRA_HINT = ("run `uv sync --extra local` (or prefix the command with "
                    "`uv run --extra local`)")


def missing_imports(mods: tuple[str, ...]) -> list[str]:
    import importlib
    missing = []
    for mod in mods:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    return missing


def run_deps(cfg: dict) -> int:
    """Verify imports and (if requested) pre-download the dense embed model.

    Core requirements are fatal when absent; the local-inference stack lives in
    the opt-in ``local`` extra, so its absence is reported as a warning and the
    BM25-only workflow continues (``--steps index`` without ``--skip-dense``
    then fails later with the same hint, because that build does need it).
    """
    missing = missing_imports(CORE_IMPORTS)
    for mod in CORE_IMPORTS:
        if mod in missing:
            print(f"  [deps] MISSING: {mod} — run: uv sync",
                  file=sys.stderr)
        else:
            print(f"  [deps] ok: {mod}")
    if missing:
        return 1

    embed_model = cfg.get("embed_model", "BAAI/bge-m3")
    if embed_model in ("", "none") or cfg.get("skip_dense"):
        print(f"  [deps] dense disabled (embed_model={embed_model!r}); "
              f"nothing to pre-download")
        return 0

    missing_local = missing_imports(LOCAL_IMPORTS)
    if missing_local:
        print(f"  [deps] WARNING: {', '.join(missing_local)} not installed — "
              f"dense/hybrid search and the dense index build need the optional "
              f"local-inference stack; {LOCAL_EXTRA_HINT}. BM25 search and "
              f"corpus generation work without it.", file=sys.stderr)
        return 0

    try:
        from rag.index.vector_index import ensure_embed_model
        ensure_embed_model(embed_model)
        print(f"  [deps] embed model ready: {embed_model}")
    except Exception as exc:
        print(f"  [deps] embed model '{embed_model}' unavailable: {exc}\n"
              f"         dense index will fail; check HF connectivity or set "
              f"embed_model in your config", file=sys.stderr)
        return 1
    return 0


async def run_corpus_step(
    cfg: dict,
    pconfigs: list[ProviderConfig],
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
    base_dir: Path | None = None,
) -> CompileReport:
    root = Path(base_dir) if base_dir is not None else config_dir(cfg)
    corpus_dir = data_path(cfg, "corpus_dir", "corpus", base=root)
    dirs = cfg.get("dirs")
    if not dirs:
        raise SystemExit(
            "config is missing 'dirs' (the input directories to compile), e.g. "
            "{\"dirs\": [\"docs\"]} in your config.json")
    fm = FileManager(root=root, dirs=list(dirs), corpus_dir=corpus_dir)
    store = CorpusStore(corpus_dir)

    if no_thinking:
        # thinking roughly triples per-page latency on reasoning gateways;
        # corpus chunking does not need it. Disabled by stripping the
        # capability so clients send thinking.type=disabled explicitly.
        stripped = []
        for pcfg in pconfigs:
            stripped.append(ProviderConfig(
                model=pcfg.model, type=pcfg.type, base_url=pcfg.base_url,
                api_key=pcfg.api_key, max_tokens=pcfg.max_tokens,
                max_context_size=pcfg.max_context_size, thinking_effort=None,
                capabilities=frozenset(), timeout=pcfg.timeout, env=pcfg.env,
                path=pcfg.path, raw=pcfg.raw))
        pconfigs = stripped
    if no_thinking:
        print("[corpus] thinking disabled (--no-thinking)", file=sys.stderr)

    # Repair phantom manifest entries BEFORE diffing. A run that is killed before
    # finalize leaves entries inherited from the last checkpoint, which may claim
    # pages whose corpus file does not exist (the checkpoint merges the previous
    # manifest, so old phantoms persist). Planning already tolerates them — it
    # checks the filesystem — but they make `status` overstate progress and they
    # never converge on their own. Costs ~1 s (a stat pass over the manifest).
    #
    # Skipped for --dry-run: the audit WRITES the manifest, and a dry run must be
    # read-only so it can be run safely alongside an active build.
    if not dry_run:
        audit = fm.audit_manifest(verbose=False)
        if audit["dropped"]:
            print(f"[corpus] manifest repaired: dropped {audit['dropped']:,} entries "
                  f"claiming pages with no corpus file", file=sys.stderr)

    scanned = fm.scan()
    diff = fm.diff(scanned)

    # provisional shards (clients created lazily after the dry-run path)
    proto_shards = [ProviderShard(None, c.model) for c in pconfigs]  # type: ignore[arg-type]
    valid_keys = valid_gen_keys(proto_shards)
    # Corpus already generated by a model that has since left the fleet (retired
    # provider, exhausted balance) is still valid content — accept its key so
    # those pages are not pointlessly regenerated with the remaining pools.
    # Configured by MODEL NAME in the config: accept_legacy_models.
    for legacy_model in cfg.get("accept_legacy_models") or []:
        valid_keys.add(gen_key_for_model(legacy_model)[0])
        print(f"[corpus] accepting existing corpus from retired model "
              f"{legacy_model!r}", file=sys.stderr)

    work = plan_work(fm, store, diff, valid_gen_keys=valid_keys, force=force,
                     regen=regen, only=only, max_files=max_files)
    prov_desc = ", ".join(f"{c.model} [{c.type}]" for c in pconfigs)
    print(f"[corpus] providers: {prov_desc}", file=sys.stderr)
    print(f"[corpus] scan: {len(scanned)} pages | added={len(diff.added)} "
          f"changed={len(diff.changed)} removed={len(diff.removed)} "
          f"unchanged={len(diff.unchanged)} | to process: {len(work)}",
          file=sys.stderr)

    report = CompileReport(pages_total=len(scanned), pages_planned=len(work),
                           dry_run=dry_run)
    if not work:
        print("[corpus] nothing to do — corpus is up to date", file=sys.stderr)
        if diff.removed:
            # prune only: keep the existing per-page key registry intact
            finalize(fm, diff, scanned,
                     fm.load_manifest().get("page_gen_keys", {}),
                     set_gen_key(proto_shards), dict(proto_shards[0].gen_parts),
                     report)
        return report

    sys_p = system_prompt(cfg.get("max_chunk_chars", 1200))
    if dry_run:
        import random
        import time as _time
        from rag.corpus import extract_page

        t0 = _time.time()
        # Dry-run must stay CHEAP: extracting all 43k pages takes minutes (158 s
        # measured alone, far longer while a build competes for CPU), which
        # defeats the point of a preview. Sample a bounded set and extrapolate by
        # mean page size — the estimator is calibrated on means anyway, so the
        # extrapolation is as accurate as extracting everything, just not exact to
        # the page. Small work lists (< sample cap) are extracted in full.
        cap = 200
        if len(work) <= cap:
            sample = list(work)
            sampled = False
        else:
            sample = random.Random(0).sample(work, cap)
            sampled = True
        pages = []
        for rel in sample:
            p = extract_page(fm.root / rel, root=fm.root,
                             max_chars=cfg.get("max_input_chars", 24_000))
            if p:
                pages.append(p)
        if not pages:
            print("[dry-run] no extractable pages in the sample", file=sys.stderr)
            return report
        est_in, est_out = estimate_tokens(pages, sys_p, thinking=not no_thinking)
        if sampled and pages:
            scale = len(work) / len(pages)
            est_in, est_out = int(est_in * scale), int(est_out * scale)
        report.est_input_tokens, report.est_output_tokens = est_in, est_out
        scope = (f"extrapolated from {len(pages)} sampled pages" if sampled
                 else f"from all {len(pages)} pages")
        print(f"[dry-run] {len(work):,} pages to regenerate ({scope}, "
              f"extraction took {_time.time() - t0:.0f}s)")
        print(report_cost(est_in, est_out, price_in, price_out))
        return report

    shards = []
    for pcfg in pconfigs:
        client = create_llm(pcfg, timeout=cfg.get("llm_timeout", 180))
        shards.append(ProviderShard(client, pcfg.model))
    report = await run_corpus_compile(
        shards, cfg, work=work, scanned=scanned, fm=fm,
        workers_per_provider=max(1, workers // len(shards)),
        wait_budget_s=float(cfg.get("provider_wait_budget_s", ALL_DOWN_WAIT_BUDGET)),
        # The compile unit is ONE PAGE: progress and the incremental manifest
        # checkpoint both advance per finished page. Set "checkpoint_every" /
        # "progress_every" in the config to batch those writes on huge builds.
        checkpoint_every=int(cfg.get("checkpoint_every", 1)),
        progress_every=int(cfg.get("progress_every", 1)),
    )
    # run_corpus_compile already checkpointed the true per-page gen_keys to disk
    # (including pages served by a failover provider). Read them back rather than
    # recomputing from the preferred shard, which would mislabel failover pages.
    manifest = fm.load_manifest()
    page_gen_keys = manifest.get("page_gen_keys", {})
    finalize(fm, diff, scanned, page_gen_keys, set_gen_key(shards),
             dict(shards[0].gen_parts), report)
    print_report(report)
    return report


def run_index_step(cfg: dict, *, force: bool = False,
                   skip_dense: bool = False) -> int:
    from rag.index.build import build_indexes
    return build_indexes(cfg, force=force, skip_dense=skip_dense)


def run_compile(
    *,
    configs: list[str] | None,
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
    skip_dense: bool = False,
    install_embed_model: bool = False,
    clean: bool = False,
) -> int:
    rc: RagConfig = load_config(configs)
    cfg = rc.settings
    if not steps:
        steps = list(ALL_STEPS)
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        raise SystemExit(f"unknown --steps {unknown}; expected any of {ALL_STEPS}")
    if clean:
        # Guards run BEFORE anything is deleted. --clean is the "rm -rf corpus
        # index, then rebuild" operation: unlike --force (which requeues every
        # page but keeps the old corpus files until they are overwritten), the
        # generated artefacts are gone before the build starts, so nothing stale
        # can survive. The wipe covers the indexes too: a fresh corpus under the
        # same gen_key would otherwise hit the index step's "up to date" shortcut
        # and silently keep serving the old chunks.
        if dry_run:
            raise SystemExit("--clean cannot be combined with --dry-run: "
                             "a dry run must stay read-only")
        if only:
            raise SystemExit("--clean cannot be combined with --only")
        if max_files is not None:
            raise SystemExit("--clean cannot be combined with --max-files: "
                             "a clean build must regenerate every page")
        if "corpus" not in steps:
            raise SystemExit("--clean needs --steps to include 'corpus' — "
                             "otherwise nothing regenerates what it deletes")
        if not rc.providers:
            raise SystemExit(
                "the corpus step needs an LLM provider: add model/type/api_key to "
                "your config (see config.example.json) and pass it with --config "
                "(default: ./config.json)")
        wiped = wipe_generated_dirs(cfg, base_dir=rc.base_dir)
        for d in wiped:
            print(f"[clean] deleted {d}", file=sys.stderr)
        if not wiped:
            print("[clean] nothing to delete (no generated artefacts yet)",
                  file=sys.stderr)
    if install_embed_model:
        # standalone model download; needs no provider and does not touch corpus
        from rag.index.vector_index import ensure_embed_model
        model = cfg.get("embed_model", "BAAI/bge-m3")
        print(f"[deps] installing embed model {model} ...", file=sys.stderr)
        try:
            m = ensure_embed_model(model)
        except ImportError as exc:
            print(f"[deps] {exc}", file=sys.stderr)
            return 1
        dim = int(getattr(m, "get_embedding_dimension", None)()
                  if hasattr(m, "get_embedding_dimension")
                  else m.get_sentence_embedding_dimension())
        print(f"[deps] embed model ready: {model} (dim={dim})")
        return 0
    workers = workers or cfg.get("compile_workers", 8)

    if "deps" in steps:
        print("[compile] step: deps", file=sys.stderr)
        if run_deps(cfg):
            return 1
    corpus_report = None
    if "corpus" in steps:
        if not rc.providers:
            raise SystemExit(
                "the corpus step needs an LLM provider: add model/type/api_key to "
                "your config (see config.example.json) and pass it with --config "
                "(default: ./config.json)")
        print("[compile] step: corpus", file=sys.stderr)
        corpus_report = asyncio.run(run_corpus_step(
            cfg, rc.providers, workers=workers, max_files=max_files, force=force,
            regen=regen, only=only, dry_run=dry_run, no_thinking=no_thinking,
            price_in=price_in, price_out=price_out, base_dir=rc.base_dir))
        if corpus_report.aborted_all_providers_down:
            # Non-zero exit: the run stopped because every provider's quota was
            # exhausted, so pages remain ungenerated. Exiting 0 here would make an
            # auto-resume wrapper print COMPILE COMPLETE and never retry.
            return 3
    if "index" in steps and not dry_run:
        print("[compile] step: index", file=sys.stderr)
        return run_index_step(cfg, force=force, skip_dense=skip_dense)
    return 0
