"""The ``status`` command: corpus/index freshness + counts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rag import resolve_path
from rag.compile import load_rag_config


def run_audit_corpus(*, config_path: str = "rag_config.json") -> int:
    """Repair manifest entries that claim pages with no corpus file.

    Safe only when no compile is running (it rewrites corpus/manifest.json, which
    a live build also owns).
    """
    from rag.store import FileManager

    cfg = load_rag_config(config_path)
    fm = FileManager(root=resolve_path("."), dirs=cfg.get("dirs", []),
                     corpus_dir=resolve_path(cfg.get("corpus_dir", "corpus")))
    running = _compile_running()
    if running:
        print("REFUSING: a `rag.py compile` process appears to be running and owns "
              "corpus/manifest.json. Wait for it to finish (or stop it) and rerun.",
              file=sys.stderr)
        return 1
    before = len(fm.load_manifest().get("files", {}))
    res = fm.audit_manifest()
    print(f"[audit] manifest files {before} -> {res['files']} "
          f"(dropped {res['dropped']}); gen_keys {res['gen_keys']}")
    print("[audit] dropped pages now show as 'added' on the next compile, which is "
          "correct — they have no corpus file.")
    return 0


def _compile_running() -> bool:
    """Best-effort check for a live compile process (Windows + POSIX)."""
    import os
    import subprocess

    me = os.getpid()
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                 "Where-Object { $_.CommandLine -like '*steps corpus*' } | "
                 "ForEach-Object { $_.ProcessId }"],
                capture_output=True, text=True, timeout=60).stdout
            pids = {int(x) for x in out.split() if x.strip().isdigit()}
            return bool(pids - {me})
        out = subprocess.run(["pgrep", "-f", "rag.py compile"],
                             capture_output=True, text=True, timeout=60).stdout
        pids = {int(x) for x in out.split() if x.strip().isdigit()}
        return bool(pids - {me, os.getppid()})
    except Exception:
        return False  # cannot tell -> allow (caller was warned in the docstring)


def estimate_pending(pending: int, sampled_rels: list[str], cfg: dict) -> str:
    """Token/cost estimate for *pending* pages, from a small extraction sample.

    Extracting every pending page to size the estimate costs minutes (measured
    158 s for the full 44k mirror), so sample a few and extrapolate by mean
    markdown size. The estimator constants are the ones --dry-run uses, which are
    calibrated against real gateway usage (see compile_cmd), so this agrees with
    a dry run without paying for the full scan.
    """
    import random

    from rag.cli.compile_cmd import estimate_tokens
    from rag.corpus import extract_page
    from rag.corpus.prompts import system_prompt

    if not pending:
        return "est. cost  : nothing pending"
    if not sampled_rels:
        return "est. cost  : no pages available to sample"

    rng = random.Random(0)
    sample = rng.sample(sampled_rels, min(40, len(sampled_rels)))
    root = resolve_path(".")
    pages = []
    for rel in sample:
        p = extract_page(root / rel, root=root,
                         max_chars=cfg.get("max_input_chars", 24_000))
        if p:
            pages.append(p)
    if not pages:
        return "est. cost  : sample extraction failed"

    sys_p = system_prompt(cfg.get("max_chunk_chars", 1200))
    s_in, s_out = estimate_tokens(pages, sys_p, thinking=False)
    scale = pending / len(pages)
    est_in, est_out = int(s_in * scale), int(s_out * scale)
    mean_chars = sum(p.char_len for p in pages) // len(pages)
    lines = [f"est. cost  : {pending:,} pages pending -> "
             f"~{est_in/1e6:.1f}M input / ~{est_out/1e6:.1f}M output tokens",
             f"             (extrapolated from {len(pages)} sampled pages, mean "
             f"{mean_chars:,} chars; pass --price-in/--price-out to compile for "
             f"a priced estimate)"]
    return "\n".join(lines)


def run_status(*, config_path: str = "rag_config.json") -> int:
    cfg = load_rag_config(config_path)
    corpus_dir = resolve_path(cfg.get("corpus_dir", "corpus"))
    index_dir = resolve_path(cfg.get("index_dir", "index"))

    from rag.store import CorpusStore, FileManager
    fm = FileManager(root=resolve_path("."), dirs=cfg.get("dirs", []),
                     corpus_dir=corpus_dir)
    scanned = fm.scan()
    diff = fm.diff(scanned)
    store = CorpusStore(corpus_dir)
    n_corpus_files = sum(1 for _ in store.iterate_all())
    manifest = fm.load_manifest()

    print("== rag status ==")
    print(f"doc mirror : {len(scanned)} html pages")
    print(f"corpus     : {n_corpus_files} .rag.json files under {corpus_dir}")
    print(f"manifest   : gen_key={manifest.get('gen_key', '-')!r} "
          f"({len(manifest.get('files', {}))} recorded)")
    n_claimed = len(manifest.get("files", {}))
    if n_claimed > n_corpus_files:
        # Phantom entries: the manifest claims pages whose corpus file does not
        # exist (scar of an earlier bug where a run claimed the whole scan). They
        # do NOT break planning — plan_work verifies the filesystem, so those
        # pages still requeue — but they overstate progress here. Repair with
        # `rag.py audit-corpus` when no compile is running.
        print(f"             WARNING: manifest claims {n_claimed} pages but only "
              f"{n_corpus_files} corpus files exist ({n_claimed - n_corpus_files} "
              f"phantom). Coverage/diff below use the FILESYSTEM, so planning is "
              f"still correct. Repair when idle: rag.py audit-corpus")
    print(f"coverage   : {n_corpus_files}/{len(scanned)} pages "
          f"({(n_corpus_files / len(scanned) if scanned else 0):.1%}) have a corpus file")
    n_flagged = len(manifest.get("needs_regen", []))
    if n_flagged:
        print(f"needs_regen: {n_flagged} pages hold heuristic-fallback chunks and "
              f"will be retried automatically")
    print(f"diff       : added={len(diff.added)} changed={len(diff.changed)} "
          f"removed={len(diff.removed)} unchanged={len(diff.unchanged)}")
    # Honest pending count: diff trusts the manifest, so phantom entries inflate
    # `unchanged`. Count pages that genuinely still need work, and keep their
    # paths so the cost estimate can sample from the real pending set.
    flagged = set(manifest.get("needs_regen", []))
    pending_rels = [rel for rel in diff.unchanged
                    if store.missing(rel) or rel in flagged]
    pending_rels = sorted(set(pending_rels) | set(diff.added) | set(diff.changed))
    pending = len(pending_rels)
    print(f"pending    : {pending} pages still need generation "
          f"({len(diff.added) + len(diff.changed)} new/changed + "
          f"{pending - len(diff.added) - len(diff.changed)} missing-or-flagged)")
    print(estimate_pending(pending, pending_rels, cfg))
    failures = corpus_dir / "failures.jsonl"
    if failures.exists():
        n_fail = sum(1 for _ in open(failures, encoding="utf-8"))
        print(f"failures   : {n_fail} recorded in corpus/failures.jsonl")

    im_path = index_dir / "manifest.json"
    if im_path.exists():
        im = json.loads(im_path.read_text(encoding="utf-8"))
        fresh = im.get("corpus_gen_key") == manifest.get("gen_key")
        print(f"index      : {im.get('n_chunks')} chunks / {im.get('n_pages')} pages, "
              f"built {im.get('built_at')} ({im.get('wall_s')}s)")
        print(f"             corpus gen_key match: {'YES' if fresh else 'STALE — rerun compile --steps index'}")
        print(f"             embed model: {im.get('embed_model')}")
        for f in ("chunks.msgpack", "bm25_word.pkl", "vectors.f32", "vector_meta.json"):
            p = index_dir / f
            if p.exists():
                print(f"             {f}: {p.stat().st_size / 1e6:.1f} MB")
    else:
        print("index      : NOT BUILT — run: uv run python rag.py compile --steps index")

    # Which engine a query would actually use, and why. The RAG index only
    # replaces the legacy full-corpus index once it covers the mirror.
    from rag.search.engine_select import choose_engine
    engine, st = choose_engine(cfg)
    if st.get("nothing_built"):
        print("engine     : NEITHER engine is built in this checkout")
        print(f"             RAG index: {st['reason']}")
        print(f"             legacy:    missing {', '.join(st['legacy']['missing'])}")
        print("             build with: uv run python rag.py compile "
              "--provider <cfg> --no-thinking  (then --steps index)")
    else:
        print(f"engine     : {engine.upper()} would serve `hybrid_retrieve.py --query ...`")
        print(f"             {st['reason']}")
        if engine != "rag":
            print("             (hybrid_retrieve.py --rag forces the RAG engine; "
                  "--legacy forces the old one)")
    return 0
