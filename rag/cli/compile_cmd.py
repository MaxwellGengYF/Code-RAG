"""The ``compile`` command: LLM corpus generation with md5-incremental scheduling.

Flow: scan → diff vs manifest → (per-page gen_key mismatch forces regen) → async
LLM generation with per-provider worker semaphores → per-page corpus files
(atomic) → checkpointed manifest flushes → failures.jsonl → final report.

Never aborts the run on per-page failures: a page that fails every LLM attempt
falls back to heuristic chunking (see rag.corpus.generate) and is logged.

Multi-provider sharding: pages are assigned to providers deterministically
(md5(rel) % n_providers), which doubles throughput when two gateways are
available and keeps reruns stable (a page always maps to the same provider).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from rag.corpus import (
    EXTRACTOR_VERSION,
    SCHEMA_VERSION,
    PageCorpus,
    extract_page,
    generate_page_corpus,
)
from rag.corpus.prompts import PROMPT_VERSION, system_prompt
from rag.llm.base import LLMClient
from rag.store import CorpusStore, FileManager, ManifestDiff

log = logging.getLogger(__name__)

CHECKPOINT_EVERY = 25
#: empirical output/input token ratios (20-page SA-2 samples): thinking burns
#: ~3x the input tokens on reasoning gateways; without it output ~= input
EST_OUTPUT_INPUT_RATIO_THINKING = 3.0
EST_OUTPUT_INPUT_RATIO_NO_THINKING = 1.0
#: chars-per-token for rough estimation
CHARS_PER_TOKEN = 4


@dataclass
class CompileReport:
    pages_total: int = 0       # pages in the scan
    pages_planned: int = 0     # pages selected for regeneration
    pages_done: int = 0
    chunks: int = 0
    first_try: int = 0
    repaired: int = 0
    fallback: int = 0
    pruned: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: list[dict] = field(default_factory=list)
    wall_s: float = 0.0
    dry_run: bool = False
    est_input_tokens: int = 0
    est_output_tokens: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# --------------------------------------------------------------------------------------
# generation keys + provider sharding
# --------------------------------------------------------------------------------------


def gen_key_for_model(model: str) -> tuple[str, dict]:
    parts = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "extractor_version": EXTRACTOR_VERSION,
        "schema_version": SCHEMA_VERSION,
    }
    return FileManager.gen_key(**parts), parts


def shard_index(rel: str, n_shards: int) -> int:
    """Deterministic provider assignment for one page."""
    return int(hashlib.md5(rel.encode("utf-8")).hexdigest(), 16) % n_shards


class ProviderShard:
    """One provider's client + its generation key."""

    def __init__(self, client: LLMClient, model: str):
        self.client = client
        self.model = model
        self.gen_key, self.gen_parts = gen_key_for_model(model)


def make_expected_gen_key(shards: Sequence[ProviderShard]) -> Callable[[str], str]:
    keys = [s.gen_key for s in shards]
    return lambda rel: keys[shard_index(rel, len(keys))]


def set_gen_key(shards: Sequence[ProviderShard]) -> str:
    """Manifest-level marker for a multi-provider build (informational only;
    per-page validity is tracked in ``page_gen_keys``)."""
    return "|".join(sorted({s.gen_key for s in shards}))


# --------------------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------------------


def plan_work(
    fm: FileManager,
    store: CorpusStore,
    diff: ManifestDiff,
    *,
    expected_gen_key: Callable[[str], str] | None = None,
    force: bool = False,
    regen: bool = False,
    only: str | None = None,
    max_files: int | None = None,
) -> list[str]:
    """Pages needing (re)generation, sorted for determinism."""
    if only:
        rel = only.replace("\\", "/").lstrip("./")
        if rel not in fm.scan():
            raise SystemExit(f"--only {only!r}: not an html page under {fm.dirs}")
        return [rel]
    if force or regen:
        return sorted(diff.added + diff.changed + diff.unchanged)[:max_files] if max_files \
            else sorted(diff.added + diff.changed + diff.unchanged)
    work = sorted(diff.added + diff.changed)
    # Requeue: corpus file missing/corrupt, or generated with a different key
    # (prompt/model/extractor/schema bump, or a different provider shard).
    manifest = fm.load_manifest()
    page_gen_keys = manifest.get("page_gen_keys", {})
    for rel in diff.unchanged:
        if store.missing_or_corrupt(rel):
            work.append(rel)
        elif expected_gen_key is not None and page_gen_keys.get(rel) != expected_gen_key(rel):
            work.append(rel)
    return sorted(set(work))[:max_files] if max_files else sorted(set(work))


# --------------------------------------------------------------------------------------
# cost estimation
# --------------------------------------------------------------------------------------


def estimate_tokens(pages: list, sys_prompt: str, *,
                    thinking: bool = True) -> tuple[int, int]:
    """(input, output) token estimates for a list of PageInputs."""
    est_in = 0
    for p in pages:
        est_in += (len(sys_prompt) + p.char_len) // CHARS_PER_TOKEN + 32
    ratio = (EST_OUTPUT_INPUT_RATIO_THINKING if thinking
             else EST_OUTPUT_INPUT_RATIO_NO_THINKING)
    est_out = int(est_in * ratio)
    return est_in, est_out


def report_cost(est_in: int, est_out: int, price_in: float | None,
                price_out: float | None) -> str:
    lines = [f"estimated tokens: input={est_in:,} output={est_out:,}"]
    if price_in and price_out:
        cost = (est_in * price_in + est_out * price_out) / 1e6
        lines.append(f"estimated cost: {cost:.2f} (price_in={price_in}/M, "
                     f"price_out={price_out}/M)")
    else:
        lines.append("cost: pass --price-in/--price-out (per 1M tokens) to monetize")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# the compile run
# --------------------------------------------------------------------------------------


async def run_corpus_compile(
    shards: Sequence[ProviderShard],
    cfg: dict,
    *,
    work: list[str],
    scanned: dict[str, str],
    fm: FileManager,
    workers_per_provider: int = 4,
    failures_path: Path | None = None,
    progress: bool = True,
) -> CompileReport:
    store = CorpusStore(fm.corpus_dir)
    report = CompileReport(pages_total=len(scanned), pages_planned=len(work))
    t0 = time.time()
    processed: dict[str, str] = {}  # rel -> gen_key actually used
    sems = [asyncio.Semaphore(workers_per_provider) for _ in shards]
    done_counter = 0
    lock = asyncio.Lock()
    expected = make_expected_gen_key(shards)

    if failures_path is None:
        failures_path = fm.corpus_dir / "failures.jsonl"

    def write_failure(rec: dict) -> None:
        with open(failures_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    async def one(rel: str) -> None:
        nonlocal done_counter
        shard = shards[shard_index(rel, len(shards))]
        async with sems[shard_index(rel, len(shards))]:
            page = await asyncio.to_thread(
                extract_page, fm.root / rel, root=fm.root,
                max_chars=cfg.get("max_input_chars", 24_000))
            if page is None:
                corpus, stats = _empty_page_corpus(rel, scanned[rel], shard.gen_key)
            else:
                corpus, stats = await generate_page_corpus(
                    shard.client, page, gen_key=shard.gen_key,
                    html_md5=scanned[rel],
                    max_chunk_chars=cfg.get("max_chunk_chars", 1200))
            store.save(rel, corpus.to_json_dict())
            async with lock:
                processed[rel] = shard.gen_key
                report.pages_done += 1
                report.chunks += len(corpus.chunks)
                report.input_tokens += stats.input_tokens
                report.output_tokens += stats.output_tokens
                report.first_try += int(stats.first_try_valid)
                report.repaired += int(stats.repaired and not stats.first_try_valid)
                report.fallback += int(stats.fallback)
                if stats.fallback or stats.errors:
                    report.failures.append({"source": rel, "errors": stats.errors[:6]})
                    await asyncio.to_thread(write_failure,
                                            {"source": rel, "errors": stats.errors})
                done_counter += 1
                if done_counter % CHECKPOINT_EVERY == 0:
                    # Checkpoint processed pages MERGED with the existing
                    # manifest: (a) only actually-processed pages are claimed
                    # (an interrupted run must leave the rest "added"), and
                    # (b) entries from earlier partial runs survive.
                    old = fm.load_manifest()
                    files_ckpt = {**old.get("files", {}),
                                  **{rel: scanned[rel] for rel in processed}}
                    keys_ckpt = {**old.get("page_gen_keys", {}), **processed}
                    fm.save_manifest(files_ckpt, set_gen_key(shards),
                                     dict(shards[0].gen_parts),
                                     page_gen_keys=keys_ckpt)
                if progress and report.pages_done % 50 == 0:
                    print(f"  [{report.pages_done}/{len(work)}] "
                          f"{time.time() - t0:.0f}s", file=sys.stderr)

    await asyncio.gather(*(one(rel) for rel in work))
    report.wall_s = time.time() - t0
    return report


def _empty_page_corpus(rel: str, html_md5: str, gen_key: str) -> tuple[PageCorpus, Any]:
    """Pages with no extractable content still get a (chunk-less) corpus record."""
    from rag.corpus.generate import PageGenStats
    from rag.corpus.schema import now_iso
    stats = PageGenStats(source=rel)
    stats.fallback = True
    stats.errors.append("no extractable content")
    return PageCorpus(source=rel, title="", html_md5=html_md5, gen_key=gen_key,
                      generated_at=now_iso(), chunks=[]), stats


def finalize(fm: FileManager, diff: ManifestDiff, files_state: dict[str, str],
             page_gen_keys: dict[str, str], gen_key: str, gen_parts: dict,
             report: CompileReport) -> None:
    """Prune removed pages and write the final manifest."""
    fm.prune(diff.removed)
    report.pruned = len(diff.removed)
    fm.save_manifest(files_state, gen_key, gen_parts,
                     page_gen_keys=page_gen_keys)


def print_report(report: CompileReport, out=sys.stdout) -> None:
    print(f"\n== compile report ==", file=out)
    print(f"pages total={report.pages_total} planned={report.pages_planned} "
          f"done={report.pages_done}", file=out)
    print(f"chunks written={report.chunks}", file=out)
    print(f"first-try valid={report.first_try} repaired={report.repaired} "
          f"fallback={report.fallback}", file=out)
    print(f"tokens: input={report.input_tokens:,} output={report.output_tokens:,}",
          file=out)
    print(f"pruned corpus files={report.pruned}", file=out)
    print(f"wall time: {report.wall_s:.0f}s", file=out)
    if report.failures:
        print(f"pages with errors/fallback: {len(report.failures)} (see "
              f"corpus/failures.jsonl)", file=out)
