"""SA-2 acceptance re-measurement: first-try valid JSON rate on real pages.

Plan SA-2 acceptance: on sampled pages, >=90% first-try valid JSON and 100% usable
after repair/fallback, avg <=4 chunks/page, and the verbatim substring invariant
must hold for every chunk.

Earlier small samples gave 80% (thinking on) / 95% (thinking off). This measures
at a larger sample with the shipped configuration (--no-thinking equivalent,
PROMPT_VERSION v2) using the same provider fleet as the full build, and also
asserts the verbatim invariant on every chunk rather than trusting validation.

Uses a scratch manifest so it cannot race the running full build.

Usage: uv run python eval_corpus_quality.py [n_pages]
"""
import asyncio
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import ROOT
from rag.cli.compile_cmd import ProviderShard, run_corpus_compile, set_gen_key
from rag.corpus.schema import norm_verbatim
from rag.llm import ProviderConfig, create_llm
from rag.store import CorpusStore, FileManager

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
PROVIDERS = ["D:/ds_ali.json", "D:/qwen_flash.json"]
SCRATCH = "manifest_quality_probe.json"


async def main():
    fm = FileManager(root=ROOT, dirs=["Manual", "ScriptReference"],
                     corpus_dir=ROOT / "corpus", manifest_name=SCRATCH)
    scanned = fm.scan()
    rng = random.Random(2026)
    rels = sorted(scanned)
    sample = rng.sample(rels, N)

    shards = []
    for p in PROVIDERS:
        pcfg = ProviderConfig.from_file(p)
        pcfg = ProviderConfig(model=pcfg.model, type=pcfg.type, base_url=pcfg.base_url,
                              api_key=pcfg.api_key, max_tokens=pcfg.max_tokens,
                              capabilities=frozenset(), thinking_effort=None)
        shards.append(ProviderShard(create_llm(pcfg, timeout=180), pcfg.model))

    t0 = time.time()
    report = await run_corpus_compile(
        shards, {"max_input_chars": 24000, "max_chunk_chars": 1200},
        work=sample, scanned=scanned, fm=fm, workers_per_provider=2,
        progress=False, wait_budget_s=0.0)
    wall = time.time() - t0
    (ROOT / "corpus" / SCRATCH).unlink(missing_ok=True)

    n = report.pages_done
    first = report.first_try
    repaired = report.repaired
    fallback = report.fallback
    usable = first + repaired

    # independent verification of the verbatim invariant on EVERY chunk of the
    # pages just generated (validation asserted it at build time; re-check here)
    store = CorpusStore(ROOT / "corpus")
    chunks_tot = 0
    verbatim_bad = []
    aux_chunks = 0
    for rel in sample:
        data = store.load(rel)
        if not data:
            continue
        page_norm = None
        for ch in data.get("chunks") or []:
            chunks_tot += 1
            if ch.get("summary") or ch.get("keywords"):
                aux_chunks += 1
            if page_norm is None:
                from rag.corpus import extract_page
                page = extract_page(ROOT / rel, root=ROOT)
                page_norm = norm_verbatim(page.markdown) if page else ""
            if page_norm and norm_verbatim(ch["text"]) not in page_norm:
                verbatim_bad.append((rel, ch["text"][:60]))

    print(f"\n== SA-2 acceptance ({n} pages, {wall:.0f}s) ==")
    print(f"  first-try valid : {first}/{n} = {first/max(n,1):.0%}   [target >=90%]")
    print(f"  repaired        : {repaired}/{n}")
    print(f"  usable          : {usable}/{n} = {usable/max(n,1):.0%}   [target 100%]")
    print(f"  fallback        : {fallback}/{n}")
    print(f"  avg chunks/page : {chunks_tot/max(n,1):.2f}   [target <=4]")
    print(f"  chunks with aux : {aux_chunks}/{chunks_tot} ({aux_chunks/max(chunks_tot,1):.0%})")
    print(f"  verbatim violations: {len(verbatim_bad)}   [target 0]")
    for rel, txt in verbatim_bad[:3]:
        print(f"    {rel}: {txt!r}")
    ok = (first / max(n, 1) >= 0.90 and usable == n and
          chunks_tot / max(n, 1) <= 4 and not verbatim_bad)
    print(f"  VERDICT: {'PASS' if ok else 'CHECK ABOVE'}")


asyncio.run(main())
