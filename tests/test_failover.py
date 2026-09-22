"""Provider failover + circuit breaker + needs_regen bookkeeping tests (network-free).

These cover the systemic failure found on the real 44k-page build: gateways die
mid-run (kimi 5-hour quota → 403, revoked model access → 403), and without
failover every page assigned to a dead provider silently degraded to aux-less
heuristic chunks while still being marked complete.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.cli.compile_cmd import (
    ProviderPool,
    ProviderShard,
    finalize,
    plan_work,
    run_corpus_compile,
    set_gen_key,
    )
from rag.llm.base import APIStatusError, GenerationResult, RateLimitError
from rag.store import CorpusStore, FileManager
from rag.store.corpus_store import corpus_path


# --------------------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------------------


class FakeClient:
    """Verbatim-chunk client; optionally always raises an API error."""

    def __init__(self, model: str, *, dead: bool = False):
        self._model = model
        self.dead = dead
        self.calls = 0

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str):
        self.calls += 1
        if self.dead:
            raise APIStatusError(
                "Error code: 403 - {'error': {'message': \"You've reached your "
                "5-hour usage limit.\", 'type': 'access_terminated_error'}}",
                status_code=403)
        text = user_prompt.rstrip().splitlines()[-1].strip()[:120]
        return GenerationResult(
            text=json.dumps({"chunks": [{"heading_path": ["T"], "text": text,
                                        "summary": "s", "keywords": ["k"],
                                        "synonyms": [], "qa": []}]}),
            input_tokens=50, output_tokens=25)


def shard(model: str, dead: bool = False) -> ProviderShard:
    return ProviderShard(FakeClient(model, dead=dead), model)


def make_site(tmp_path: Path, n: int) -> Path:
    sr = tmp_path / "ScriptReference"
    sr.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (sr / f"Page{i}.html").write_text(
            f"<html><head><title>Unity - Scripting API: Page{i}</title></head>"
            f"<body><div id='content-wrap'><div class='section'>"
            f"<h1>Page{i}</h1><p>Some documentation body {i}.</p>"
            f"</div></div></body></html>", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def env(tmp_path):
    root = make_site(tmp_path / "site", 24)
    fm = FileManager(root=root, dirs=["ScriptReference"],
                     corpus_dir=tmp_path / "corpus")
    cfg = {"max_input_chars": 24000, "max_chunk_chars": 1200}
    return root, tmp_path / "corpus", fm, cfg


# --------------------------------------------------------------------------------------
# circuit breaker unit behaviour
# --------------------------------------------------------------------------------------


def test_pool_trips_after_threshold():
    shards = [shard("a"), shard("b")]
    pool = ProviderPool(shards, threshold=3, cooldown=60)
    now = 1000.0
    for _ in range(2):
        pool.record_api_failure(0)
    assert pool.is_open(0, now) is False  # below threshold still healthy
    pool.record_api_failure(0)
    assert pool.is_open(0, now) is True
    assert pool.trips == [1, 0]


def test_pool_failover_picks_healthy():
    shards = [shard("a"), shard("b")]
    pool = ProviderPool(shards, threshold=1, cooldown=60)
    now = 1000.0
    pool.record_api_failure(0)
    assert pool.pick(0, now) == 1  # preferred dead -> backup
    assert pool.pick(1, now) == 1  # healthy preferred stays


def test_pool_half_open_probe_after_cooldown():
    shards = [shard("a"), shard("b")]
    pool = ProviderPool(shards, threshold=1, cooldown=60)
    pool.record_api_failure(0, now=1000.0)
    assert pool.is_open(0, 1000.0) is True
    assert pool.is_open(0, 1059.0) is True
    assert pool.is_open(0, 1060.0) is False  # cooldown elapsed -> probe allowed
    assert pool.pick(0, 1060.0) == 0


def test_pool_all_open_returns_none():
    shards = [shard("a"), shard("b")]
    pool = ProviderPool(shards, threshold=1, cooldown=600)
    pool.record_api_failure(0)
    pool.record_api_failure(1)
    assert pool.pick(0, 1000.0) is None
    assert pool.healthy_indices(1000.0) == []


def test_pool_success_closes_circuit():
    shards = [shard("a"), shard("b")]
    pool = ProviderPool(shards, threshold=1, cooldown=60)
    pool.record_api_failure(0)
    assert pool.is_open(0, 1000.0)
    pool.record_success(0)
    assert pool.is_open(0, 1000.0) is False
    assert pool.failures == [0, 0]


def test_pool_served_counts():
    shards = [shard("a"), shard("b")]
    pool = ProviderPool(shards)
    pool.record_success(1)
    pool.record_success(1)
    assert pool.served == [0, 2]
    assert "served=2" in pool.summary()


# --------------------------------------------------------------------------------------
# end-to-end failover through the compile run
# --------------------------------------------------------------------------------------


async def test_dead_provider_fails_over(env):
    """A dead provider's pages are served by the healthy one, with real chunks."""
    root, corpus_dir, fm, cfg = env
    dead, healthy = shard("dead-model", dead=True), shard("live-model")
    shards = [dead, healthy]
    pool_threshold = 2
    scanned = fm.scan()
    work = plan_work(fm, CorpusStore(corpus_dir), fm.diff(scanned))
    assert len(work) == 24

    report = await run_corpus_compile(
        shards, cfg, work=work, scanned=scanned, fm=fm,
        workers_per_provider=2, progress=False,
        circuit_threshold=pool_threshold, circuit_cooldown=600)

    store = CorpusStore(corpus_dir)
    assert report.pages_done == 24
    assert report.fallback == 0, "failover should avoid heuristic fallback entirely"
    assert report.needs_regen == set()
    assert healthy.client.calls > 0
    # every page recorded the HEALTHY provider's key (not the dead one's)
    keys = fm.load_manifest()["page_gen_keys"]
    assert len(keys) == 24
    assert all(v == healthy.gen_key for v in keys.values()), \
        "failover pages must record the provider that actually served them"
    # and the chunks carry real aux fields
    for rel, data in store.iterate_all():
        assert data["chunks"], rel
        assert data["chunks"][0]["summary"] == "s"
        assert data.get("needs_regen", False) is False


async def test_all_providers_dead_aborts_not_degrades(env):
    """When every circuit is open the run STOPS instead of grinding the work list
    into aux-less fallback chunks.

    Rationale: both providers on one gateway hit the same 5-hour quota window and
    die together (this actually happened on the live build). Degrading would
    overwrite every remaining page with heuristic chunks and then exit 0, so an
    auto-resume wrapper would report COMPILE COMPLETE and never retry. Aborting
    leaves the pages untouched for a later run.
    """
    root, corpus_dir, fm, cfg = env
    shards = [shard("dead-a", dead=True), shard("dead-b", dead=True)]
    scanned = fm.scan()
    store = CorpusStore(corpus_dir)
    work = plan_work(fm, store, fm.diff(scanned))

    # cooldown longer than the wait budget so recovery never happens
    report = await run_corpus_compile(
        shards, cfg, work=work, scanned=scanned, fm=fm,
        workers_per_provider=2, progress=False,
        circuit_threshold=1, circuit_cooldown=10_000, wait_budget_s=0.0)

    assert report.aborted_all_providers_down is True
    assert report.pages_done < len(work), (report.pages_done, len(work))
    written = len(list(store.iterate_all()))
    assert written < len(work), "must not degrade every page"
    # anything written during the trip must be self-flagged for retry
    for rel, data in store.iterate_all():
        if not data.get("chunks"):
            continue
        assert data.get("needs_regen") is True
    assert len(fm.load_manifest()["needs_regen"]) == written or \
        report.pages_done == 0


async def test_all_providers_dead_flags_needs_regen(env):
    """Pages that DO get written while providers are down are flagged needs_regen,
    so they stay searchable but retry automatically once quota returns."""
    root, corpus_dir, fm, cfg = env
    shards = [shard("dead-a", dead=True), shard("dead-b", dead=True)]
    scanned = fm.scan()
    work = plan_work(fm, CorpusStore(corpus_dir), fm.diff(scanned))

    # cooldown SHORTER than the wait budget: the pool re-probes, each probe fails,
    # and pages served during a probe window land on the fallback path
    report = await run_corpus_compile(
        shards, cfg, work=work, scanned=scanned, fm=fm,
        workers_per_provider=2, progress=False,
        circuit_threshold=1, circuit_cooldown=0.0, wait_budget_s=0.0)

    store = CorpusStore(corpus_dir)
    written = list(store.iterate_all())
    for rel, data in written:
        assert data["needs_regen"] is True
    assert len(report.needs_regen) == len(written)
    manifest = fm.load_manifest()
    assert set(manifest["needs_regen"]) == {rel for rel, _ in written}


async def test_needs_regen_pages_stay_queued_until_regenerated(env):
    """Flagged pages are backlog diagnostics, not a requeue trigger.

    A later healthy run does NOT pick them up via planning (only an md5 diff or
    a missing corpus file requeues). The supported heal path is deleting their
    corpus files — a filesystem trigger — which requeues exactly those pages.
    """
    root, corpus_dir, fm, cfg = env
    dead = [shard("dead-a", dead=True)]
    scanned = fm.scan()
    store = CorpusStore(corpus_dir)
    # run 1: only a dead provider. cooldown=0 so probes keep happening and pages
    # land on the fallback path (flagged) rather than aborting immediately.
    await run_corpus_compile(dead, cfg,
                             work=plan_work(fm, store, fm.diff(scanned)),
                             scanned=scanned, fm=fm, workers_per_provider=2,
                             progress=False, circuit_threshold=1,
                             circuit_cooldown=0.0, wait_budget_s=0.0)
    flagged1 = fm.load_manifest()["needs_regen"]
    assert flagged1, "pages written while the provider was down must be flagged"
    assert len(flagged1) <= 24
    # run 2 planning: flags requeue NOTHING — the build resumes at the md5 diff
    # (empty here), not at the flagged backlog
    assert plan_work(fm, store, fm.diff(scanned)) == []
    # heal path: delete the flagged corpus files (a filesystem trigger)
    for rel in flagged1:
        corpus_path(corpus_dir, rel).unlink()
    work3 = plan_work(fm, store, fm.diff(scanned))
    assert sorted(work3) == sorted(flagged1), \
        "only the deleted (flagged) corpus files requeue"
    # run 3: healthy provider -> the requeued pages heal and the flag clears
    healthy = [shard("dead-a", dead=True), shard("live")]
    report3 = await run_corpus_compile(
        healthy, cfg, work=work3, scanned=scanned, fm=fm,
        workers_per_provider=2, progress=False, circuit_threshold=1,
        circuit_cooldown=0.0, wait_budget_s=0.0)
    assert report3.fallback == 0
    assert report3.needs_regen == set()
    assert fm.load_manifest()["needs_regen"] == []
    for rel, data in store.iterate_all():
        assert data.get("needs_regen", False) is False
        assert data["chunks"][0]["summary"] == "s"
        assert data.get("needs_regen", False) is False
        assert data["chunks"][0]["summary"] == "s"
        assert data["chunks"][0]["summary"] == "s"


async def test_max_files_does_not_claim_unprocessed(env):
    """finalize must not mark pages without a corpus file as processed."""
    root, corpus_dir, fm, cfg = env
    shards = [shard("a")]
    scanned = fm.scan()
    store = CorpusStore(corpus_dir)
    work = plan_work(fm, store, fm.diff(scanned), max_files=6)
    assert len(work) == 6
    report = await run_corpus_compile(shards, cfg, work=work, scanned=scanned,
                                      fm=fm, workers_per_provider=2,
                                      progress=False)
    diff = fm.diff(scanned)
    finalize(fm, diff, scanned, fm.load_manifest().get("page_gen_keys", {}),
             set_gen_key(shards), dict(shards[0].gen_parts), report)

    manifest = fm.load_manifest()
    assert len(manifest["files"]) == 6, "only processed pages may be claimed"
    assert len(manifest["page_gen_keys"]) == 6
    # the other 18 are still pending
    remaining = plan_work(fm, store, fm.diff(fm.scan()))
    assert len(remaining) == 18


def test_pool_all_open_and_probe_timing():
    """The wait loop relies on all_open()/next_probe_in() to decide when to nap."""
    shards = [shard("a"), shard("b")]
    pool = ProviderPool(shards, threshold=1, cooldown=60)
    pool.record_api_failure(0, now=1000.0)
    pool.record_api_failure(1, now=1000.0)
    assert pool.all_open(1000.0) is True
    assert pool.next_probe_in(1000.0) == pytest.approx(60.0)
    assert pool.next_probe_in(1030.0) == pytest.approx(30.0)
    assert pool.all_open(1060.0) is False
    assert pool.next_probe_in(1060.0) == 0.0
    # one provider healthy -> never "all open", no wait
    pool2 = ProviderPool(shards, threshold=1, cooldown=60)
    pool2.record_api_failure(0, now=1000.0)
    assert pool2.all_open(1000.0) is False
    assert pool2.next_probe_in(1000.0) == pytest.approx(60.0)


def test_plan_work_ignores_recorded_gen_keys(env):
    """Page gen_keys are diagnostics: planning never consults them.

    A page recorded under another model's key (as failover would), an old
    scheme's key, or outright garbage stays done as long as its md5 is
    unchanged and its corpus file exists.
    """
    root, corpus_dir, fm, cfg = env
    a, b = shard("model-a"), shard("model-b")
    store = CorpusStore(corpus_dir)
    scanned = fm.scan()
    for r in scanned:
        store.save(r, {"source": r, "chunks": []})
    fm.save_manifest(scanned, set_gen_key([a, b]), dict(a.gen_parts),
                     page_gen_keys={r: "foreign-or-legacy-key" for r in scanned},
                     needs_regen=[])
    assert plan_work(fm, store, fm.diff(scanned)) == []


def test_plan_work_rejects_missing_only(env):
    """--only must name a real page, not fail obscurely later."""
    root, corpus_dir, fm, cfg = env
    with pytest.raises(SystemExit):
        plan_work(fm, CorpusStore(corpus_dir), fm.diff(fm.scan()),
                  only="Nope/Missing.html")


def test_plan_work_ignores_needs_regen_flag(env):
    """needs_regen flags are backlog diagnostics, NOT a requeue trigger.

    Only an md5 diff or a missing corpus file requeues pages; heuristic-
    fallback pages stay indexed until regenerated via --only/--regen or by
    deleting their corpus file.
    """
    root, corpus_dir, fm, cfg = env
    a = shard("model-a")
    store = CorpusStore(corpus_dir)
    scanned = fm.scan()
    rels = sorted(scanned)
    fm.save_manifest(scanned, set_gen_key([a]), dict(a.gen_parts),
                     page_gen_keys={r: a.gen_key for r in rels},
                     needs_regen=rels[:5])
    for r in rels:
        store.save(r, {"source": r, "chunks": []})
    assert plan_work(fm, store, fm.diff(scanned)) == []


async def test_finalize_is_self_healing_for_phantom_entries(env):
    """finalize must drop manifest entries whose corpus file does not exist.

    A previous broken run claimed the whole scan (43,938 pages) while writing only
    ~18k corpus files. An earlier finalize filtered on `rel in old_files or
    exists(rel)`, which preserved those inherited phantoms forever -- status kept
    reporting 100% claimed while coverage was 42%. Filtering on existence alone
    makes finalize repair the manifest on every run.
    """
    root, corpus_dir, fm, cfg = env
    shards = [shard("fake-model")]
    scanned = fm.scan()
    rels = sorted(scanned)
    store = CorpusStore(corpus_dir)

    # simulate the damage: manifest claims ALL pages, but only 5 corpus files exist
    for r in rels[:5]:
        store.save(r, {"source": r, "chunks": []})
    fm.save_manifest(scanned, set_gen_key(shards), dict(shards[0].gen_parts),
                     page_gen_keys={r: shards[0].gen_key for r in rels},
                     needs_regen=[])
    assert len(fm.load_manifest()["files"]) == 24  # phantoms present
    assert len(fm.load_manifest()["page_gen_keys"]) == 24

    # a run that processes nothing still finalizes
    report = await run_corpus_compile(shards, cfg, work=[], scanned=scanned,
                                      fm=fm, workers_per_provider=2,
                                      progress=False)
    from rag.cli.compile_cmd import finalize
    finalize(fm, fm.diff(scanned), scanned,
             fm.load_manifest().get("page_gen_keys", {}),
             set_gen_key(shards), dict(shards[0].gen_parts), report)

    m = fm.load_manifest()
    assert len(m["files"]) == 5, f"phantom entries survived: {len(m['files'])}"
    assert len(m["page_gen_keys"]) == 5
    # the 19 phantom pages now correctly requeue as work
    work = plan_work(fm, store, fm.diff(fm.scan()))
    assert len(work) == 19, len(work)


async def test_finalize_drops_stale_needs_regen_flags(env):
    """Flags for pages the manifest no longer claims must not accumulate."""
    root, corpus_dir, fm, cfg = env
    shards = [shard("fake-model")]
    scanned = fm.scan()
    rels = sorted(scanned)
    store = CorpusStore(corpus_dir)
    for r in rels[:3]:
        store.save(r, {"source": r, "chunks": []})
    # flag a page that has no corpus file (stale/phantom flag)
    fm.save_manifest({r: scanned[r] for r in rels[:3]}, set_gen_key(shards),
                     dict(shards[0].gen_parts), page_gen_keys={},
                     needs_regen=[rels[0], rels[20]])
    report = await run_corpus_compile(shards, cfg, work=[], scanned=scanned,
                                      fm=fm, workers_per_provider=2,
                                      progress=False)
    from rag.cli.compile_cmd import finalize
    finalize(fm, fm.diff(scanned), {r: scanned[r] for r in rels[:3]},
             {}, set_gen_key(shards), dict(shards[0].gen_parts), report)
    flagged = fm.load_manifest()["needs_regen"]
    assert rels[0] in flagged, "flag for a real page must survive"
    assert rels[20] not in flagged, "flag for a phantom page must be dropped"


async def test_persistent_per_page_failure_still_exits_cleanly(env):
    """A page that always fails validation must NOT wedge the run or the resume loop.

    Termination contract: per-page fallbacks exit 0 (loop terminates, page stays
    flagged for the next run); only all-providers-down aborts exit 3 (where
    retrying genuinely helps, e.g. a quota window reset). A permanently
    unchunkable page must never make the whole build non-terminating.
    """
    root, corpus_dir, fm, cfg = env
    shards = [shard("fake-model")]
    scanned = fm.scan()
    store = CorpusStore(corpus_dir)

    # every page fails validation both attempts -> heuristic fallback, but the API
    # is healthy so the circuit never opens
    class AlwaysInvalid(FakeClient):
        async def generate(self, system_prompt, user_prompt):
            from rag.llm.base import GenerationResult
            self.calls += 1
            return GenerationResult(text="this is not JSON at all",
                                    input_tokens=10, output_tokens=5)

    bad = ProviderShard(AlwaysInvalid("fake-model"), "fake-model")
    report = await run_corpus_compile(
        [bad], cfg, work=sorted(scanned), scanned=scanned, fm=fm,
        workers_per_provider=2, progress=False, wait_budget_s=0.0)

    # run completed rather than aborting
    assert report.aborted_all_providers_down is False
    assert report.pages_done == len(scanned)
    assert report.fallback == len(scanned)
    # every page is flagged for retry but still has a usable corpus file
    assert len(report.needs_regen) == len(scanned)
    assert len(list(store.iterate_all())) == len(scanned)
    assert report.input_tokens > 0  # the LLM really was called (and really failed)


async def test_api_death_sets_abort_flag_but_validation_failure_does_not(env):
    """Distinguish 'provider is down' (abort, exit 3) from 'model output is bad'
    (fallback, exit 0) — they need opposite handling from the resume loop."""
    root, corpus_dir, fm, cfg = env
    scanned = fm.scan()
    store = CorpusStore(corpus_dir)

    dead = [shard("dead", dead=True)]
    r_dead = await run_corpus_compile(
        dead, cfg, work=sorted(scanned)[:4], scanned=scanned, fm=fm,
        workers_per_provider=1, progress=False,
        circuit_threshold=1, circuit_cooldown=10_000, wait_budget_s=0.0)
    assert r_dead.aborted_all_providers_down is True

    class AlwaysInvalid(FakeClient):
        async def generate(self, system_prompt, user_prompt):
            from rag.llm.base import GenerationResult
            self.calls += 1
            return GenerationResult(text="not json", input_tokens=1, output_tokens=1)

    bad = [ProviderShard(AlwaysInvalid("bad"), "bad")]
    r_bad = await run_corpus_compile(
        bad, cfg, work=sorted(scanned)[4:8], scanned=scanned, fm=fm,
        workers_per_provider=1, progress=False, wait_budget_s=0.0)
    assert r_bad.aborted_all_providers_down is False, \
        "bad model output must not look like provider death"
    assert r_bad.pages_done == 4
