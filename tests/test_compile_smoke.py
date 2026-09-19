"""Compile pipeline smoke tests: fake shards + tmp corpus (network-free)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rag.cli.compile_cmd import (
    ProviderPool,
    ProviderShard,
    plan_work,
    run_corpus_compile,
    set_gen_key,
    shard_index,
    valid_gen_keys,
)
from rag.store import CorpusStore, FileManager


class FakeClient:
    def __init__(self, model="fake-model"):
        self._model = model
        self.calls = 0

    @property
    def model_name(self):
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str):
        from rag.llm.base import GenerationResult
        self.calls += 1
        # verbatim chunk: the last line of the prompt is page markdown
        text = user_prompt.rstrip().splitlines()[-1].strip()[:120]
        return GenerationResult(
            text=json.dumps({"chunks": [{
                "heading_path": ["T"], "text": text,
                "summary": "s", "keywords": ["k"], "synonyms": [],
                "qa": [],
            }]}),
            input_tokens=50, output_tokens=25)


def make_shard(model="fake-model"):
    return ProviderShard(FakeClient(model), model)


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
    root = make_site(tmp_path / "site", 40)
    corpus_dir = root / "corpus"   # matches cfg + the monkeypatched resolve_path
    fm = FileManager(root=root, dirs=["ScriptReference"], corpus_dir=corpus_dir)
    cfg = {"max_input_chars": 24000, "max_chunk_chars": 1200}
    return root, corpus_dir, fm, cfg


async def compile_all(fm, cfg, n_expected, shards=None):
    shards = shards or [make_shard()]
    scanned = fm.scan()
    diff = fm.diff(scanned)
    keys = valid_gen_keys(shards)
    work = plan_work(fm, CorpusStore(fm.corpus_dir), diff, valid_gen_keys=keys)
    assert len(work) == n_expected
    report = await run_corpus_compile(
        shards, cfg, work=work, scanned=scanned, fm=fm,
        workers_per_provider=4, progress=False)
    from rag.cli.compile_cmd import finalize
    # finalize reads the checkpointed per-page keys from disk (true provider used)
    page_keys = fm.load_manifest().get("page_gen_keys", {})
    finalize(fm, diff, scanned, page_keys, set_gen_key(shards),
             dict(shards[0].gen_parts), report)
    return report, shards


async def test_compile_40_pages_then_noop(env):
    root, corpus_dir, fm, cfg = env
    report, shards = await compile_all(fm, cfg, 40)
    assert report.pages_done == 40
    assert report.chunks >= 40
    assert report.fallback == 0
    manifest = fm.load_manifest()
    assert len(manifest["files"]) == 40
    assert len(manifest["page_gen_keys"]) == 40
    assert plan_work(fm, CorpusStore(corpus_dir), fm.diff(fm.scan()),
                     valid_gen_keys=valid_gen_keys(shards)) == []


async def test_touch_3_files_processes_3(env):
    root, corpus_dir, fm, cfg = env
    shards = [make_shard()]
    await compile_all(fm, cfg, 40, shards)
    for i in range(3):
        p = root / "ScriptReference" / f"Page{i}.html"
        p.write_text(p.read_text(encoding="utf-8").replace(
            "documentation body", "updated documentation body"), encoding="utf-8")
    work = plan_work(fm, CorpusStore(corpus_dir), fm.diff(fm.scan()),
                     valid_gen_keys=valid_gen_keys(shards))
    assert len(work) == 3, work


async def test_delete_prunes_corpus(env):
    root, corpus_dir, fm, cfg = env
    await compile_all(fm, cfg, 40)
    (root / "ScriptReference" / "Page7.html").unlink()
    diff = fm.diff(fm.scan())
    assert diff.removed == ["ScriptReference/Page7.html"]
    fm.prune(diff.removed)
    store = CorpusStore(corpus_dir)
    assert store.missing_or_corrupt("ScriptReference/Page7.html")
    assert not store.missing_or_corrupt("ScriptReference/Page8.html")


async def test_gen_key_bump_forces_regen(env):
    root, corpus_dir, fm, cfg = env
    shards = [make_shard("model-a")]
    await compile_all(fm, cfg, 40, shards)
    fm2 = FileManager(root=root, dirs=["ScriptReference"], corpus_dir=corpus_dir)
    bumped = [make_shard("model-b")]  # different model -> different gen_key
    work = plan_work(fm2, CorpusStore(corpus_dir), fm2.diff(fm2.scan()),
                     valid_gen_keys=valid_gen_keys(bumped))
    assert len(work) == 40


async def test_two_provider_sharding(env):
    root, corpus_dir, fm, cfg = env
    shards = [make_shard("model-a"), make_shard("model-b")]
    report, _ = await compile_all(fm, cfg, 40, shards)
    assert report.pages_done == 40
    assert shards[0].client.calls > 0 and shards[1].client.calls > 0
    manifest = fm.load_manifest()
    keys = manifest["page_gen_keys"]
    n_a = sum(1 for v in keys.values() if v == shards[0].gen_key)
    n_b = sum(1 for v in keys.values() if v == shards[1].gen_key)
    assert n_a + n_b == 40 and 10 < n_a < 30, (n_a, n_b)
    # rerunning with the same shards is a no-op even with a different shard
    # ORDER: validity is set membership (failover-safe), not exact-shard match.
    swapped = [make_shard("model-b"), make_shard("model-a")]
    work = plan_work(fm, CorpusStore(corpus_dir), fm.diff(fm.scan()),
                     valid_gen_keys=valid_gen_keys(swapped))
    assert work == [], "shard order must not requeue pages"
    # a fleet missing a model DOES requeue that model's pages
    only_a = [make_shard("model-a")]
    work = plan_work(fm, CorpusStore(corpus_dir), fm.diff(fm.scan()),
                     valid_gen_keys=valid_gen_keys(only_a))
    n_b = sum(1 for v in keys.values() if v == shards[1].gen_key)
    assert len(work) == n_b, (len(work), n_b)


async def test_only_and_max_files(env):
    root, corpus_dir, fm, cfg = env
    diff = fm.diff(fm.scan())
    store = CorpusStore(corpus_dir)
    work = plan_work(fm, store, diff, max_files=5)
    assert len(work) == 5
    work = plan_work(fm, store, diff, only="ScriptReference/Page11.html")
    assert work == ["ScriptReference/Page11.html"]


async def test_interrupted_run_resumes(env, monkeypatch):
    root, corpus_dir, fm, cfg = env
    shards = [make_shard()]
    scanned = fm.scan()
    diff = fm.diff(scanned)
    keys = valid_gen_keys(shards)
    store = CorpusStore(corpus_dir)
    work = plan_work(fm, store, diff, valid_gen_keys=keys)

    class Boom(Exception):
        pass

    async def flaky(shards_, cfg_, **kw):
        sem = asyncio.Semaphore(4)
        completed: list[str] = []
        killed = {"v": False}

        async def guarded(rel):
            async with sem:
                if killed["v"]:
                    raise Boom("simulated kill")
                page_corpus = PageCorpus(source=rel, title="t",
                                         html_md5=scanned[rel],
                                         gen_key=shards_[0].gen_key,
                                         generated_at=now_iso(), chunks=[])
                CorpusStore(fm.corpus_dir).save(rel, page_corpus.to_json_dict())
                completed.append(rel)
                if len(completed) == 10:
                    # honest checkpoint: only the 10 processed pages. The old
                    # all-or-nothing bug recorded everything and lost pages.
                    fm.save_manifest(
                        {r: scanned[r] for r in completed}, set_gen_key(shards_),
                        dict(shards_[0].gen_parts),
                        page_gen_keys={r: shards_[0].gen_key for r in completed})
                if len(completed) > 10:
                    killed["v"] = True
                    raise Boom("simulated kill")

        await asyncio.gather(*(guarded(r) for r in kw["work"]))
        raise Boom()

    monkeypatch.setattr("rag.cli.compile_cmd.run_corpus_compile", flaky)
    import rag.cli.compile_cmd as compile_cmd
    with pytest.raises(Boom):
        await compile_cmd.run_corpus_compile(
            shards, cfg, work=work, scanned=scanned, fm=fm,
            workers_per_provider=4, progress=False)
    monkeypatch.undo()

    remaining = plan_work(fm, store, fm.diff(fm.scan()),
                          valid_gen_keys=keys)
    assert len(remaining) == 30, len(remaining)
    report = await run_corpus_compile(
        shards, cfg, work=remaining, scanned=scanned, fm=fm,
        workers_per_provider=4, progress=False)
    from rag.cli.compile_cmd import finalize
    page_keys = fm.load_manifest().get("page_gen_keys", {})
    finalize(fm, diff, scanned, page_keys, set_gen_key(shards),
             dict(shards[0].gen_parts), report)
    assert report.pages_done == 30
    assert len(list(store.iterate_all())) == 40
    assert plan_work(fm, store, fm.diff(fm.scan()),
                     valid_gen_keys=keys) == []


def test_plan_work_rejects_missing_only(env):
    root, corpus_dir, fm, cfg = env
    with pytest.raises(SystemExit):
        plan_work(fm, CorpusStore(corpus_dir), fm.diff(fm.scan()),
                  only="Nope/Missing.html")


def test_shard_index_deterministic():
    assert shard_index("a.html", 2) in (0, 1)
    assert shard_index("a.html", 2) == shard_index("a.html", 2)
    counts = {shard_index(f"p{i}.html", 4) for i in range(200)}
    assert counts == {0, 1, 2, 3}


from rag.corpus.schema import PageCorpus, now_iso  # noqa: E402  (used by flaky)


# --------------------------------------------------------------------------------------
# cost estimator calibration (SA-4: dry-run estimate within 20% of actual)
# --------------------------------------------------------------------------------------


def test_estimate_tokens_matches_measured_gateway_usage():
    """The --dry-run estimate must land within 20% of real gateway usage.

    Calibrated against a measured 24-page run: actual input 9,569 / output 19,482
    tokens for 29,373 chars of page markdown. The pre-calibration estimator
    over-shot by +217% because it billed the system prompt per page, when these
    gateways prompt-cache the identical system prompt across requests.
    """
    from rag.cli.compile_cmd import estimate_tokens

    class FakePage:
        def __init__(self, n):
            self.markdown = "x" * n
            self.char_len = n

    # 24 pages totalling the same markdown volume as the measured run
    pages = [FakePage(29_373 // 24)] * 24
    sys_prompt = "S" * 1862  # the real system prompt is ~1862 chars

    est_in, est_out = estimate_tokens(pages, sys_prompt, thinking=False)
    act_in, act_out = 9_569, 19_482
    assert abs(est_in - act_in) / act_in <= 0.20, \
        f"input estimate {est_in} vs actual {act_in} exceeds 20%"
    assert abs(est_out - act_out) / act_out <= 0.20, \
        f"output estimate {est_out} vs actual {act_out} exceeds 20%"


def test_estimate_excludes_cached_system_prompt():
    """The system prompt must not be billed per page (it is prompt-cached)."""
    from rag.cli.compile_cmd import estimate_tokens

    class FakePage:
        markdown = "x" * 3000
        char_len = 3000

    pages = [FakePage()]
    sys_prompt = "S" * 10_000  # huge: if billed, the estimate explodes

    cached_in, _ = estimate_tokens(pages, sys_prompt, thinking=False)
    billed_in, _ = estimate_tokens(pages, sys_prompt, thinking=False,
                                   system_prompt_billed=True)
    assert cached_in == pytest.approx(3000 / 3.07 + 16, rel=0.01)
    assert billed_in > cached_in * 3, "system prompt should dominate when billed"


def test_estimate_uses_capped_markdown_not_pre_cap_length():
    """Long pages are truncated before sending; estimate the SENT size."""
    from rag.cli.compile_cmd import estimate_tokens

    class FakePage:
        def __init__(self, sent, declared):
            self.markdown = "x" * sent
            self.char_len = declared

    page = FakePage(sent=24_000, declared=500_000)  # capped at 24k
    est_in, _ = estimate_tokens([page], "", thinking=False)
    assert est_in == pytest.approx(24_000 / 3.07 + 16, rel=0.01)
    assert est_in < 24_000 / 3.07 * 2, "must not use the 500k pre-cap length"


def test_estimate_thinking_ratio_is_higher():
    from rag.cli.compile_cmd import estimate_tokens

    class FakePage:
        markdown = "x" * 3000
        char_len = 3000

    _in_plain, out_plain = estimate_tokens([FakePage()], "", thinking=False)
    _in_think, out_think = estimate_tokens([FakePage()], "", thinking=True)
    assert out_think > out_plain, "thinking must estimate more output tokens"


# --------------------------------------------------------------------------------------
# --dry-run: must be cheap (sampled) and must never call the LLM
# --------------------------------------------------------------------------------------


def _write_provider(tmp_path, model="dry-model"):
    """A provider config pointing at a host that cannot resolve, so any accidental
    LLM call would fail loudly rather than silently succeed."""
    p = tmp_path / "prov.json"
    p.write_text(json.dumps({
        "model": model, "type": "openai_legacy", "api_key": "sk-test",
        "url": "http://127.0.0.1:1/v1",  # closed port: connection refused
        "max_tokens": 1000,
    }), encoding="utf-8")
    return p


async def test_dry_run_never_calls_llm(tmp_path, monkeypatch):
    from rag.compile import run_corpus_step

    root = make_site(tmp_path / "site", 30)
    corpus_dir = tmp_path / "corpus"
    prov = _write_provider(tmp_path)
    cfg = {"corpus_dir": str(corpus_dir),
           "dirs": [str(root / "ScriptReference")],
           "max_input_chars": 24000, "max_chunk_chars": 1200}
    # FileManager resolves dirs against root; point rag's resolve_path at tmp root
    import rag.compile as rc
    monkeypatch.setattr(rc, "resolve_path", lambda p=".", *a, **k: (root if str(p) == "." else root / str(p)))

    calls = {"n": 0}

    async def no_llm(*a, **k):
        calls["n"] += 1
        raise AssertionError("dry-run must not call the LLM")

    monkeypatch.setattr("rag.llm.create_llm", lambda *a, **k: no_llm)

    report = await run_corpus_step(cfg, [str(prov)], workers=2, max_files=None,
                                   force=False, regen=False, only=None,
                                   dry_run=True, no_thinking=True,
                                   price_in=None, price_out=None)
    assert report.dry_run is True
    assert calls["n"] == 0
    assert report.pages_planned == 30
    assert report.est_input_tokens > 0 and report.est_output_tokens > 0


async def test_dry_run_samples_large_work_lists(tmp_path, monkeypatch):
    """A large work list must be SAMPLED, not fully extracted (dry-run must be cheap)."""
    from rag.compile import run_corpus_step

    n = 600
    root = make_site(tmp_path / "site", n)
    corpus_dir = tmp_path / "corpus"
    prov = _write_provider(tmp_path)
    cfg = {"corpus_dir": str(corpus_dir),
           "dirs": [str(root / "ScriptReference")],
           "max_input_chars": 24000, "max_chunk_chars": 1200}
    import rag.compile as rc
    monkeypatch.setattr(rc, "resolve_path", lambda p=".", *a, **k: (root if str(p) == "." else root / str(p)))
    monkeypatch.setattr("rag.llm.create_llm",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no LLM in dry-run")))

    extracted = {"n": 0}
    import rag.corpus.extract as ex
    real_extract = ex.extract_page

    def counting_extract(*a, **k):
        extracted["n"] += 1
        return real_extract(*a, **k)

    monkeypatch.setattr(rc, "extract_page", counting_extract, raising=False)
    monkeypatch.setattr("rag.corpus.extract_page", counting_extract)

    report = await run_corpus_step(cfg, [str(prov)], workers=2, max_files=None,
                                   force=False, regen=False, only=None,
                                   dry_run=True, no_thinking=True,
                                   price_in=None, price_out=None)
    assert report.pages_planned == n
    # sampled at the cap, not all n pages
    print(f"OBSERVED extracted={extracted['n']} planned={report.pages_planned}")
    assert extracted["n"] <= 200, f"extracted {extracted['n']} pages; dry-run must sample"
    assert extracted["n"] > 0, "sampling must still extract SOME pages"
    assert report.est_input_tokens > 0


# --------------------------------------------------------------------------------------
# startup manifest repair: phantoms must not survive a killed run
# --------------------------------------------------------------------------------------


async def _run_corpus_step(root, tmp_path, monkeypatch, *, dry_run: bool,
                           client=None):
    """Drive run_corpus_step against a tmp mirror with a fake provider."""
    from rag.compile import run_corpus_step

    prov = tmp_path / "prov.json"
    prov.write_text(json.dumps({
        "model": "fake-model", "type": "openai_legacy", "api_key": "sk-test",
        "url": "http://127.0.0.1:1/v1", "max_tokens": 1000,
    }), encoding="utf-8")

    # corpus_dir is RELATIVE on purpose: the monkeypatched resolve_path rebases
    # non-"." paths under root, so an absolute tmp path would land somewhere else
    # than the helper below writes to.
    cfg = {"corpus_dir": "corpus",
           "dirs": [str(root / "ScriptReference")],
           "max_input_chars": 24000, "max_chunk_chars": 1200}
    import rag.compile as rc
    monkeypatch.setattr(rc, "resolve_path",
                        lambda p=".", *a, **k: (root if str(p) == "." else root / str(p)))
    if client is None:
        client = FakeClient()
    monkeypatch.setattr(rc, "create_llm", lambda *a, **k: client)
    return await run_corpus_step(cfg, [str(prov)], workers=2, max_files=None,
                                 force=False, regen=False, only=None,
                                 dry_run=dry_run, no_thinking=True,
                                 price_in=None, price_out=None)


def _plant_phantoms(root, tmp_path, n_real: int, n_phantom: int):
    """Write a manifest claiming more pages than have corpus files."""
    from rag.store import CorpusStore, FileManager

    corpus_dir = root / "corpus"   # matches cfg + the monkeypatched resolve_path
    fm = FileManager(root=root, dirs=["ScriptReference"], corpus_dir=corpus_dir)
    store = CorpusStore(corpus_dir)
    scanned = fm.scan()
    rels = sorted(scanned)
    for r in rels[:n_real]:
        store.save(r, {"source": r, "chunks": []})
    fake = {r: scanned.get(r, "deadbeef") for r in rels}
    fake.update({f"ScriptReference/Ghost{i}.html": "0" * 32
                 for i in range(n_phantom)})
    fm.save_manifest(fake, "gk", {"model": "fake-model"},
                     page_gen_keys={r: "k" for r in fake}, needs_regen=[])
    return fm, len(fake)


async def test_startup_audit_repairs_phantoms(tmp_path, monkeypatch):
    """A real (non-dry) run repairs phantom entries before diffing, so they
    cannot persist across runs that are killed before finalize."""
    root = make_site(tmp_path / "site", 10)
    fm, claimed = _plant_phantoms(root, tmp_path, n_real=4, n_phantom=6)
    assert len(fm.load_manifest()["files"]) == claimed  # 10 + 6 ghosts

    await _run_corpus_step(root, tmp_path, monkeypatch, dry_run=False)

    after = fm.load_manifest()
    # ghosts dropped; the 10 real pages remain (6 newly generated + 4 pre-existing)
    assert len(after["files"]) == 10, len(after["files"])
    assert all("Ghost" not in r for r in after["files"])


async def test_dry_run_leaves_manifest_untouched(tmp_path, monkeypatch):
    """--dry-run must be read-only so it can run alongside an active build."""
    root = make_site(tmp_path / "site", 10)
    fm, claimed = _plant_phantoms(root, tmp_path, n_real=4, n_phantom=6)
    mtime_before = (root / "corpus" / "manifest.json").stat().st_mtime_ns

    await _run_corpus_step(root, tmp_path, monkeypatch, dry_run=True)

    after = fm.load_manifest()
    assert len(after["files"]) == claimed, "dry-run must not repair the manifest"
    assert (root / "corpus" / "manifest.json").stat().st_mtime_ns == mtime_before
