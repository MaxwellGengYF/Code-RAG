"""Compile pipeline smoke tests: fake shards + tmp corpus (network-free)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rag.cli.compile_cmd import (
    ProviderShard,
    make_expected_gen_key,
    plan_work,
    run_corpus_compile,
    set_gen_key,
    shard_index,
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
    corpus_dir = tmp_path / "corpus"
    fm = FileManager(root=root, dirs=["ScriptReference"], corpus_dir=corpus_dir)
    cfg = {"max_input_chars": 24000, "max_chunk_chars": 1200}
    return root, corpus_dir, fm, cfg


async def compile_all(fm, cfg, n_expected, shards=None):
    shards = shards or [make_shard()]
    scanned = fm.scan()
    diff = fm.diff(scanned)
    expected = make_expected_gen_key(shards)
    work = plan_work(fm, CorpusStore(fm.corpus_dir), diff, expected_gen_key=expected)
    assert len(work) == n_expected
    report = await run_corpus_compile(
        shards, cfg, work=work, scanned=scanned, fm=fm,
        workers_per_provider=4, progress=False)
    page_keys = {rel: expected(rel) for rel in work}
    from rag.cli.compile_cmd import finalize
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
                     expected_gen_key=make_expected_gen_key(shards)) == []


async def test_touch_3_files_processes_3(env):
    root, corpus_dir, fm, cfg = env
    shards = [make_shard()]
    await compile_all(fm, cfg, 40, shards)
    for i in range(3):
        p = root / "ScriptReference" / f"Page{i}.html"
        p.write_text(p.read_text(encoding="utf-8").replace(
            "documentation body", "updated documentation body"), encoding="utf-8")
    work = plan_work(fm, CorpusStore(corpus_dir), fm.diff(fm.scan()),
                     expected_gen_key=make_expected_gen_key(shards))
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
                     expected_gen_key=make_expected_gen_key(bumped))
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
    # rerunning with the same shards is a no-op; swapping shard order rekeys all
    swapped = [make_shard("model-b"), make_shard("model-a")]
    work = plan_work(fm, CorpusStore(corpus_dir), fm.diff(fm.scan()),
                     expected_gen_key=make_expected_gen_key(swapped))
    assert len(work) == 40, "shard order must not change page->provider mapping"


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
    expected = make_expected_gen_key(shards)
    store = CorpusStore(corpus_dir)
    work = plan_work(fm, store, diff, expected_gen_key=expected)

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
                        page_gen_keys={r: expected(r) for r in completed})
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
                          expected_gen_key=expected)
    assert len(remaining) == 30, len(remaining)
    report = await run_corpus_compile(
        shards, cfg, work=remaining, scanned=scanned, fm=fm,
        workers_per_provider=4, progress=False)
    from rag.cli.compile_cmd import finalize
    page_keys = {rel: expected(rel) for rel in remaining}
    old_keys = fm.load_manifest().get("page_gen_keys", {})
    finalize(fm, diff, scanned, {**old_keys, **page_keys}, set_gen_key(shards),
             dict(shards[0].gen_parts), report)
    assert report.pages_done == 30
    assert len(list(store.iterate_all())) == 40
    assert plan_work(fm, store, fm.diff(fm.scan()),
                     expected_gen_key=expected) == []


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
