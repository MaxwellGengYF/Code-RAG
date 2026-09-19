"""Compile pipeline smoke tests: fake client + tmp corpus (network-free)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rag.cli.compile_cmd import (
    compute_gen_key,
    plan_work,
    run_corpus_compile,
)
from rag.corpus.generate import PageGenStats
from rag.corpus.schema import PageCorpus, now_iso
from rag.store import CorpusStore, FileManager


class FakeClient:
    def __init__(self):
        self.calls = 0

    @property
    def model_name(self):
        return "fake-model"

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


async def compile_all(fm, cfg, n_expected):
    scanned = fm.scan()
    diff = fm.diff(scanned)
    gen_key, gen_parts = compute_gen_key("fake-model")
    work = plan_work(fm, CorpusStore(fm.corpus_dir), diff)
    assert len(work) == n_expected
    report = await run_corpus_compile(
        FakeClient(), cfg, work=work, scanned=scanned, fm=fm,
        gen_key=gen_key, gen_parts=gen_parts, workers=4, progress=False)
    from rag.cli.compile_cmd import finalize
    finalize(fm, diff, scanned, gen_key, gen_parts, report)
    return report, gen_key


async def test_compile_40_pages_then_noop(env):
    root, corpus_dir, fm, cfg = env
    report, gen_key = await compile_all(fm, cfg, 40)
    assert report.pages_done == 40
    assert report.chunks >= 40
    assert report.fallback == 0
    # manifest records everything
    manifest = fm.load_manifest()
    assert len(manifest["files"]) == 40
    assert manifest["gen_key"] == gen_key
    # immediate rerun: nothing to do
    scanned = fm.scan()
    diff = fm.diff(scanned)
    work = plan_work(fm, CorpusStore(corpus_dir), diff)
    assert work == []


async def test_touch_3_files_processes_3(env):
    root, corpus_dir, fm, cfg = env
    await compile_all(fm, cfg, 40)
    for i in range(3):
        p = root / "ScriptReference" / f"Page{i}.html"
        p.write_text(p.read_text(encoding="utf-8").replace(
            "documentation body", "updated documentation body"), encoding="utf-8")
    diff = fm.diff(fm.scan())
    work = plan_work(fm, CorpusStore(corpus_dir), diff)
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
    await compile_all(fm, cfg, 40)
    fm2 = FileManager(root=root, dirs=["ScriptReference"], corpus_dir=corpus_dir)
    diff = fm2.diff(fm2.scan())
    work = plan_work(fm2, CorpusStore(corpus_dir), diff, gen_key_changed=True)
    assert len(work) == 40


async def test_only_and_max_files(env):
    root, corpus_dir, fm, cfg = env
    diff = fm.diff(fm.scan())
    work = plan_work(fm, CorpusStore(corpus_dir), diff, max_files=5)
    assert len(work) == 5
    work = plan_work(fm, CorpusStore(corpus_dir), diff,
                     only="ScriptReference/Page11.html")
    assert work == ["ScriptReference/Page11.html"]


async def test_interrupted_run_resumes(env, monkeypatch):
    root, corpus_dir, fm, cfg = env
    scanned = fm.scan()
    diff = fm.diff(scanned)
    gen_key, gen_parts = compute_gen_key("fake-model")
    work = plan_work(fm, CorpusStore(corpus_dir), diff)

    # first run "crashes" after 10 pages: only the checkpoint manifest (every 25)
    # plus 10 corpus files exist; manifest was never finalized.
    real_one = run_corpus_compile

    class Boom(Exception):
        pass

    async def flaky(client, cfg_, **kw):
        sem = asyncio.Semaphore(4)
        completed: list[str] = []
        killed = {"v": False}

        async def guarded(rel):
            async with sem:
                if killed["v"]:
                    raise Boom("simulated kill")
                page_corpus = PageCorpus(source=rel, title="t", html_md5=scanned[rel],
                                         gen_key=gen_key, generated_at=now_iso(),
                                         chunks=[])
                CorpusStore(fm.corpus_dir).save(rel, page_corpus.to_json_dict())
                completed.append(rel)
                if len(completed) == 10:
                    # checkpoint claims ONLY the 10 processed pages; the old
                    # all-or-nothing checkpoint bug would record all 40 here
                    # and lose 30 pages forever
                    fm.save_manifest(
                        {r: scanned[r] for r in completed}, gen_key, gen_parts)
                if len(completed) > 10:
                    killed["v"] = True
                    raise Boom("simulated kill")

        await asyncio.gather(*(guarded(r) for r in kw["work"]))
        raise Boom()

    monkeypatch.setattr("rag.cli.compile_cmd.run_corpus_compile", flaky)
    import rag.cli.compile_cmd as compile_cmd
    with pytest.raises(Boom):
        await compile_cmd.run_corpus_compile(None, cfg, work=work, scanned=scanned,
                                             fm=fm, gen_key=gen_key,
                                             gen_parts=gen_parts,
                                             workers=4, progress=False)
    monkeypatch.undo()

    # pages completed before the kill have corpus files; the manifest may not
    # know about them, in which case they count as "added" again and their
    # (atomic) corpus files are simply overwritten. Either way the rerun
    # completes the remainder and every page ends up with a corpus file.
    store = CorpusStore(corpus_dir)
    # after the kill + honest checkpoint, only the 30 unprocessed pages requeue
    remaining = plan_work(fm, store, fm.diff(fm.scan()))
    assert len(remaining) == 30, len(remaining)
    report = await run_corpus_compile(
        FakeClient(), cfg, work=remaining, scanned=scanned, fm=fm,
        gen_key=gen_key, gen_parts=gen_parts, workers=4, progress=False)
    from rag.cli.compile_cmd import finalize
    finalize(fm, diff, scanned, gen_key, gen_parts, report)
    assert report.pages_done == 30
    assert len(list(store.iterate_all())) == 40
    # and a final rerun is a full no-op
    assert plan_work(fm, store, fm.diff(fm.scan())) == []


def test_plan_work_rejects_missing_only(env):
    root, corpus_dir, fm, cfg = env
    with pytest.raises(SystemExit):
        plan_work(fm, CorpusStore(corpus_dir), fm.diff(fm.scan()),
                  only="Nope/Missing.html")
