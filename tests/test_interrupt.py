"""Ctrl-C handling: interrupting a compile must quit FRIENDLY, never with a
traceback, and must not throw away work that was already paid for.

Reproduces the reported failure mode: Ctrl-C during the corpus step used to
surface as an ``asyncio`` ``CancelledError`` + ``KeyboardInterrupt`` traceback
thrown out of ``asyncio.runners``. The contract now is:

* the FIRST interrupt stops the workers from taking new pages and lets the pages
  in flight finish, be written and be checkpointed (so it costs ZERO pages), then
  the run returns a report with ``interrupted`` set;
* a SECOND interrupt (a ``KeyboardInterrupt`` raised inside an LLM call) is NOT
  catchable in the event loop — asyncio re-raises ``KeyboardInterrupt`` straight
  out of the loop (``asyncio.tasks``) — but the pages that had finished are still
  checkpointed, and the CLI still exits 130 with a friendly line;
* the CLI turns either one into a friendly line + exit status 130, and skips the
  index step (a Ctrl-C must not start a multi-minute dense build).

Real signal delivery is not portable inside a test process (``os.kill(pid,
SIGINT)`` kills the process on Windows), so the installed handler is looked up
with ``signal.getsignal`` and invoked directly — same code path, no signal.
"""
from __future__ import annotations

import asyncio
import io
import json
import signal
from pathlib import Path

import pytest

from rag.cli.compile_cmd import (
    INTERRUPT_EXIT_CODE,
    CompileReport,
    ProviderShard,
    install_interrupt_stop,
    interrupt_message,
    plan_work,
    restore_interrupt_stop,
    run_corpus_compile,
    set_gen_key,
    valid_gen_keys,
)
from rag.llm.base import GenerationResult
from rag.store import CorpusStore, FileManager

N_PAGES = 12


class FakeClient:
    """Fake provider: one verbatim chunk per page, one call per page.

    ``on_call`` runs INSIDE the LLM call, which is where the test gets to
    interleave: a real Ctrl-C lands exactly there too (the worker is suspended in
    the provider's HTTP read).
    """

    def __init__(self, model="fake-model", *, interrupt_on: int | None = None,
                 on_call=None):
        self._model = model
        self.calls = 0
        self.interrupt_on = interrupt_on
        self.on_call = on_call

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str):
        self.calls += 1
        if self.on_call is not None:
            await self.on_call(self.calls)
        if self.interrupt_on is not None and self.calls == self.interrupt_on:
            # a second Ctrl-C delivered while an LLM call is in flight: the handler
            # raises KeyboardInterrupt, which unwinds the gather
            raise KeyboardInterrupt()
        text = user_prompt.rstrip().splitlines()[-1].strip()[:120]
        return GenerationResult(
            text=json.dumps({"chunks": [{
                "heading_path": ["T"], "text": text, "summary": "s",
                "keywords": ["k"], "synonyms": [], "qa": [],
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
    root = make_site(tmp_path / "site", N_PAGES)
    corpus_dir = root / "corpus"
    fm = FileManager(root=root, dirs=["ScriptReference"], corpus_dir=corpus_dir)
    cfg = {"max_input_chars": 24000, "max_chunk_chars": 1200}
    return root, corpus_dir, fm, cfg


def start_run(fm, cfg, client, *, workers=1):
    """Schedule a one-worker corpus run; returns the task."""
    shard = ProviderShard(client, client.model_name)
    scanned = fm.scan()
    diff = fm.diff(scanned)
    work = plan_work(fm, CorpusStore(fm.corpus_dir), diff,
                     valid_gen_keys=valid_gen_keys([shard]))
    assert len(work) == N_PAGES
    task = asyncio.create_task(run_corpus_compile(
        [shard], cfg, work=work, scanned=scanned, fm=fm,
        workers_per_provider=workers, progress=False))
    return task


async def run_directly(fm, cfg, client, *, workers=1):
    """Await a one-worker corpus run in the CURRENT task.

    Used for the hard-abort case: a KeyboardInterrupt raised inside a task is
    re-raised out of the event loop, so it must be driven by a caller that owns
    the loop (``asyncio.run``) rather than awaited from a test coroutine.
    """
    shard = ProviderShard(client, client.model_name)
    scanned = fm.scan()
    diff = fm.diff(scanned)
    work = plan_work(fm, CorpusStore(fm.corpus_dir), diff,
                     valid_gen_keys=valid_gen_keys([shard]))
    return await run_corpus_compile(
        [shard], cfg, work=work, scanned=scanned, fm=fm,
        workers_per_provider=workers, progress=False)


def manifest_files(corpus_dir: Path) -> dict:
    path = corpus_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))["files"]


async def test_first_ctrl_c_drains_in_flight_pages(env, capfd):
    """The first Ctrl-C finishes the page in flight, then stops: no traceback,
    no discarded work, and the manifest on disk reflects every finished page."""
    root, corpus_dir, fm, cfg = env
    held = asyncio.Event()
    release = asyncio.Event()

    async def on_call(n: int) -> None:
        if n == 3:
            held.set()          # page 3 is now mid-LLM-call
            await release.wait()  # keep it there until the test has interrupted

    client = FakeClient(on_call=on_call)
    prev_handler = signal.getsignal(signal.SIGINT)
    task = start_run(fm, cfg, client)

    await held.wait()
    handler = signal.getsignal(signal.SIGINT)
    assert handler is not prev_handler, (
        "the corpus run must hook SIGINT while it is generating")
    handler(signal.SIGINT, None)                       # <- the first Ctrl-C
    assert "[interrupt]" in capfd.readouterr().err, "the user gets a notice"
    assert not task.done(), "in-flight pages are finished, not dropped"

    release.set()
    report = await task

    assert report.interrupted is True
    assert report.aborted_all_providers_down is False, (
        "a Ctrl-C is not a quota exhaustion; the two must not be confused")
    assert report.pages_done == 3, report.pages_done
    assert report.pages_planned == N_PAGES
    assert report.wall_s > 0
    # every page that finished is on disk AND claimed by the manifest
    assert len(manifest_files(corpus_dir)) == 3
    assert client.calls == 3
    assert "3 of 12" in interrupt_message(report)
    assert signal.getsignal(signal.SIGINT) is prev_handler, (
        "the run must restore the previous SIGINT handler")


def test_second_ctrl_c_checkpoints_what_finished(env):
    """A KeyboardInterrupt landing inside an LLM call (the second Ctrl-C) is
    loop-fatal by design — asyncio re-raises it out of the event loop, so it
    cannot be caught in the coroutine. What must still hold: the pages that had
    finished are on disk (per-page checkpoint + the final flush), and the caller
    sees a KeyboardInterrupt, which the CLI reports as exit 130."""
    root, corpus_dir, fm, cfg = env
    client = FakeClient(interrupt_on=3)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(run_directly(fm, cfg, client))

    assert len(manifest_files(corpus_dir)) == 2
    assert "were abandoned" in interrupt_message(hard=True)
    assert "Ctrl-C" in interrupt_message(None)


async def test_external_cancel_stays_a_cancellation(env):
    """An external cancellation must propagate as a CancelledError (never be
    swallowed) — asyncio.run turns its own SIGINT-driven cancellation back into
    the KeyboardInterrupt the CLI reports — after flushing what finished."""
    root, corpus_dir, fm, cfg = env
    held = asyncio.Event()
    release = asyncio.Event()

    async def on_call(n: int) -> None:
        if n == 3:
            held.set()
            await release.wait()

    client = FakeClient(on_call=on_call)
    prev_handler = signal.getsignal(signal.SIGINT)
    task = start_run(fm, cfg, client)
    await held.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(manifest_files(corpus_dir)) == 2
    assert signal.getsignal(signal.SIGINT) is prev_handler


def test_install_interrupt_stop_restores_and_second_call_raises():
    """The handler is a plain callable pair: first call stops, second raises."""
    aborted, interrupted = {"v": False}, {"v": False}
    prev = signal.getsignal(signal.SIGINT)
    try:
        installed = install_interrupt_stop(aborted, interrupted)
        assert installed is not None
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        assert aborted["v"] is True and interrupted["v"] is True
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)
    finally:
        restore_interrupt_stop(prev)
    assert signal.getsignal(signal.SIGINT) is prev


def test_run_compile_reports_interrupt_and_skips_index(tmp_path, monkeypatch):
    """Exit 130, the report carries the interrupt summary, and the index step is
    NOT started: the user just asked to stop, not to begin a multi-minute dense
    build."""
    import rag.compile as rc
    from rag.compile import run_compile
    from rag.cli.compile_cmd import print_report

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "dirs": ["docs"], "corpus_dir": "corpus", "index_dir": "index",
        "model": "fake", "type": "openai_legacy", "api_key": "sk-test",
        "url": "http://127.0.0.1:1/v1",
    }), encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.html").write_text("<html><body>a</body></html>",
                                              encoding="utf-8")

    buf = io.StringIO()

    def fake_corpus_step(*args, **kwargs):
        report = CompileReport(pages_total=10, pages_planned=10, pages_done=4)
        report.interrupted = True

        async def run():
            # run_corpus_step's tail: the report (whose last line is the interrupt
            # summary) is what the user reads — run_compile then only has to pick
            # the exit status.
            print_report(report, out=buf)
            return report
        return run()

    def boom(*args, **kwargs):
        raise AssertionError("the index step must not run after Ctrl-C")

    monkeypatch.setattr(rc, "run_corpus_step", fake_corpus_step)
    monkeypatch.setattr(rc, "run_index_step", boom)

    code = run_compile(configs=[str(cfg_path)], steps=["corpus", "index"])
    assert code == INTERRUPT_EXIT_CODE
    report_text = buf.getvalue()
    assert "Ctrl-C" in report_text and "4 of 10" in report_text
    assert "Traceback" not in report_text


def test_hard_abort_reports_friendly_exit_and_skips_index(tmp_path, monkeypatch,
                                                         capsys):
    """A KeyboardInterrupt out of the event loop (second Ctrl-C) is reported with
    the hard-abort wording, exits 130, and skips the index step just the same."""
    import rag.compile as rc
    from rag.compile import run_compile

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "dirs": ["docs"], "corpus_dir": "corpus",
        "model": "fake", "type": "openai_legacy", "api_key": "sk-test",
        "url": "http://127.0.0.1:1/v1",
    }), encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.html").write_text("<html><body>a</body></html>",
                                              encoding="utf-8")

    def fake_corpus_step(*args, **kwargs):
        async def run():
            raise KeyboardInterrupt
        return run()

    def boom(*args, **kwargs):
        raise AssertionError("the index step must not run after Ctrl-C")

    monkeypatch.setattr(rc, "run_corpus_step", fake_corpus_step)
    monkeypatch.setattr(rc, "run_index_step", boom)

    code = run_compile(configs=[str(cfg_path)], steps=["corpus", "index"])
    assert code == INTERRUPT_EXIT_CODE
    err = capsys.readouterr().err
    assert "Ctrl-C" in err and "abandoned" in err and "Traceback" not in err


def test_cli_main_turns_keyboard_interrupt_into_a_friendly_exit(monkeypatch,
                                                                capsys):
    """The last-resort guard: any command (compile, search, repl, ...) that is
    interrupted reports one line on stderr and exits 130 — never a traceback."""
    import rag.__main__ as cli

    def boom(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_dispatch", boom)
    assert cli.main(["status"]) == cli.INTERRUPT_EXIT_CODE
    err = capsys.readouterr().err
    assert "[interrupt]" in err
    assert "Traceback" not in err
    assert cli.INTERRUPT_EXIT_CODE == 130
