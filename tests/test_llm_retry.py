"""Rate-limit / transient-failure retry schedule tests (network-free).

The compile corpus step runs for a day against gateways that rate limit (HTTP
429 past ~4 concurrent requests per key) and exhaust rolling 5-hour quota
windows. A transient failure must therefore cost LATENCY, not corpus quality:
each LLM call waits out a fixed schedule — 2 s -> 4 s -> 1 min -> 10 min -> 1 h ->
2 h -> 4 h — and only then is the page demoted to the aux-less heuristic
fallback. These tests pin the schedule, the give-up boundary, the retriable /
non-retriable split, the Ctrl-C-during-backoff contract, and the plumbing from
config to the compile loop.

No test ever really waits: sleeps are recorded by a patched ``asyncio.sleep``, or
the schedule is injected as zeros.
"""
from __future__ import annotations

import asyncio
import json
import signal
from pathlib import Path

import pytest

from rag.cli.compile_cmd import (
    ProviderShard,
    plan_work,
    run_corpus_compile,
    )
from rag.corpus.extract import PageInput
from rag.corpus.generate import MAX_RETRIES, generate_page_corpus
from rag.llm import base as llm_base
from rag.llm.base import (
    RETRY_DELAYS,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    GenerationResult,
    RateLimitError,
    RetryAborted,
    is_retriable,
    retry_schedule,
    with_retry,
)
from rag.store import CorpusStore, FileManager

# --------------------------------------------------------------------------------------
# the schedule itself
# --------------------------------------------------------------------------------------


def test_default_schedule_is_the_agreed_ladder():
    """2 s -> 4 s -> 1 min -> 10 min -> 1 h -> 2 h -> 4 h, in seconds."""
    assert RETRY_DELAYS == (2.0, 4.0, 60.0, 600.0, 3600.0, 7200.0, 14400.0)
    assert retry_schedule() == RETRY_DELAYS
    # every LLM call gets one retry per schedule entry (8 attempts total)
    assert MAX_RETRIES == len(RETRY_DELAYS)


def test_schedule_overrides_and_caps():
    assert retry_schedule([1, 2]) == (1.0, 2.0)
    assert retry_schedule([1, 2, 3], retries=2) == (1.0, 2.0)
    # retries only CAPS: a shortened schedule is never padded with its last entry
    assert retry_schedule([1, 2], retries=9) == (1.0, 2.0)
    assert retry_schedule(None, retries=0) == ()


def test_bad_schedule_fails_loudly():
    with pytest.raises(ValueError):
        retry_schedule([])
    with pytest.raises(ValueError):
        retry_schedule([1, -2])
    with pytest.raises((ValueError, TypeError)):
        retry_schedule("not a list of numbers")


def test_retriable_classification():
    assert is_retriable(RateLimitError("429"))                    # the headline case
    assert is_retriable(APIStatusError("busy", status_code=503))
    assert is_retriable(APITimeoutError("t"))
    assert is_retriable(APIConnectionError("reset"))
    # quota / access errors are the circuit breaker's business, not a backoff's
    assert not is_retriable(APIStatusError("quota", status_code=403))
    assert not is_retriable(APIStatusError("auth", status_code=401))
    assert not is_retriable(ValueError("model output"))


# --------------------------------------------------------------------------------------
# with_retry
# --------------------------------------------------------------------------------------


@pytest.fixture()
def recorded_sleeps(monkeypatch):
    """Replace asyncio.sleep inside the retry helper with a recorder."""
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(llm_base.asyncio, "sleep", fake_sleep)
    return slept


async def test_with_retry_waits_the_schedule_in_order(recorded_sleeps):
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] <= 3:
            raise RateLimitError("429 too many requests")
        return GenerationResult(text="ok")

    seen: list[tuple[int, float]] = []
    result = await with_retry(
        flaky, on_retry=lambda exc, n, delay: seen.append((n, delay)))

    assert result.text == "ok"
    assert calls["n"] == 4
    assert recorded_sleeps == [2.0, 4.0, 60.0]      # the first three rungs
    assert seen == [(1, 2.0), (2, 4.0), (3, 60.0)]


async def test_with_retry_spends_the_whole_schedule_then_raises(recorded_sleeps):
    async def always_limited():
        raise RateLimitError("429")

    with pytest.raises(RateLimitError):
        await with_retry(always_limited)

    assert recorded_sleeps == list(RETRY_DELAYS), (
        "every rung is used once, including the 4-hour one, before giving up")


async def test_with_retry_honours_a_custom_schedule(recorded_sleeps):
    async def always_limited():
        raise RateLimitError("429")

    with pytest.raises(RateLimitError):
        await with_retry(always_limited, delays=(0.1, 0.2))
    assert recorded_sleeps == [0.1, 0.2]


async def test_with_retry_does_not_wait_on_a_permanent_failure(recorded_sleeps):
    async def denied():
        raise APIStatusError("AccessDenied.Unpurchased", status_code=403)

    with pytest.raises(APIStatusError):
        await with_retry(denied)
    assert recorded_sleeps == [], "403 must fail over, not back off"


async def test_with_retry_recovers_from_a_transient_blip_without_retrying_forever(
        recorded_sleeps):
    """5xx/timeouts take the same ladder as 429 — that is the 'other errors' case."""
    calls = {"n": 0}

    async def timeout_then_ok():
        calls["n"] += 1
        if calls["n"] == 1:
            raise APITimeoutError("read timed out")
        return GenerationResult(text="ok")

    assert (await with_retry(timeout_then_ok)).text == "ok"
    assert recorded_sleeps == [2.0]


# --------------------------------------------------------------------------------------
# Ctrl-C during a long backoff
# --------------------------------------------------------------------------------------


async def test_abort_during_backoff_stops_immediately(monkeypatch):
    """Ctrl-C must not be held hostage by a 1-hour rung: no second attempt, no sleep."""
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(llm_base.asyncio, "sleep", fake_sleep)
    calls = {"n": 0}
    aborted = {"v": False}

    async def limited():
        calls["n"] += 1
        aborted["v"] = True          # the user pressed Ctrl-C during this request
        raise RateLimitError("429")

    with pytest.raises(RetryAborted):
        await with_retry(limited, should_abort=lambda: aborted["v"])

    assert calls["n"] == 1, "the aborted retry must never be sent"
    assert slept == [], "the 2s/4s/.../4h wait is skipped entirely"


async def test_should_abort_is_polled_during_a_long_wait(monkeypatch):
    """A multi-hour rung is slept in slices so Ctrl-C lands within seconds."""
    monkeypatch.setattr(llm_base, "_ABORT_POLL_S", 0.01)
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(llm_base.asyncio, "sleep", fake_sleep)
    polls = {"n": 0}
    calls = {"n": 0}

    def abort_after_three_polls() -> bool:
        polls["n"] += 1
        return polls["n"] > 3

    async def limited():
        calls["n"] += 1
        raise RateLimitError("429")

    with pytest.raises(RetryAborted):
        await with_retry(limited, delays=(14400.0,),
                         should_abort=abort_after_three_polls)

    assert calls["n"] == 1, "the aborted retry was never sent"
    assert len(slept) == 3 and all(s <= 0.01 for s in slept), (
        f"a 4-hour wait must be sliced, never one sleep: {slept}")


# --------------------------------------------------------------------------------------
# page level: a rate limit is waited out, an abort degrades nothing
# --------------------------------------------------------------------------------------


MARKDOWN = """# Rigidbody

## Description

Controls the position and velocity of a GameObject through physics simulation.

The velocity of the rigidbody. You can read it in every frame.
"""


class ScriptedClient:
    """Pops error/response per call and records prompts (no network)."""

    def __init__(self, responses: list[str], errors: list | None = None,
                 model: str = "fake-model"):
        self.responses = list(responses)
        self.errors = list(errors or [])
        self.prompts: list[tuple[str, str]] = []
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str):
        self.prompts.append((system_prompt, user_prompt))
        if self.errors:
            raise self.errors.pop(0)
        if not self.responses:
            raise AssertionError("scripted client out of responses")
        return GenerationResult(text=self.responses.pop(0),
                                input_tokens=100, output_tokens=50)


def good_json(text: str) -> str:
    return json.dumps({"chunks": [{
        "heading_path": ["Rigidbody"], "text": text,
        "summary": "s", "keywords": ["k"], "synonyms": [], "qa": [],
    }]})


@pytest.fixture()
def page() -> PageInput:
    return PageInput(source="ScriptReference/Rigidbody.html", title="Rigidbody",
                     markdown=MARKDOWN, char_len=len(MARKDOWN))


async def test_page_rides_out_rate_limits_and_keeps_real_chunks(page):
    """Two 429s then success: real LLM chunks, NOT the heuristic fallback."""
    excerpt = "Controls the position and velocity of a GameObject through physics simulation."
    client = ScriptedClient([good_json(excerpt)],
                            errors=[RateLimitError("429"), RateLimitError("429")])
    corpus, stats = await generate_page_corpus(
        client, page, gen_key="gk", html_md5="md5", retry_delays=(0.0, 0.0, 0.0))

    assert not stats.fallback and stats.first_try_valid
    assert stats.attempts == 1, "retries live INSIDE one LLM session"
    assert len(client.prompts) == 3, (
        "two 429s plus the retry that succeeded — all on the same page prompt")
    assert len({p[1] for p in client.prompts}) == 1
    assert stats.errors[:2] == ["retry 1: 429", "retry 2: 429"], stats.errors
    assert corpus.chunks[0].summary == "s", "real aux fields, not the fallback"
    assert corpus.needs_regen is False


async def test_page_still_degrades_when_the_schedule_is_exhausted(page):
    """Genuinely dead provider: the old contract still holds — flag, don't wedge."""
    client = ScriptedClient([], errors=[RateLimitError("429")] * 4)
    corpus, stats = await generate_page_corpus(
        client, page, gen_key="gk", html_md5="md5", retry_delays=(0.0,))

    assert stats.fallback and stats.api_failed
    assert corpus.chunks[0].summary == ""      # aux-less heuristic chunks


async def test_long_backoff_is_announced_on_stderr(page, capsys, recorded_sleeps):
    """A run waiting 10 minutes must look like waiting, not like hanging."""
    excerpt = "Controls the position and velocity of a GameObject through physics simulation."
    client = ScriptedClient([good_json(excerpt)], errors=[RateLimitError("429")])
    await generate_page_corpus(client, page, gen_key="gk", html_md5="md5",
                               retry_delays=(600.0,))

    assert recorded_sleeps[-1] == 600.0
    err = capsys.readouterr().err
    assert "[retry] fake-model" in err and "10min" in err, err


async def test_abort_during_backoff_drops_the_page_unfinished(page):
    """Ctrl-C mid-wait propagates RetryAborted — never a heuristic fallback write."""
    client = ScriptedClient([], errors=[RateLimitError("429")])
    with pytest.raises(RetryAborted):
        await generate_page_corpus(client, page, gen_key="gk", html_md5="md5",
                                   retry_delays=(14400.0,),
                                   should_abort=lambda: True)
    assert len(client.prompts) == 1, "the aborted retry was never sent"


# --------------------------------------------------------------------------------------
# compile-loop plumbing
# --------------------------------------------------------------------------------------


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


class FlakyClient:
    """Rate limits the first ``limit`` calls, then serves verbatim chunks."""

    def __init__(self, model="flaky", *, limit: int = 2, on_first_call=None):
        self._model = model
        self.limit = limit
        self.calls = 0
        self.on_first_call = on_first_call

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str):
        self.calls += 1
        if self.calls == 1 and self.on_first_call is not None:
            await self.on_first_call()
        if self.calls <= self.limit:
            raise RateLimitError("429 rate limit exceeded")
        text = user_prompt.rstrip().splitlines()[-1].strip()[:120]
        return GenerationResult(
            text=json.dumps({"chunks": [{"heading_path": ["T"], "text": text,
                                        "summary": "s", "keywords": ["k"],
                                        "synonyms": [], "qa": []}]}),
            input_tokens=50, output_tokens=25)


@pytest.fixture()
def env(tmp_path):
    root = make_site(tmp_path / "site", 6)
    fm = FileManager(root=root, dirs=["ScriptReference"],
                     corpus_dir=tmp_path / "corpus")
    cfg = {"max_input_chars": 24000, "max_chunk_chars": 1200}
    return root, tmp_path / "corpus", fm, cfg


async def test_compile_run_passes_the_configured_schedule_through(env):
    """A 429 at the start of a page costs nothing but the wait: real chunks land."""
    root, corpus_dir, fm, cfg = env
    client = FlakyClient(limit=2)
    shards = [ProviderShard(client, client.model_name)]
    scanned = fm.scan()
    store = CorpusStore(corpus_dir)
    work = plan_work(fm, store, fm.diff(scanned))

    report = await run_corpus_compile(
        shards, cfg, work=work, scanned=scanned, fm=fm,
        workers_per_provider=2, progress=False, retry_delays=(0.0, 0.0))

    assert report.pages_done == len(work)
    assert report.fallback == 0, "rate limits must not degrade pages"
    assert report.needs_regen == set()
    for rel, data in store.iterate_all():
        assert data["chunks"][0]["summary"] == "s", rel


async def test_ctrl_c_during_a_long_backoff_leaves_everything_untouched(env):
    """The graceful-interrupt contract survives the long rungs: Ctrl-C while a page
    waits out a rate limit drops that page UNFINISHED (no heuristic write) and the
    run reports the interrupt, so a rerun resumes with nothing to clean up."""
    root, corpus_dir, fm, cfg = env

    def press_ctrl_c() -> None:
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler), "the compile run must hook SIGINT"
        handler(signal.SIGINT, None)

    async def ctrl_c_lands() -> None:
        press_ctrl_c()      # exactly when a real Ctrl-C would land: mid-request

    client = FlakyClient(limit=10, on_first_call=ctrl_c_lands)
    shards = [ProviderShard(client, client.model_name)]
    scanned = fm.scan()
    store = CorpusStore(corpus_dir)
    work = plan_work(fm, store, fm.diff(scanned))

    try:
        report = await asyncio.wait_for(run_corpus_compile(
            shards, cfg, work=work, scanned=scanned, fm=fm,
            workers_per_provider=1, progress=False,
            # a 4-hour rung: the test only passes because the abort cuts it short
            retry_delays=(14400.0,)), timeout=30)
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)

    assert report.interrupted is True
    assert report.aborted_all_providers_down is False
    assert client.calls == 1, (
        "the page WAS in flight and its first retry was cut short")
    assert report.pages_done == 0, "the aborted page is not counted as done"
    assert report.fallback == 0, "and it is certainly not degraded"
    assert list(store.iterate_all()) == [], "nothing was written for it"
    assert fm.load_manifest().get("files", {}) == {}
