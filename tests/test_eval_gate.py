"""eval_rag.emit() gate-verdict tests (network-free, monkeypatched coverage).

The coverage guard is the single most important piece of eval honesty: a subset
index has far fewer distractors than the full mirror, so its scores are optimistic
and NOT comparable to the baseline. Measured live: BM25-only MRR 0.917 on a
3k-page probe but 0.826 on 26k pages. Without this guard a partial run prints
"GATE PASS" and gets quoted as the final result — which is exactly what happened
during development. These tests pin the behaviour down.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import eval_rag


@pytest.fixture()
def cap(tmp_path, monkeypatch):
    """Route emit()'s file write and coverage inputs to controllable fakes."""
    # redirect the results file into tmp so the real one is untouched
    monkeypatch.setattr(eval_rag, "__file__", str(tmp_path / "eval_rag.py"))

    state = {"coverage_pages": 3000, "total_pages": 43938}

    class FakeCorpusStore:
        def __init__(self, *a, **k):
            pass

        def iterate_all(self):
            for i in range(state["coverage_pages"]):
                yield f"p{i}", {}

    monkeypatch.setattr("rag.store.CorpusStore", FakeCorpusStore)
    monkeypatch.setattr("rag.search.engine_select.count_html_pages",
                        lambda dirs: state["total_pages"])
    monkeypatch.setattr("rag.compile.load_rag_config",
                        lambda *a, **k: {"dirs": ["Manual"], "corpus_dir": "corpus"})
    monkeypatch.setattr("rag.resolve_path", lambda p, *a, **k: tmp_path / str(p))

    # capture what emit() prints
    lines: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: lines.append(" ".join(str(x) for x in a)))
    return SimpleNamespace(state=state, lines=lines)


def _rows(mrr, gold="base-24"):
    return [{"config": "engine bm25", "gold": gold, "MRR": mrr,
             "hit@1": mrr, "hit@3": mrr, "hit@5": mrr, "hit@10": 0.95,
             "recall@1": mrr, "recall@3": mrr, "recall@5": mrr,
             "recall@10": mrr}]


def test_partial_corpus_defers_gate(cap):
    """6.9% coverage must DEFER even when MRR clears the threshold."""
    cap.state["coverage_pages"] = 3000
    cap.state["total_pages"] = 43938
    rc = eval_rag.emit(_rows(0.917), SimpleNamespace(config="rag_config.json"))
    assert rc == 0
    joined = "\n".join(cap.lines)
    assert "DEFERRED" in joined
    assert "GATE PASS" not in joined, "a partial corpus must never print PASS"
    assert "optimistic" in joined


def test_full_corpus_passes_when_metrics_clear(cap):
    cap.state["coverage_pages"] = 43000
    cap.state["total_pages"] = 43938  # 97.9% >= 95%
    rc = eval_rag.emit(_rows(0.90), SimpleNamespace(config="rag_config.json"))
    assert rc == 0
    joined = "\n".join(cap.lines)
    assert "GATE PASS" in joined
    assert "DEFERRED" not in joined


def test_full_corpus_fails_when_metrics_short(cap):
    cap.state["coverage_pages"] = 43000
    cap.state["total_pages"] = 43938
    rc = eval_rag.emit(_rows(0.70), SimpleNamespace(config="rag_config.json"))
    assert rc == 0
    joined = "\n".join(cap.lines)
    assert "GATE FAIL" in joined


def test_contaminated_gold_set_never_verdicts(cap):
    """ext/probe sets are harvested from indexed qa.q — no gate verdict from them."""
    cap.state["coverage_pages"] = 43000
    cap.state["total_pages"] = 43938
    rc = eval_rag.emit(_rows(0.99, gold="ext-16"),
                       SimpleNamespace(config="rag_config.json"))
    assert rc == 0
    joined = "\n".join(cap.lines)
    assert "contaminated" in joined
    assert "GATE PASS" not in joined
    assert "GATE FAIL" not in joined


def test_exactly_95_percent_coverage_is_not_deferred(cap):
    cap.state["coverage_pages"] = 9500
    cap.state["total_pages"] = 10000  # exactly 95%
    eval_rag.emit(_rows(0.90), SimpleNamespace(config="rag_config.json"))
    joined = "\n".join(cap.lines)
    assert "DEFERRED" not in joined


def test_writes_latest_file_not_curated(cap, monkeypatch):
    """emit() must write eval_results_latest.md, never clobber eval_results.md."""
    cap.state["coverage_pages"] = 43000
    cap.state["total_pages"] = 43938
    # make Path writes observable
    real_write = eval_rag.Path.write_text
    written_names = []

    def spy(self, text, **k):
        written_names.append(self.name)
        return real_write(self, text, **k)

    monkeypatch.setattr(eval_rag.Path, "write_text", spy)
    eval_rag.emit(_rows(0.90), SimpleNamespace(config="rag_config.json"))
    assert "eval_results_latest.md" in written_names
    assert "eval_results.md" not in written_names
