"""Unit tests for rag/search/format.py markdown translators."""
from __future__ import annotations

from rag.search.format import (mentions_to_markdown, result_to_markdown,
                               results_to_markdown)


def _hit(rank=1, title="Rigidbody", source="ScriptReference/Rigidbody.html",
         score=0.123456, text="The velocity\n of the rigidbody.  ",
         read_more="uv run python dumpdoc.py ScriptReference/Rigidbody.html"):
    return {"rank": rank, "title": title, "source": source,
            "fused_score": score, "text": text, "read_more": read_more}


def test_result_heading_and_hits():
    r = {"query": "rigidbody velocity", "hits": [_hit()]}
    md = result_to_markdown(r)
    lines = md.splitlines()
    assert lines[0] == "### rigidbody velocity"
    assert lines[1] == ("1. **Rigidbody** `ScriptReference/Rigidbody.html` "
                        "score=0.1235")
    assert lines[2] == "  > The velocity of the rigidbody."
    assert lines[3] == ("  read: uv run python dumpdoc.py "
                        "ScriptReference/Rigidbody.html")


def test_result_bm25_only_marker():
    r = {"query": "rigidbody velocity", "hits": [_hit()]}
    assert result_to_markdown(r, dense=False, mode="hybrid").splitlines()[0] \
        == "### rigidbody velocity (bm25-only)"
    assert result_to_markdown(r, dense=False, mode="dense").splitlines()[0] \
        == "### rigidbody velocity (bm25-only)"
    # bm25 mode / dense available: no marker
    assert result_to_markdown(r, dense=False, mode="bm25").splitlines()[0] \
        == "### rigidbody velocity"
    assert result_to_markdown(r, dense=True, mode="hybrid").splitlines()[0] \
        == "### rigidbody velocity"


def test_result_zero_hits_hint():
    r = {"query": "zzz", "hits": [],
         "hint": "0 results. Try --explain ..."}
    md = result_to_markdown(r)
    assert "0 results. Try --explain ..." in md
    assert "**" not in md
    # no hint at all -> generic line
    assert "no results" in result_to_markdown({"query": "z", "hits": []})


def test_result_explain_line():
    r = {"query": "q", "hits": [_hit()],
         "explain": [{"term": "rigidbody", "df": 5, "in_index": True},
                     {"term": "zz", "df": 0, "in_index": False}]}
    md = result_to_markdown(r)
    assert md.splitlines()[-1] == "terms in index: 1/2 ['rigidbody']"


def test_results_to_markdown_uses_meta():
    payload = {
        "_meta": {"engine": "rag.search", "mode": "hybrid", "dense": False},
        "results": [{"query": "a", "hits": [_hit(rank=1)]},
                    {"query": "b", "hits": []}],
    }
    md = results_to_markdown(payload)
    parts = md.split("\n\n")
    assert len(parts) == 2
    assert parts[0].startswith("### a (bm25-only)")
    assert parts[1].startswith("### b (bm25-only)")


def test_mentions_to_markdown():
    rows = [{"source": "ScriptReference/A.html", "count": 3,
             "context": "some text"},
            {"source": "Manual/B.html", "count": 1}]
    md = mentions_to_markdown("Foo", rows)
    lines = md.splitlines()
    assert lines[0] == "### mentions: Foo (2 files)"
    assert lines[1] == "- `ScriptReference/A.html` ×3 — some text"
    assert lines[2] == "- `Manual/B.html` ×1"
