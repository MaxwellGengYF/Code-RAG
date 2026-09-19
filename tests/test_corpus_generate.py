"""Corpus generation tests with a scripted fake LLMClient (network-free)."""
from __future__ import annotations

import json

import pytest

from rag.corpus.generate import (
    chunk_text,
    extract_json_object,
    generate_page_corpus,
    validate_llm_output,
)
from rag.corpus.extract import PageInput, extract_page, render_markdown
from rag.corpus.schema import (
    PageCorpus,
    make_chunk_uid,
    norm_ws,
    snippet,
    to_embed_text,
    to_index_text,
)

MARKDOWN = """# Rigidbody

## Description

Controls the position and velocity of a GameObject through physics simulation.

## Properties

```csharp
public Vector3 velocity;
```

The velocity of the rigidbody. You can read it in every frame.
"""


class FakeClient:
    """Scripted client: pops a response per call, records prompts."""

    def __init__(self, responses: list[str], errors: list | None = None):
        self.responses = list(responses)
        self.errors = list(errors or [])
        self.prompts: list[tuple[str, str]] = []

    @property
    def model_name(self):
        return "fake"

    async def generate(self, system_prompt: str, user_prompt: str):
        from rag.llm.base import GenerationResult
        self.prompts.append((system_prompt, user_prompt))
        if self.errors:
            raise self.errors.pop(0)
        if not self.responses:
            raise AssertionError("fake client out of responses")
        return GenerationResult(text=self.responses.pop(0),
                                input_tokens=100, output_tokens=50)


def good_json() -> str:
    return json.dumps({
        "chunks": [{
            "heading_path": ["Rigidbody", "Properties"],
            "text": "```csharp\npublic Vector3 velocity;\n```\n\nThe velocity of the rigidbody.",
            "summary": "Rigidbody.velocity linear velocity property.",
            "keywords": ["rigidbody", "velocity", "physics"],
            "synonyms": ["linear velocity", "linearVelocity (script alias)"],
            "qa": [{"q": "How do I read a Rigidbody's velocity?",
                    "a": "Read the Rigidbody.velocity property."}],
        }]
    })


@pytest.fixture()
def page() -> PageInput:
    return PageInput(source="ScriptReference/Rigidbody.html", title="Rigidbody",
                     markdown=MARKDOWN, char_len=len(MARKDOWN))


async def test_happy_path(page):
    client = FakeClient([good_json()])
    corpus, stats = await generate_page_corpus(
        client, page, gen_key="gk", html_md5="md5", max_chunk_chars=1200)
    assert stats.first_try_valid and not stats.fallback
    assert stats.attempts == 1
    assert stats.input_tokens == 100
    assert corpus.source == page.source
    assert corpus.html_md5 == "md5"
    assert corpus.gen_key == "gk"
    ch = corpus.chunks[0]
    assert ch.chunk_uid == make_chunk_uid(page.source, 0)
    assert norm_ws(ch.text) in norm_ws(page.markdown)  # verbatim invariant
    assert ch.summary and ch.keywords and ch.synonyms and ch.qa


async def test_malformed_json_then_repair(page):
    client = FakeClient(["this is not json at all", good_json()])
    corpus, stats = await generate_page_corpus(
        client, page, gen_key="gk", html_md5="md5")
    assert stats.repaired and not stats.fallback
    assert stats.attempts == 2
    # repair prompt echoes validation errors
    assert "failed validation" in client.prompts[1][1]


async def test_non_substring_text_repairs(page):
    bad = json.dumps({"chunks": [{"heading_path": [], "text": "Completely made up prose that appears nowhere in the page."}]})
    client = FakeClient([bad, good_json()])
    corpus, stats = await generate_page_corpus(
        client, page, gen_key="gk", html_md5="md5")
    assert stats.repaired
    assert norm_ws(corpus.chunks[0].text) in norm_ws(page.markdown)


async def test_fenced_json_accepted(page):
    client = FakeClient(["```json\n" + good_json() + "\n```"])
    corpus, stats = await generate_page_corpus(
        client, page, gen_key="gk", html_md5="md5")
    assert stats.first_try_valid


async def test_api_failure_falls_back(page):
    from rag.llm.base import RateLimitError
    client = FakeClient([], errors=[RateLimitError("429"), RateLimitError("429")])
    corpus, stats = await generate_page_corpus(
        client, page, gen_key="gk", html_md5="md5", retries=1)
    assert stats.fallback
    assert len(corpus.chunks) >= 1
    assert corpus.chunks[0].summary == ""  # empty aux fields in fallback
    assert norm_ws(corpus.chunks[0].text) in norm_ws(page.markdown)


async def test_persisted_failure_falls_back(page):
    # both attempts return invalid output
    client = FakeClient(["no json here", "still not json"])
    corpus, stats = await generate_page_corpus(
        client, page, gen_key="gk", html_md5="md5")
    assert stats.fallback and stats.attempts == 2


def test_chunk_text_port():
    text = "\n".join(f"paragraph {i} with some words." for i in range(100))
    chunks = chunk_text(text, max_len=800, overlap=120)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 800 + 40  # paragraph overrun tolerance


def test_extract_json_object():
    assert extract_json_object('noise {"a": 1} tail') == {"a": 1}
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json_object("nothing here")
    with pytest.raises(ValueError):
        extract_json_object("[1, 2]")


def test_validate_caps_qa(page):
    obj = {"chunks": [{
        "heading_path": [], "text": MARKDOWN[:100],
        "qa": [{"q": f"q{i}", "a": f"a{i}"} for i in range(6)],
    }]}
    chunks, errors = validate_llm_output(obj, page, 1200)
    assert not errors
    assert len(chunks[0].qa) == 3


def test_index_text_composition():
    from rag.corpus.schema import CorpusChunk, QA
    ch = CorpusChunk(chunk_uid="u", heading_path=["Rigidbody", "Properties"],
                     text="body text", summary="sum", keywords=["k1"],
                     synonyms=["s1"], qa=[QA(q="question", a="answer")])
    idx = to_index_text(ch, title="Rigidbody")
    for part in ("Rigidbody", "Properties", "body text", "sum", "k1", "s1", "question"):
        assert part in idx
    emb = to_embed_text(ch, title="Rigidbody")
    assert "body text" in emb and "sum" not in emb and "k1" not in emb
    assert "question" not in emb


def test_snippet_centers_on_hit():
    from rag.corpus.schema import CorpusChunk
    body = ("intro " * 100) + "needle target" + (" outro" * 100)
    ch = CorpusChunk(chunk_uid="u", text=body)
    snip, truncated = snippet(ch, ["needle"], width=120)
    assert "needle" in snip
    assert truncated


def test_extract_page_real_html(tmp_site):
    pages = list(tmp_site.rglob("*.html"))
    results = [extract_page(p, root=tmp_site) for p in pages]
    results = [r for r in results if r]
    assert len(results) == 2
    rb = next(r for r in results if "Rigidbody" in r.source)
    assert rb.title == "Rigidbody"
    assert "```csharp" in rb.markdown
    assert "public Vector3 velocity;" in rb.markdown


def test_extract_page_truncation(tmp_site):
    p = next(tmp_site.rglob("Rigidbody.html"))
    r = extract_page(p, root=tmp_site, max_chars=50)
    assert r is not None
    assert "truncated" in r.markdown
    assert r.char_len > 50


def test_pagecorpus_roundtrip():
    corpus = PageCorpus(source="s", title="t", html_md5="m", gen_key="g",
                        generated_at="now")
    data = json.dumps(corpus.to_json_dict()).encode()
    back = PageCorpus.from_json_bytes(data)
    assert back.source == "s" and back.chunks == []
