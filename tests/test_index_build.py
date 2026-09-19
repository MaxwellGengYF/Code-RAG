"""SA-5 index-build acceptance: chunk_uid stability + dual-write consistency.

SA-5 requires "chunk_uid stable across rebuilds" and that one chunk table keys
both index layers. uid stability matters because it is the join key between BM25
rows and dense rows: if a rebuild permuted uids, cached references and any
external consumer would silently point at the wrong chunk.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.corpus.schema import make_chunk_uid
from rag.index.build import decode_rows, encode_rows, flatten_corpus
from rag.store import CorpusStore


def write_corpus(corpus_dir: Path, rel: str, chunks: list[dict]) -> None:
    path = corpus_dir / (rel + ".rag.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "source": rel, "title": Path(rel).stem, "html_md5": "x",
        "gen_key": "gk", "generated_at": "now", "chunks": chunks,
    }, ensure_ascii=False), encoding="utf-8")


def chunk(i: int, text: str) -> dict:
    return {"chunk_uid": "ignored", "heading_path": ["H"], "text": text,
            "summary": f"summary {i}", "keywords": [f"k{i}"], "synonyms": [],
            "qa": [{"q": f"q{i}", "a": f"a{i}"}]}


@pytest.fixture()
def corpus(tmp_path):
    d = tmp_path / "corpus"
    write_corpus(d, "ScriptReference/Rigidbody.html",
                 [chunk(0, "velocity of the rigidbody"),
                  chunk(1, "mass of the rigidbody")])
    write_corpus(d, "Manual/Physics.html", [chunk(0, "physics overview")])
    return d


def test_chunk_uid_stable_across_rebuilds(corpus):
    """Flattening the same corpus twice must produce identical uids in the same order."""
    store = CorpusStore(corpus)
    a, _ = flatten_corpus(store)
    b, _ = flatten_corpus(store)
    assert [r.chunk_uid for r in a] == [r.chunk_uid for r in b]
    assert [r.source for r in a] == [r.source for r in b]


def test_chunk_uid_survives_msgpack_roundtrip(corpus, tmp_path):
    """The persisted chunk table must round-trip uids exactly (they key both indexes)."""
    store = CorpusStore(corpus)
    rows, _ = flatten_corpus(store)
    blob = encode_rows(rows)
    (tmp_path / "chunks.msgpack").write_bytes(blob)

    back = decode_rows((tmp_path / "chunks.msgpack").read_bytes())
    assert [r.chunk_uid for r in back] == [r.chunk_uid for r in rows]
    assert [r.text for r in back] == [r.text for r in rows]
    assert back[0].keywords == rows[0].keywords
    assert back[0].qa[0].q == rows[0].qa[0].q


def test_chunk_uid_is_deterministic_function_of_source_and_index():
    """uid = sha1(source#idx): same inputs -> same id, different idx -> different id."""
    a = make_chunk_uid("ScriptReference/Rigidbody.html", 0)
    b = make_chunk_uid("ScriptReference/Rigidbody.html", 0)
    c = make_chunk_uid("ScriptReference/Rigidbody.html", 1)
    d = make_chunk_uid("ScriptReference/Other.html", 0)
    assert a == b
    assert a != c and a != d
    assert len(a) == 16 and all(ch in "0123456789abcdef" for ch in a)


def test_flatten_is_deterministic_regardless_of_walk_order(corpus):
    """iterate_all() order must not leak into the table order (sorted by rel)."""
    store = CorpusStore(corpus)
    rows, n_pages = flatten_corpus(store)
    assert n_pages == 2
    sources = [r.source for r in rows]
    assert sources == sorted(sources), "chunk table must be sorted by source"
    # Manual/ sorts before ScriptReference/
    assert sources[0] == "Manual/Physics.html"


def test_dual_write_alignment_bm25_rows_match_dense_rows(corpus, tmp_path):
    """One chunk table feeds both indexes; row i must be the same chunk in both.

    This is the invariant vectors.f32 relies on: dense rows carry no ids, so
    alignment is purely positional. Asserting it here means a future refactor that
    reorders rows between the two builders breaks a test instead of silently
    mis-attributing every dense score.
    """
    from rag.corpus.schema import to_embed_text, to_index_text
    from rag.index.bm25_index import doc_tokens
    from unity_tokenizer import WordTokenizer

    store = CorpusStore(corpus)
    rows, _ = flatten_corpus(store)
    chunks = [r.to_corpus_chunk() for r in rows]

    tok = WordTokenizer()
    bm25_side = [doc_tokens(chunks[i], rows[i].source, rows[i].title, tok, 3)
                 for i in range(len(rows))]
    dense_side = [to_embed_text(chunks[i], rows[i].title) for i in range(len(rows))]

    assert len(bm25_side) == len(dense_side) == len(rows)
    # row 0 is the same chunk on both sides: its own text appears in both renderings
    assert "physics" in dense_side[0].lower()
    assert "physics" in " ".join(bm25_side[0]).lower()
    # and the aux field only appears on the BM25 side (the design invariant)
    assert "summary 0" not in dense_side[0]
    assert "summary" in " ".join(bm25_side[0])


def test_index_text_includes_aux_embed_text_does_not(corpus):
    """The core design invariant: aux boosts lexical recall, never the vector space."""
    from rag.corpus.schema import to_embed_text, to_index_text

    store = CorpusStore(corpus)
    rows, _ = flatten_corpus(store)
    row = rows[0]
    ch = row.to_corpus_chunk()

    idx = to_index_text(ch, row.title)
    emb = to_embed_text(ch, row.title)
    assert ch.text in idx and ch.text in emb
    for aux in (ch.summary, *ch.keywords, ch.qa[0].q):
        assert aux in idx, f"aux {aux!r} missing from BM25 text"
        assert aux not in emb, f"aux {aux!r} leaked into embed text"


def test_empty_corpus_flattens_to_nothing(tmp_path):
    store = CorpusStore(tmp_path / "empty")
    (tmp_path / "empty").mkdir()
    rows, pages = flatten_corpus(store)
    assert rows == [] and pages == 0


def test_page_with_no_chunks_contributes_no_rows(corpus):
    write_corpus(corpus, "Manual/Stub.html", [])
    rows, pages = flatten_corpus(CorpusStore(corpus))
    assert pages == 3
    assert all(r.source != "Manual/Stub.html" for r in rows)


def test_decode_rows_yields_typed_qa_not_dicts(corpus):
    """Regression: decoding without the type leaves qa as dicts, and the first
    to_index_text() then raises AttributeError on q.q. decode_rows must return
    real QA structs so the engine can index aux text."""
    import msgspec

    from rag.corpus.schema import to_index_text

    store = CorpusStore(corpus)
    rows, _ = flatten_corpus(store)
    blob = encode_rows(rows)

    decoded = decode_rows(blob)
    # the typed decode gives objects, not dicts
    assert isinstance(decoded[0].qa[0].q, str)
    assert to_index_text(decoded[0].to_corpus_chunk(), decoded[0].title)

    # and the naive path really would break (documents WHY decode_rows exists)
    naive = msgspec.msgpack.decode(blob)
    assert isinstance(naive[0]["qa"][0], dict), \
        "if msgspec ever types nested structs by default, this guard is obsolete"
