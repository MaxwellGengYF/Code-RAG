"""Index build: flatten the corpus -> chunk table -> BM25 + dense (dual-write).

Artefacts in ``index_dir``:
  chunks.msgpack    — the single chunk table (both indexes derive from it)
  bm25_word.pkl     — word-tokenizer BM25 over to_index_text + path terms
  vectors.f32       — raw (n, dim) float32 BGE-M3 embeddings of to_embed_text
  vector_meta.json  — dim/count/model/normalized
  manifest.json     — n_chunks, corpus gen_key, input sha1s (staleness guard)

``chunk_uid`` keys both index layers; doc ids in the BM25 index are positional
row ids, and dense row i is the same chunk — dual-write consistency by
construction from one table. Rebuilds are deterministic: same corpus in, same
uids, same rows.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import msgspec

from rag import resolve_path
from rag.corpus.schema import CorpusChunk, QA
from rag.store import CorpusStore, FileManager


class ChunkRow(msgspec.Struct, kw_only=True):
    """One row of the flat chunk table."""

    chunk_uid: str
    source: str
    title: str
    heading_path: list[str] = []
    text: str = ""
    summary: str = ""
    keywords: list[str] = []
    synonyms: list[str] = []
    qa: list[QA] = []

    def to_corpus_chunk(self) -> CorpusChunk:
        return CorpusChunk(
            chunk_uid=self.chunk_uid, heading_path=self.heading_path,
            text=self.text, summary=self.summary, keywords=self.keywords,
            synonyms=self.synonyms, qa=self.qa,
        )


def encode_rows(rows: list[ChunkRow]) -> bytes:
    """Serialize the chunk table."""
    return msgspec.msgpack.encode(rows)


def decode_rows(blob: bytes) -> list[ChunkRow]:
    """Deserialize the chunk table into real ChunkRow/QA structs.

    Decoding WITHOUT the type yields plain dicts for nested structs, so
    ``ChunkRow(**r)`` would leave ``qa`` as a list of dicts and the first
    ``to_index_text`` call would raise AttributeError on ``q.q``. Always decode
    with ``type=list[ChunkRow]``.
    """
    return msgspec.msgpack.decode(blob, type=list[ChunkRow])


def load_rows(path: str | Path) -> list[ChunkRow]:
    """Read a persisted chunk table."""
    return decode_rows(Path(path).read_bytes())


def flatten_corpus(store: CorpusStore) -> tuple[list[ChunkRow], int]:
    """Corpus files -> (deterministic flat chunk table, page count with corpus)."""
    rows: list[ChunkRow] = []
    n_pages = 0
    for rel, data in sorted(store.iterate_all(), key=lambda kv: kv[0]):
        n_pages += 1
        chunks = data.get("chunks") or []
        title = data.get("title", "")
        for ch in chunks:
            rows.append(ChunkRow(
                chunk_uid=ch.get("chunk_uid", ""),
                source=rel,
                title=title,
                heading_path=ch.get("heading_path") or [],
                text=ch.get("text", ""),
                summary=ch.get("summary", ""),
                keywords=ch.get("keywords") or [],
                synonyms=ch.get("synonyms") or [],
                qa=[QA(q=q.get("q", ""), a=q.get("a", "")) for q in ch.get("qa") or []],
            ))
    return rows, n_pages


def _sha1_8(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:8]


def build_indexes(cfg: dict, *, force: bool = False, skip_dense: bool = False,
                  verbose: bool = True) -> int:
    t0 = time.time()
    corpus_dir = resolve_path(cfg.get("corpus_dir", "corpus"))
    index_dir = resolve_path(cfg.get("index_dir", "index"))
    index_dir.mkdir(parents=True, exist_ok=True)
    store = CorpusStore(corpus_dir)

    fm = FileManager(root=resolve_path("."), dirs=cfg.get("dirs", []),
                     corpus_dir=corpus_dir)
    corpus_manifest = fm.load_manifest()
    gen_key = corpus_manifest.get("gen_key", "")
    n_corpus_files = len(corpus_manifest.get("files", {}))

    im_path = index_dir / "manifest.json"
    if im_path.exists() and not force:
        im = json.loads(im_path.read_text(encoding="utf-8"))
        if im.get("corpus_gen_key") != gen_key:
            print(f"[index] REFUSING to build: corpus gen_key changed "
                  f"({im.get('corpus_gen_key')} -> {gen_key}). The index would "
                  f"mix chunks from two generations. Re-run with --force after "
                  f"the corpus step completes.", file=sys.stderr)
            return 1
        # which artefacts must be present depends on whether dense is being built
        required = ["chunks.msgpack", "bm25_word.pkl"]
        if not skip_dense:
            required += ["vector_meta.json", "vectors.f32"]
        have_all = all((index_dir / f).exists() for f in required)
        if have_all:
            print(f"[index] up to date ({im.get('n_chunks')} chunks)", file=sys.stderr)
            return 0

    rows, n_corpus_pages = flatten_corpus(store)
    if not rows:
        print("[index] corpus is empty — run the corpus step first", file=sys.stderr)
        return 1
    print(f"[index] flattened {len(rows)} chunks from {n_corpus_pages} pages",
          file=sys.stderr)

    chunks = [r.to_corpus_chunk() for r in rows]
    sources = [r.source for r in rows]
    titles = [r.title for r in rows]

    # single chunk table
    chunks_bytes = encode_rows(rows)
    chunks_sha = hashlib.sha1(chunks_bytes).hexdigest()[:8]
    (index_dir / "chunks.msgpack").write_bytes(chunks_bytes)
    print(f"[index] wrote chunks.msgpack ({time.time() - t0:.0f}s)", file=sys.stderr)

    from rag.index.bm25_index import build_bm25
    # bm25_aux controls whether the synthetic aux fields (summary/keywords/
    # synonyms/qa.q) are added to the BM25 index text. Default OFF, measured:
    # at 26k pages aux=False MRR 0.881 vs aux=True 0.826 (base-24). Aux helps on
    # a small subset (0.906 vs 0.917 flipped at 3k pages) but every sibling page's
    # aux repeats the parent symbol, and aux lengthens documents (avgdl 124 ->
    # 169), so the dilution grows with corpus size. See eval_results.md.
    bm25_aux = bool(cfg.get("bm25_aux", False))
    index, _searcher = build_bm25(chunks, sources, titles,
                                  path_boost=cfg.get("path_boost", 3),
                                  aux=bm25_aux, verbose=verbose)
    index.save(str(index_dir / "bm25_word.pkl"))
    print(f"[index] wrote bm25_word.pkl ({time.time() - t0:.0f}s)", file=sys.stderr)

    embed_model = cfg.get("embed_model", "BAAI/bge-m3")
    vecs_path = index_dir / "vectors.f32"
    meta_path = index_dir / "vector_meta.json"
    prev_im = json.loads(im_path.read_text(encoding="utf-8")) if im_path.exists() else {}

    if not skip_dense and embed_model not in ("", "none"):
        # The embed text (title + heading_path + clean text) is a pure function of
        # the chunk table — it does NOT depend on the BM25-only knobs (path_boost,
        # aux). So when the chunk table is byte-identical to the last build and the
        # existing vectors describe exactly these rows with the same model, reuse
        # them instead of re-embedding (which costs ~1.5 h on CPU for 41k chunks).
        # This is what makes the path_boost/aux ablation loop fast.
        reusable = (
            prev_im.get("chunks_sha1_8") == chunks_sha
            and prev_im.get("embed_model") == embed_model
            and vecs_path.exists() and meta_path.exists()
        )
        if reusable:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            reusable = (int(meta.get("count", -1)) == len(rows)
                        and meta.get("model") == embed_model)
        if reusable:
            print(f"[index] reusing vectors.f32 — chunk table unchanged "
                  f"(sha1_8={chunks_sha}, {len(rows)} rows); dense does not depend "
                  f"on path_boost/aux", file=sys.stderr)
        else:
            from rag.index.vector_index import (build_vectors_resumable,
                                                ensure_embed_model, save_meta)
            m = ensure_embed_model(embed_model)
            dim = int(getattr(m, "get_embedding_dimension", None)()
                      if hasattr(m, "get_embedding_dimension")
                      else m.get_sentence_embedding_dimension())
            build_vectors_resumable(
                chunks, titles, out_path=vecs_path,
                model=embed_model, dim=dim,
                # rows belong to THIS chunk table; a partial file from a different
                # one must be discarded, not resumed (see build_vectors_resumable)
                stamp=chunks_sha,
                log=lambda msg: print(msg, file=sys.stderr))
            save_meta(meta_path, model=embed_model, dim=dim, count=len(chunks))
            print(f"[index] wrote vectors.f32 ({len(chunks)}, {dim}) "
                  f"({time.time() - t0:.0f}s)", file=sys.stderr)
    elif skip_dense:
        # A stale vectors.f32 from an earlier build would be positionally
        # misaligned with the freshly flattened rows (vectors carry no ids), so
        # remove it rather than let dense silently score the wrong chunks.
        for stale in ("vectors.f32", "vectors.f32.stamp",
                    "vectors.f32.progress", "vector_meta.json"):
            p = index_dir / stale
            if p.exists():
                p.unlink()
                print(f"[index] removed stale {stale} (dense skipped)",
                      file=sys.stderr)
        print("[index] dense skipped (--skip-dense)", file=sys.stderr)

    im = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_chunks": len(rows),
        "n_pages": n_corpus_pages,
        "corpus_gen_key": gen_key,
        "chunks_sha1_8": _sha1_8(index_dir / "chunks.msgpack"),
        "embed_model": embed_model,
        "bm25": {"tokenizer": "word", "path_boost": cfg.get("path_boost", 3),
                 "aux": bm25_aux},
        "wall_s": round(time.time() - t0, 1),
    }
    im_path.write_text(json.dumps(im, indent=2), encoding="utf-8")
    print(f"[index] manifest written; total {time.time() - t0:.0f}s", file=sys.stderr)
    return 0
