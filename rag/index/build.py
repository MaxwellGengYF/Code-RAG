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
        have_all = all((index_dir / f).exists() for f in (
            "chunks.msgpack", "bm25_word.pkl", "vector_meta.json",
            "vectors.f32"))
        if have_all and not skip_dense:
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
    (index_dir / "chunks.msgpack").write_bytes(
        msgspec.msgpack.encode([msgspec.to_builtins(r) for r in rows]))
    print(f"[index] wrote chunks.msgpack ({time.time() - t0:.0f}s)", file=sys.stderr)

    from rag.index.bm25_index import build_bm25
    index, _searcher = build_bm25(chunks, sources, titles,
                                  path_boost=cfg.get("path_boost", 3),
                                  verbose=verbose)
    index.save(str(index_dir / "bm25_word.pkl"))
    print(f"[index] wrote bm25_word.pkl ({time.time() - t0:.0f}s)", file=sys.stderr)

    embed_model = cfg.get("embed_model", "BAAI/bge-m3")
    if not skip_dense and embed_model not in ("", "none"):
        from rag.index.vector_index import build_vectors, save_vectors, save_meta
        vecs = build_vectors(chunks, titles, model=embed_model)
        save_vectors(vecs, index_dir / "vectors.f32")
        save_meta(index_dir / "vector_meta.json", model=embed_model,
                  dim=int(vecs.shape[1]), count=int(vecs.shape[0]))
        print(f"[index] wrote vectors.f32 {vecs.shape} ({time.time() - t0:.0f}s)",
              file=sys.stderr)
    elif skip_dense:
        print("[index] dense skipped (--skip-dense)", file=sys.stderr)

    im = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_chunks": len(rows),
        "n_pages": n_corpus_pages,
        "corpus_gen_key": gen_key,
        "chunks_sha1_8": _sha1_8(index_dir / "chunks.msgpack"),
        "embed_model": embed_model,
        "bm25": {"tokenizer": "word", "path_boost": cfg.get("path_boost", 3),
                 "aux": True},
        "wall_s": round(time.time() - t0, 1),
    }
    im_path.write_text(json.dumps(im, indent=2), encoding="utf-8")
    print(f"[index] manifest written; total {time.time() - t0:.0f}s", file=sys.stderr)
    return 0
