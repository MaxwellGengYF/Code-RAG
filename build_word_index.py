"""Build an identifier-word BM25 index over the Unity docs (complements index.pkl).

    uv run python build_word_index.py                # full build -> index_word.pkl
    uv run python build_word_index.py --max-files 2000   # quick smoke test

Writes index_word.pkl + reuses the existing chunks.pkl (no re-parse needed).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

from tqdm import tqdm

from hybrid_retrieve import Chunk, doc_tokens, load_chunks_pickle
from retrieval import InvertedIndex, Searcher
from unity_tokenizer import WordTokenizer, path_terms


def chunk_tokens(text, source, wt, path_boost):
    """Deprecated shim: tokenization now lives in hybrid_retrieve.doc_tokens."""
    from hybrid_retrieve import Chunk as _C
    return doc_tokens(_C(0, source, text), wt, path_boost)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="retriever_config.json")
    ap.add_argument("--chunks-path", default=None, help="reuse an existing chunks.pkl")
    ap.add_argument("--out-index", default="index_word.pkl")
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--rebuild-chunks", action="store_true")
    ap.add_argument("--path-boost", type=int, default=2,
                    help="repeat path/title field terms N times (0 disables). Default 2.")
    ap.add_argument("--build-args", default="", help="extra args for chunk build, e.g. '--max-files 3000'")
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    chunks_path = a.chunks_path or cfg["chunks_path"]

    if a.rebuild_chunks:
        from hybrid_retrieve import load_chunks
        chunks = load_chunks(cfg["dirs"])
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)
    else:
        if not Path(chunks_path).exists():
            print(f"missing {chunks_path}; run hybrid_retrieve.py --build first", file=sys.stderr)
            return 1
        print(f"loading {chunks_path} ...", file=sys.stderr, flush=True)
        chunks: list[Chunk] = load_chunks_pickle(chunks_path)

    if a.max_chunks:
        chunks = chunks[: a.max_chunks]
    print(f"{len(chunks)} chunks; tokenizing with WordTokenizer", file=sys.stderr, flush=True)

    tok = WordTokenizer()
    index = InvertedIndex()
    t0 = time.time()
    for c in tqdm(chunks, desc="word-index"):
        index.add_document(c.chunk_id, doc_tokens(c, tok, a.path_boost))
    index.finalize()
    print(f"built in {time.time()-t0:.1f}s; unique terms={len(index.terms())}", file=sys.stderr)
    Path("index_word.meta.json").write_text(
        json.dumps({"tokenizer": "word", "path_boost": a.path_boost,
                    "n_chunks": len(chunks)}, indent=2), encoding="utf-8")

    index.save(a.out_index)
    print(f"saved {a.out_index} ({Path(a.out_index).stat().st_size/1e6:.1f} MB)", file=sys.stderr)

    searcher = Searcher(index, tokenizer=tok, fuzziness=0, min_should_match=0.6)
    for q in ["MaterialPropertyBlock", "SRP Batcher MaterialPropertyBlock compatibility"]:
        top = searcher.search(q, top_k=5)
        print(f"\n{q}\n  " + "\n  ".join(
            f"{i:>7.3f} {Path(chunks[d].source).as_posix()[:70]}" for d, i in top), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
