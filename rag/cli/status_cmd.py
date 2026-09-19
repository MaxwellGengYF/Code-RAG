"""The ``status`` command: corpus/index freshness + counts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rag import resolve_path
from rag.compile import load_rag_config


def run_status(*, config_path: str = "rag_config.json") -> int:
    cfg = load_rag_config(config_path)
    corpus_dir = resolve_path(cfg.get("corpus_dir", "corpus"))
    index_dir = resolve_path(cfg.get("index_dir", "index"))

    from rag.store import CorpusStore, FileManager
    fm = FileManager(root=resolve_path("."), dirs=cfg.get("dirs", []),
                     corpus_dir=corpus_dir)
    scanned = fm.scan()
    diff = fm.diff(scanned)
    store = CorpusStore(corpus_dir)
    n_corpus_files = sum(1 for _ in store.iterate_all())
    manifest = fm.load_manifest()

    print("== rag status ==")
    print(f"doc mirror : {len(scanned)} html pages")
    print(f"corpus     : {n_corpus_files} .rag.json files under {corpus_dir}")
    print(f"manifest   : gen_key={manifest.get('gen_key', '-')!r} "
          f"({len(manifest.get('files', {}))} recorded)")
    print(f"diff       : added={len(diff.added)} changed={len(diff.changed)} "
          f"removed={len(diff.removed)} unchanged={len(diff.unchanged)}")
    failures = corpus_dir / "failures.jsonl"
    if failures.exists():
        n_fail = sum(1 for _ in open(failures, encoding="utf-8"))
        print(f"failures   : {n_fail} recorded in corpus/failures.jsonl")

    im_path = index_dir / "manifest.json"
    if im_path.exists():
        im = json.loads(im_path.read_text(encoding="utf-8"))
        fresh = im.get("corpus_gen_key") == manifest.get("gen_key")
        print(f"index      : {im.get('n_chunks')} chunks / {im.get('n_pages')} pages, "
              f"built {im.get('built_at')} ({im.get('wall_s')}s)")
        print(f"             corpus gen_key match: {'YES' if fresh else 'STALE — rerun compile --steps index'}")
        print(f"             embed model: {im.get('embed_model')}")
        for f in ("chunks.msgpack", "bm25_word.pkl", "vectors.f32", "vector_meta.json"):
            p = index_dir / f
            if p.exists():
                print(f"             {f}: {p.stat().st_size / 1e6:.1f} MB")
    else:
        print("index      : NOT BUILT — run: uv run python rag.py compile --steps index")
    return 0
