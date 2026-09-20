"""RAG-backed ``--repl``: persistent JSONL session over the new engine.

Mirrors the legacy ``hybrid_retrieve.py --repl`` protocol so existing clients keep
working after the engine switch:

  {"query": "...", "k": 5, "mode": "hybrid", "explain": true,
   "dump": 1, "context": 1, "snippet_width": 500, "no_rerank": false}
  {"mentions": "TERM", "limit": 60, "context": 80, "dirs": ["Manual"]}
  {"read": "ScriptReference/Rigidbody.AddForce.html", "max_chars": 20000}

One JSON object per stdin line, one JSON response per stdout line, ``{"error": ...}``
for bad requests. The session stays alive across per-request failures and the index
loads once, which is the whole point of the REPL (a cold index load costs seconds).

``dump`` inlines full page markdown via dumpdoc; ``context`` inlines the neighbour
chunks of the same page around each hit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rag import ROOT
from rag.config import config_dir, load_settings


def _read_page(source: str, max_chars: int | None, base: Path | None = None) -> dict:
    """Full markdown for one doc page (the 'read_more' action)."""
    from rag.legacy import dumpdoc

    path = (base or ROOT) / source
    if not path.exists():
        return {"source": source, "error": f"not found: {path}"}
    md = dumpdoc.dump(str(path), heading=False)
    out = {"source": source, "chars": len(md)}
    if max_chars and len(md) > max_chars:
        out["markdown"] = md[:max_chars]
        out["truncated"] = True
    else:
        out["markdown"] = md
    return out


def _build_page_index(engine) -> dict[str, list]:
    """source -> rows in document order. Built once per session: _attach_context
    would otherwise rescan every chunk for every hit (O(n_rows) per hit)."""
    pages: dict[str, list] = {}
    for row in engine.rows:
        pages.setdefault(row.source, []).append(row)
    return pages


def _attach_context(pages: dict[str, list], hit: dict, n: int,
                    snippet_width: int) -> list[dict]:
    """Neighbour chunks of the same page around *hit*, in document order."""
    from rag.corpus.schema import snippet as make_snippet

    rows = pages.get(hit["source"], [])
    pos = next((i for i, r in enumerate(rows)
                if r.chunk_uid == hit["chunk_uid"]), None)
    if pos is None:
        return []
    lo, hi = max(0, pos - n), min(len(rows), pos + n + 1)
    out = []
    for i in range(lo, hi):
        if i == pos:
            continue
        row = rows[i]
        text, _trunc = make_snippet(row.to_corpus_chunk(), None,
                                    width=snippet_width)
        out.append({"chunk_uid": row.chunk_uid, "chunk_index": i,
                    "context_of": hit["chunk_uid"], "text": text})
    return out


def repl_loop(*, config_path: str | None = None,
              mode: str | None = None) -> int:
    from rag.search.engine import SearchEngine

    cfg = load_settings(config_path)
    if mode:
        cfg["mode"] = mode
    engine = SearchEngine(cfg)
    try:
        engine.load()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    base = config_dir(cfg)
    print(f"[repl] ready (rag engine) — {engine.n_chunks} chunks, "
          f"dense={engine.has_dense}; JSON requests on stdin, one JSON response "
          f"per line", file=sys.stderr, flush=True)
    pages = _build_page_index(engine)

    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"error": f"invalid JSON: {exc}"}), flush=True)
            continue
        try:
            if "query" in req:
                resp = engine.search(
                    str(req["query"]),
                    k=int(req.get("k", req.get("final_k", 8))),
                    mode=str(req.get("mode", cfg.get("mode", "hybrid"))),
                    per_file=req.get("per_file", 1),
                    explain=bool(req.get("explain", False)),
                    no_rerank=bool(req.get("no_rerank", False)),
                    snippet_width=int(req.get("snippet_width", 500)),
                )
                n_ctx = int(req.get("context", 0))
                if n_ctx:
                    for hit in resp["hits"]:
                        hit["context"] = _attach_context(
                            pages, hit, n_ctx,
                            int(req.get("snippet_width", 500)))
                n_dump = int(req.get("dump", 0))
                if n_dump:
                    seen: set[str] = set()
                    for hit in resp["hits"]:
                        if len(seen) >= n_dump:
                            break
                        src = hit["source"]
                        if src in seen:
                            continue
                        seen.add(src)
                        try:
                            hit["page_markdown"] = _read_page(src, None, base)["markdown"]
                        except Exception as exc:
                            hit["page_markdown_error"] = str(exc)
                resp["_meta"] = {"schema": 2, "engine": "rag.search"}
            elif "mentions" in req:
                dirs = req.get("dirs")
                resp = {str(req["mentions"]): engine.mentions(
                    str(req["mentions"]),
                    limit=int(req.get("limit", 60)),
                    context=int(req.get("context", 0)),
                    dirs=tuple(str(d) for d in dirs) if dirs else None)}
            elif "read" in req:
                resp = _read_page(str(req["read"]), req.get("max_chars"), base)
            else:
                resp = {"error": "request must contain one of: query, mentions, read"}
        except Exception as exc:  # keep the session alive on per-request failures
            resp = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(resp, ensure_ascii=False), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None, metavar="CFG.json")
    ap.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default=None)
    args = ap.parse_args(argv)
    return repl_loop(config_path=args.config, mode=args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
