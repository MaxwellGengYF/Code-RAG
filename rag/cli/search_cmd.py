"""The ``search`` command: parity with the legacy hybrid_retrieve.py features."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from rag.config import load_settings
from rag.search.format import mentions_to_markdown, results_to_markdown


def _read_queries(queries: list[str] | None, query_file: str | None) -> list[str]:
    out: list[str] = []
    for q in queries or []:
        if q and q.strip():
            out.append(q.strip())
    if query_file:
        for ln in Path(query_file).read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                out.append(ln)
    return out


def _server_payload(cfg, *, qs, mentions, mentions_limit, mentions_context,
                    k, mode, explain, no_rerank, snippet_width) -> dict | None:
    """One attempt at the rag.server HTTP path; None when it is unavailable.

    Raises RuntimeError on genuine server errors (HTTP 4xx/5xx other than 503).
    """
    from rag import server as srv
    t0 = time.time()
    if mentions:
        return srv.server_request(cfg, "/mentions",
                                  {"term": mentions, "limit": mentions_limit,
                                   "context": mentions_context})
    results: list[dict] = []
    merged: dict = {}
    for q in qs:
        resp = srv.server_request(cfg, "/search",
                                  {"query": q, "k": k, "mode": mode,
                                   "explain": explain, "no_rerank": no_rerank,
                                   "snippet_width": snippet_width})
        merged.update(resp.pop("_meta", None) or {})
        results.append(resp)
    return {
        "_meta": {
            "engine": "rag.server",
            "mode": mode,
            "elapsed_s": round(time.time() - t0, 3),
            "chunks": merged.get("chunks"),
            "dense": merged.get("dense"),
        },
        "results": results,
    }


def _emit(payload: dict, *, term: str | None, out: str | None,
          text: bool, as_json: bool) -> None:
    """Shared output sink. --out always gets raw JSON; stdout is raw JSON
    (--json), compact text (--text), or markdown (default). *term* set means
    a mentions payload ({term: rows}), otherwise a search payload."""
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    if out:
        Path(out).write_text(body, encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    if text:
        if term is not None:
            rows = payload[term]
            print(f"# {term}: {len(rows)} files")
            for r in rows:
                line = f"  {r['count']:>4}  {r['source']}"
                if r.get("context"):
                    line += f"\n        …{r['context']}…"
                print(line)
        else:
            for r in payload["results"]:
                print(f"\n### {r['query']}")
                if r.get("explain"):
                    known = [t["term"] for t in r["explain"] if t["in_index"]]
                    print(f"  terms in index: {len(known)}/{len(r['explain'])} {known}")
                for h in r["hits"]:
                    print(f"  {h['fused_score']:<9.4f} {h['title']}  ({h['source']})")
                    print(f"            {h['text'][:160]}")
    elif as_json:
        print(body)
    elif term is not None:
        print(mentions_to_markdown(term, payload[term]))
    else:
        print(results_to_markdown(payload))


def run_search(
    *,
    queries: list[str] | None = None,
    query_file: str | None = None,
    k: int | None = None,
    mode: str | None = None,
    mentions: str | None = None,
    mentions_limit: int = 60,
    mentions_context: int = 0,
    explain: bool = False,
    no_rerank: bool = False,
    text: bool = False,
    legacy: bool = False,
    as_json: bool = False,
    out: str | None = None,
    snippet_width: int = 500,
    config_path: str | None = None,
) -> int:
    cfg = load_settings(config_path)
    k = k if k is not None else int(cfg.get("final_k", 8))
    mode = mode or cfg.get("mode", "hybrid")

    if legacy:
        # Preserve the old engine (and its legacy artefacts) behind an explicit flag.
        from rag.legacy import hybrid_retrieve
        argv: list[str] = []
        for q in queries or []:
            argv += ["--query", q]
        if query_file:
            argv += ["--query-file", query_file]
        argv += ["--final-k", str(k)]
        if explain:
            argv.append("--explain")
        if mentions:
            argv += ["--mentions", mentions]
        if mentions_context:
            argv += ["--mentions-context", str(mentions_context)]
        if mentions_limit != 60:
            argv += ["--mentions-limit", str(mentions_limit)]
        if snippet_width != 500:
            argv += ["--snippet-width", str(snippet_width)]
        if text:
            argv.append("--text")
        return hybrid_retrieve.main(argv)

    qs = _read_queries(queries, query_file)

    # Server-first: try the HTTP server once before paying the local engine-load
    # cost; set RAG_NO_SERVER=1 to force the local path.
    payload: dict | None = None
    if (mentions or qs) and not os.environ.get("RAG_NO_SERVER"):
        from rag import server as srv
        try:
            payload = _server_payload(cfg, qs=qs, mentions=mentions,
                                      mentions_limit=mentions_limit,
                                      mentions_context=mentions_context,
                                      k=k, mode=mode, explain=explain,
                                      no_rerank=no_rerank,
                                      snippet_width=snippet_width)
        except srv.ServerUnavailable:
            print("[search] server unavailable; local fallback", file=sys.stderr)
            payload = None
    if payload is not None:
        if mentions and not payload.get(mentions):
            print(f"[mentions] 0 files contain {mentions!r} "
                  "(literal, case-insensitive)", file=sys.stderr)
        _emit(payload, term=mentions, out=out, text=text, as_json=as_json)
        return 0

    from rag.search.engine import SearchEngine
    engine = SearchEngine(cfg)
    try:
        engine.load()
    except FileNotFoundError as exc:
        # No index yet (fresh checkout, or corpus step not run). Report the exact
        # next command instead of a traceback — this is the expected state after
        # `git clone`, not an internal error.
        print(str(exc), file=sys.stderr)
        print("build: python -m rag compile --steps index", file=sys.stderr)
        return 1

    if mentions:
        res = engine.mentions(mentions, limit=mentions_limit,
                              context=mentions_context)
        if not res:
            print(f"[mentions] 0 files contain {mentions!r} "
                  "(literal, case-insensitive)", file=sys.stderr)
        _emit({mentions: res}, term=mentions, out=out, text=text,
              as_json=as_json)
        return 0

    if not qs:
        print("nothing to search: pass --query or --query-file", file=sys.stderr)
        return 1

    t0 = time.time()
    results = [engine.search(q, k=k, mode=mode, explain=explain,
                             no_rerank=no_rerank,
                             snippet_width=snippet_width) for q in qs]
    payload = {
        "_meta": {
            "engine": "rag.search",
            "mode": mode,
            "chunks": engine.n_chunks,
            "dense": engine.has_dense,
            "elapsed_s": round(time.time() - t0, 3),
        },
        "results": results,
    }
    _emit(payload, term=None, out=out, text=text, as_json=as_json)
    return 0
