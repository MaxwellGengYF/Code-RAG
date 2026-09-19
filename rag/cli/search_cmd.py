"""The ``search`` command: parity with the legacy hybrid_retrieve.py features."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from rag.compile import load_rag_config


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


def run_search(
    *,
    queries: list[str] | None = None,
    query_file: str | None = None,
    k: int | None = None,
    mode: str | None = None,
    mentions: str | None = None,
    explain: bool = False,
    no_rerank: bool = False,
    text: bool = False,
    legacy: bool = False,
    config_path: str = "rag_config.json",
) -> int:
    cfg = load_rag_config(config_path)
    k = k if k is not None else int(cfg.get("final_k", 8))
    mode = mode or cfg.get("mode", "hybrid")

    if legacy:
        # Preserve the old engine (and its legacy artefacts) behind an explicit flag.
        import hybrid_retrieve
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
        if text:
            argv.append("--text")
        return hybrid_retrieve.main(argv)

    from rag.search.engine import SearchEngine
    engine = SearchEngine(cfg)
    try:
        engine.load()
    except FileNotFoundError as exc:
        # No index yet (fresh checkout, or corpus step not run). Report the exact
        # next command instead of a traceback — this is the expected state after
        # `git clone`, not an internal error.
        print(str(exc), file=sys.stderr)
        print("\nnothing has been built in this checkout yet. To build:", file=sys.stderr)
        print("  1. uv run python rag.py compile --provider D:/qwen_flash.json "
              "--no-thinking   # LLM corpus (slow, resumable)", file=sys.stderr)
        print("  2. uv run python rag.py compile --steps index                "
              "                  # BM25 + dense indexes", file=sys.stderr)
        print("see AGENTS.md for the full workflow.", file=sys.stderr)
        return 1

    if mentions:
        res = engine.mentions(mentions)
        if not res:
            print(f"[mentions] 0 files contain {mentions!r} (literal, case-insensitive)",
                  file=sys.stderr)
        out = {mentions: res}
        if text:
            print(f"# {mentions}: {len(res)} files")
            for r in res:
                print(f"  {r['count']:>4}  {r['source']}")
        else:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    qs = _read_queries(queries, query_file)
    if not qs:
        print("nothing to search: pass --query or --query-file", file=sys.stderr)
        return 1

    t0 = time.time()
    results = [engine.search(q, k=k, mode=mode, explain=explain,
                             no_rerank=no_rerank) for q in qs]
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
    if text:
        for r in results:
            print(f"\n### {r['query']}")
            if r.get("explain"):
                known = [t["term"] for t in r["explain"] if t["in_index"]]
                print(f"  terms in index: {len(known)}/{len(r['explain'])} {known}")
            for h in r["hits"]:
                print(f"  {h['fused_score']:<9.4f} {h['title']}  ({h['source']})")
                print(f"            {h['text'][:160]}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0
