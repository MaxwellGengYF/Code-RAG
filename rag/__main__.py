"""python -m rag — LLM-built RAG corpus + hybrid retrieval over any document set.

Commands:
    python -m rag compile [--config cfg.json ...] [--steps corpus,index,deps]
             [--workers 8] [--max-files N] [--force] [--regen] [--clean]
             [--only REL] [--dry-run]
    python -m rag search --query "..." [--k 10] [--mode hybrid|bm25|dense]
             [--query-file q.txt] [--mentions TERM] [--explain] [--no-rerank]
    python -m rag status
    python -m rag repl
    python -m rag audit-corpus

The config JSON is BOTH the provider config and the corpus config: provider
keys (model/type/url/api_key/...), input/output locations (dirs, corpus_dir,
index_dir) and engine tuning live in one file. With no --config, ./config.json
in the current working directory is used when present.
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rag", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_compile = sub.add_parser("compile", help="LLM corpus -> indexes -> deps")
    p_compile.add_argument("--config", action="append", default=None,
                           metavar="CFG.json",
                           help="Combined provider+corpus config JSON (repeat to "
                                "shard pages across several providers; default: "
                                "./config.json when present)")
    p_compile.add_argument("--steps", default="deps,corpus,index",
                           help="Comma list: deps,corpus,index (default: all)")
    p_compile.add_argument("--workers", type=int, default=0,
                           help="Concurrent LLM calls (default: compile_workers in config)")
    p_compile.add_argument("--max-files", type=int, default=None)
    p_compile.add_argument("--force", action="store_true",
                           help="Regenerate ALL pages regardless of md5 state")
    p_compile.add_argument("--regen", action="store_true",
                           help="Ignore the manifest and regenerate everything")
    p_compile.add_argument("--clean", action="store_true",
                           help="Delete ALL generated artefacts first (corpus "
                                "pages + manifest + indexes), then recompile "
                                "every page from scratch. Unlike --force — "
                                "which requeues all pages but keeps the old "
                                "corpus until each page is overwritten — "
                                "nothing generated survives. Refuses "
                                "--dry-run/--only/--max-files; run only when "
                                "no compile is active")
    p_compile.add_argument("--only", default=None,
                           help="Regenerate exactly one page (root-relative path)")
    p_compile.add_argument("--dry-run", action="store_true",
                           help="Print page counts + token/cost estimate, no LLM calls")
    p_compile.add_argument("--no-thinking", action="store_true",
                           help="Disable model thinking for corpus generation "
                                "(~3x faster per page; recommended for big builds)")
    p_compile.add_argument("--price-in", type=float, default=None,
                           help="Input price per 1M tokens (for cost estimate)")
    p_compile.add_argument("--price-out", type=float, default=None,
                           help="Output price per 1M tokens (for cost estimate)")
    p_compile.add_argument("--install-embed-model", action="store_true",
        help="Download/cache the dense embed model (BGE-M3) and "
                                  "exit; needs no provider. Same as --steps deps but "
                                  "skips the pip-import checks. Needs the optional "
                                  "local extra: uv run --extra local ... "
                                  "--install-embed-model")
    p_compile.add_argument("--skip-dense", action="store_true",
                           help="With --steps index: build BM25 only (minutes "
                                "instead of hours); dense stays absent so hybrid "
                                "search degrades to BM25")

    p_search = sub.add_parser("search", help="query -> ranked results")
    p_search.add_argument("--query", action="append", default=None)
    p_search.add_argument("--query-file", default=None)
    p_search.add_argument("--k", type=int, default=None)
    p_search.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default=None)
    p_search.add_argument("--mentions", default=None, metavar="TERM")
    p_search.add_argument("--mentions-limit", type=int, default=60,
                          help="With --mentions: max files to list (default 60)")
    p_search.add_argument("--mentions-context", type=int, default=0,
                          help="With --mentions: capture N chars of surrounding "
                               "text per file")
    p_search.add_argument("--explain", action="store_true")
    p_search.add_argument("--out", default=None,
                          help="Write JSON results to this file")
    p_search.add_argument("--snippet-width", type=int, default=500,
                          help="Max chars of each hit snippet")
    p_search.add_argument("--no-rerank", action="store_true")
    p_search.add_argument("--text", action="store_true", help="Compact human output")
    p_search.add_argument("--legacy", action="store_true",
                          help="Use the legacy engine (rag.legacy.hybrid_retrieve)")
    p_search.add_argument("--config", default=None, metavar="CFG.json",
                          help="Config JSON (default: ./config.json when present)")

    p_repl = sub.add_parser(
        "repl", help="persistent JSONL session (query/mentions/read), index loads once")
    p_repl.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default=None)
    p_repl.add_argument("--config", default=None, metavar="CFG.json",
                        help="Config JSON (default: ./config.json when present)")

    p_status = sub.add_parser("status", help="corpus/index freshness + counts")
    p_status.add_argument("--config", default=None, metavar="CFG.json",
                          help="Config JSON (default: ./config.json when present)")

    p_audit = sub.add_parser(
        "audit-corpus",
        help="repair manifest entries claiming pages with no corpus file "
             "(run only when no compile is active)")
    p_audit.add_argument("--config", default=None, metavar="CFG.json",
                         help="Config JSON (default: ./config.json when present)")

    args = parser.parse_args(argv)
    if args.command == "compile":
        from rag.compile import run_compile
        return run_compile(
            configs=args.config,
            steps=[s.strip() for s in args.steps.split(",") if s.strip()],
            workers=args.workers, max_files=args.max_files, force=args.force,
            regen=args.regen, only=args.only, dry_run=args.dry_run,
            no_thinking=args.no_thinking, price_in=args.price_in,
            price_out=args.price_out, skip_dense=args.skip_dense,
            install_embed_model=args.install_embed_model, clean=args.clean,
        )
    if args.command == "search":
        from rag.cli.search_cmd import run_search
        return run_search(
            queries=args.query, query_file=args.query_file, k=args.k,
            mode=args.mode, mentions=args.mentions,
            mentions_limit=args.mentions_limit,
            mentions_context=args.mentions_context, explain=args.explain,
            no_rerank=args.no_rerank, text=args.text, legacy=args.legacy,
            out=args.out, snippet_width=args.snippet_width,
            config_path=args.config,
        )
    if args.command == "repl":
        from rag.cli.repl_cmd import repl_loop
        return repl_loop(config_path=args.config, mode=args.mode)
    if args.command == "status":
        from rag.cli.status_cmd import run_status
        return run_status(config_path=args.config)
    if args.command == "audit-corpus":
        from rag.cli.status_cmd import run_audit_corpus
        return run_audit_corpus(config_path=args.config)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
