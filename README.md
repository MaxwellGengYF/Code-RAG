# Unity Manual & Script API Reference (offline mirror)

Downloaded from https://docs.unity3d.com (current docs, Unity 6.x era) on 2026-08-28.

## Contents

- `Manual/` — Unity Manual (HTML). Official TOC has 3,550 pages; 3,546 saved.
- `ScriptReference/` — Unity Scripting API Reference (HTML). Official TOC has 4,774
  pages; 4,773 saved, plus ~40k additional API member pages discovered via links.
- `StaticFilesManual/`, `StaticFilesScriptReference/`, `uploads/` — CSS, JS, images,
  fonts and other assets referenced by the pages (same-host only).

Total: ~46,200 files, ~1.03 GB.

## Notes

- Open `Manual/index.html` or `ScriptReference/index.html` in a browser to browse.
  Relative links between pages work offline because the URL structure is mirrored.
- A few TOC pages are dead links on Unity's server (redirect loops / 404s) and were
  not saved; they are listed in `.crawler_state.json` as `missing`.
- External links (cdn.cookielaw.org, developer.meta.com, unity.com, other-language
  mirrors `/cn/`, `/ja/`, `/kr/`, etc.) were intentionally not downloaded.
- `.crawler_state.json` is the crawler's resume state (which URLs are done/missing);
  it can be deleted if not needed.

How it was downloaded

D:/unity_docs_downloader.py — resumable, polite concurrent crawler (8 workers,
random 0.15–0.35 s delay, retries). Seed list from Manual/docdata/toc.js and
ScriptReference/docdata/toc.js; assets discovered by parsing HTML/CSS references.

RAG retrieval over the mirror

The mirror is indexed by an LLM-built RAG system (AGENTS.md is the full
operational manual):

    uv run python rag.py compile --provider D:/qwen_flash.json --no-thinking
    uv run python rag.py search --query "Rigidbody.AddForce"
    uv run python rag.py status     # coverage, pending pages, est. cost, engine
    uv run python eval_rag.py --gold-set base --mode bm25   # the quality gate

compile sends each page (as dumpdoc-style markdown) through an LLM to produce
semantic chunks plus retrieval aux fields (summary/keywords/synonyms/QA), then
builds a word-tokenizer BM25 index over title + heading path + chunk text (aux
text excluded by measured decision) and a BGE-M3 dense index (clean text
only). Search defaults to BM25 (`"mode": "bm25"` in rag_config.json); dense
and RRF-fused hybrid remain available via `--mode dense` / `--mode hybrid` —
at full corpus scale hybrid measurably loses MRR to dense drift inside
near-duplicate API sibling families, while staying useful as a typo-tolerance
escape hatch. Final measured quality (2026-09-20, full corpus, independent
base-24 gold set): bm25 MRR 0.880 / hit@1 0.833 / hit@10 0.958 — gate PASS.
Full decision record: eval_results.md. Regeneration is md5-incremental:
only added/changed pages re-hit the LLM, and a prompt/model/version bump
rekeys every page. Generated artefacts live in corpus/ and index/
(gitignored).

A local provider needs no gateway: llama_cpp/ ships prebuilt llama.cpp
binaries plus provider-qwen35-local.json driving a Qwen3.5-9B GGUF from
models/ (~100 tok/s on an RTX 4080 SUPER) — pass it as
`--provider llama_cpp/provider-qwen35-local.json`. Details: llama_cpp/USAGE.md
and the "Local inference" section of AGENTS.md.

Two flags matter for a first full build over ~44k pages:

- `--no-thinking` — reasoning gateways otherwise apply server-side thinking and
  cost ~8k output tokens and ~100 s per page instead of ~300 tokens / ~15 s.
- repeat `--provider` once per DISTINCT quota pool (a pool is host + api_key) to
  parallelise. Two configs sharing one key add no redundancy, and if that pool's
  quota runs out the run pauses and resumes later rather than degrading pages.

The build is checkpointed and resumable: interrupt it at any point and rerun the
same command to continue. `rag.py search --mentions TERM` enumerates literal
occurrences, `--explain` shows per-term document frequencies and the fusion
breakdown, and `rag.py repl` keeps a persistent JSONL session so the index loads
once for many queries.

