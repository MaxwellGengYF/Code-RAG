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

`unity_docs_downloader.py` (the crawler used for this mirror; not part of this
repo) — resumable, polite concurrent crawler (8 workers, random 0.15–0.35 s
delay, retries). Seed list from Manual/docdata/toc.js and
ScriptReference/docdata/toc.js; assets discovered by parsing HTML/CSS references.

RAG retrieval over the mirror

The mirror is indexed by an LLM-built RAG system that works over any document
set (AGENTS.md is the full operational manual). One JSON file is both the
provider config and the corpus config: copy config.example.json to
config.json next to your documents, set `dirs` (input directories) and the
provider keys, then:

    uv run python -m rag compile --config config.json --config llama_cpp/provider-qwen35-local.json
    uv run python -m rag search --query "Rigidbody.AddForce"
    uv run python -m rag status     # coverage, pending pages, est. cost, engine
    uv run python -m rag.eval.eval_rag --gold-set base --mode bm25   # the quality gate

`uv sync` installs the default set: corpus generation over remote providers
plus the BM25 engine. Dense/hybrid search, the cross-encoder reranker and the
Ollama embedder are local inference and live in the opt-in `local` extra
(multi-GB torch wheels), so a plain `uv sync` / `uv run` never installs them:
use `uv sync --extra local`, or prefix a command with `uv run --extra local`.

With no --config, ./config.json in the current working directory is used.
Repeat --config to shard pages across several providers (one config per
DISTINCT quota pool; settings merge with the first file winning). Relative
dirs/corpus_dir/index_dir anchor at the config file's directory, so a config
plus its documents form a relocatable unit.

compile sends each page (as markdown) through an LLM to produce semantic
chunks plus retrieval aux fields (summary/keywords/synonyms/QA), then builds a
word-tokenizer BM25 index over title + heading path + chunk text (aux text
excluded by measured decision) and a BGE-M3 dense index (clean text only).
Search defaults to BM25 ("mode": "bm25" in config.json); dense and RRF-fused
hybrid remain available via --mode dense / --mode hybrid. Final measured
quality on this mirror (2026-09-20, full corpus, independent base-24 gold
set): bm25 MRR 0.880 / hit@1 0.833 / hit@10 0.958 — gate PASS. Full decision
record: eval_results.md. Regeneration is md5-incremental: only added/changed
pages re-hit the LLM, and a prompt/model/version bump rekeys every page.
Generated artefacts live in corpus/ and index/ (gitignored).

A local provider needs no gateway: llama_cpp/ ships prebuilt llama.cpp
binaries plus provider-qwen35-local.json driving a Qwen3.5-9B GGUF from
models/ (~100 tok/s on an RTX 4080 SUPER). Details: llama_cpp/USAGE.md and
the "Local inference" section of AGENTS.md.

Two flags matter for a first full build over ~44k pages:

- --no-thinking — reasoning gateways otherwise apply server-side thinking and
  cost ~8k output tokens and ~100 s per page instead of ~300 tokens / ~15 s.
- repeat --config once per DISTINCT quota pool (a pool is host + api_key) to
  parallelise. Two configs sharing one key add no redundancy, and if that
  pool's quota runs out the run pauses and resumes later rather than degrading
  pages.

The build is checkpointed and resumable per PAGE: the compile unit is one page,
so one page = one LLM session = one corpus file = one progress line = one
incremental manifest flush. Interrupt it at any point and rerun the same command
to continue — a kill costs at most the pages the workers had in flight, never a
25/50-page batch of already-paid-for LLM work. search --mentions TERM enumerates
literal
occurrences, --explain shows per-term document frequencies and the fusion
breakdown, and repl keeps a persistent JSONL session so the index loads once
for many queries.

