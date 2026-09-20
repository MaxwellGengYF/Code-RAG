# Agent Usage: doc-rag (LLM-built corpus + hybrid retrieval over any document set)

A two-command RAG system that compiles and searches ANY set of HTML
documents — this checkout hosts the Unity 6.x offline-mirror deployment
(`Manual/` + `ScriptReference/`, 43,938 pages). The same commands work on any
other document set: point `dirs` in the config at your HTML and go.

- `compile` — LLM-generated corpus (semantic chunks + aux fields per page) →
  BM25 + dense indexes. md5-incremental, checkpointed, resumable.
- `search` — BM25 (word tokenizer + path terms) by default; BGE-M3 dense and
  RRF-fused hybrid selectable via `--mode`.

## Project structure

- `Manual/`, `ScriptReference/` — this deployment's input: the offline Unity
  6.x docs mirror (~44k HTML pages). Read-only input; any other HTML tree works.
- `python -m rag` — CLI entry point (rag/__main__.py): `compile` / `search` /
  `repl` / `status` / `audit-corpus`.
- `rag/` — the RAG package (ALL Python code lives under here):
  - `__main__.py` — the CLI (compile / search / repl / status / audit-corpus).
  - `config.py` — the unified config loader: one JSON = provider + corpus
    settings (see "Provider configs").
  - `compile.py` — pipeline orchestration (deps → corpus → index).
  - `cli/` — command implementations (compile / search / status / repl).
  - `corpus/` — page→markdown extraction, LLM chunk generation, prompts +
    msgspec schema (`extract.py`, `generate.py`, `prompts.py`, `schema.py`).
  - `index/` — BM25 + BGE-M3 vector indexes, RRF fusion, index build
    (`bm25_index.py`, `vector_index.py`, `fuse.py`, `build.py`).
  - `search/` — hybrid query engine + cross-encoder reranker (`engine.py`,
    `rerank.py`; `engine_select.py` is the RAG/legacy auto-switch used only by
    the legacy `hybrid_retrieve.py`).
  - `llm/` — vendored tool-free LLM provider clients (anthropic / kimi /
    openai_legacy / openai_responses; retry + circuit breaking in `base.py`).
  - `store/` — corpus store + md5 file manager behind the incremental build
    (`corpus_store.py`, `fileman.py`).
- `config.json` — THIS deployment's active config (input/output
  paths, embed model, `rrf_k`, mode, `rerank` — settings only, no
  provider). The generic self-contained template is
  `config.example.json`; `rag_probe_config.json` /
  `rag_probe_rerank.json` are small-corpus probe variants.
- `rag/legacy/` — the old BM25-only retriever, kept only for
  baseline reproduction (`hybrid_retrieve.py`, `retrieval.py`,
  `unity_tokenizer.py`, `doc_clean.py`, `dumpdoc.py`,
  `build_word_index.py`).
- `rag/eval/` — retrieval-quality harnesses + gold sets
  (`eval_rag.py`, `eval_retrieval.py`, `eval_lib.py`, ...).
- `corpus/`, `index/` — generated artefacts (gitignored; layout in the table
  below). `corpus_probe/`, `index_probe/` — probe-corpus variants.
- `tests/` — pytest suite, network-free.
- `eval_results.md`, `eval_results_latest.md` — curated + raw
  retrieval-quality results (the harnesses live in `rag/eval/`).
- Legacy generated artefacts (baseline reproduction,
  gitignored): `chunks.pkl`, `index_word.pkl`,
  `retriever_config.json`.

## Usage

### Build (compile)

```bash
# full pipeline (deps -> corpus -> index), local model, no quota, no gateway.
# The SAME file family is both provider and corpus config: config.json carries
# this deployment's dirs/paths/tuning; the provider config adds model+keys.
# With no --config at all, ./config.json in the CWD is used when present.
uv run python -m rag compile --config config.json --config llama_cpp/provider-qwen35-local.json
# same, on API gateways: one provider config per DISTINCT quota pool (see
# gateway facts below), --no-thinking is ~3x faster per page, 4 workers per provider:
uv run python -m rag compile --config config.json --config <pool-a.json> --config <pool-b.json> \n    --no-thinking --workers 8
# preview before spending anything (page counts, token/cost estimate):
uv run python -m rag compile --config llama_cpp/provider-qwen35-local.json --config config.json --dry-run
# maintenance after the full build exists:
uv run python -m rag compile --config config.json --config llama_cpp/provider-qwen35-local.json # incremental: md5 diff, changed pages only, no-op when clean
uv run python -m rag compile --config config.json --config ... --only Manual/foo.html # regenerate exactly one page
uv run python -m rag compile --steps index # rebuild search indexes from corpus (BM25 ~10 s; dense ~7 min on GPU, resumable)
uv run python -m rag compile --steps index --skip-dense # BM25 only, minutes
uv run python -m rag status # freshness, coverage, pending pages
```

Compile is md5-incremental, checkpointed, and safe to interrupt — kill it any
time and rerun the same command; it continues where it left off. First full
build of the 44k-page corpus takes ~1 day with a gateway fleet, longer on the
local 9B; every later run is minutes. A full unattended build is driven by
`.kimix_cache/run_full_compile.sh`, an auto-resume loop around `compile --steps
corpus`.

### Search

```bash
uv run python -m rag search --query "Rigidbody.AddForce"            # top-k with snippets (default mode from config: bm25)
uv run python -m rag search --query "..." --mode hybrid --k 10      # dense+RRF fusion — escape hatch for typos/paraphrase (measured: loses to bm25 on exact API names)
uv run python -m rag search --query-file q.txt --k 10               # batch, one index load
uv run python -m rag search --mentions MaterialPropertyBlock        # literal term enumeration across docs, with counts + context
uv run python -m rag search --query "..." --explain                 # per-hit BM25/dense scores + ranks + RRF breakdown, term df table
uv run python -m rag search --query "..." --out hits.json           # JSON output for scripts
```

Defaults come from `config.json` (or the `--config` you passed): `corpus_dir`, `index_dir`, `embed_model`,
`rrf_k`, `mode` (bm25 — see the gate numbers below), `dense_k`, `rerank` (off).

## How compile works (and why it is safe to interrupt)

1. **Scan + md5 diff** against `corpus/manifest.json` → added / changed /
   removed / unchanged. Only added+changed pages hit the LLM.
2. **Per-page gen_key** = sha1(prompt_version | model | extractor_version |
   schema_version). A bump of any component (or a page moving to a different
   provider shard) requeues that page. Per-page keys live in
   `manifest.page_gen_keys`.
3. Generation (tool-free, system+user prompt only): strict JSON →
   truncated-closer repair → json_repair salvage (hallucinated breakage:
   trailing commas, unterminated strings, stray closers) → msgspec validation
   (chunk text MUST be a verbatim excerpt of the page markdown, checked
   whitespace/punctuation/quote-insensitively) → up to 3 fresh self-repair
   sessions with the error list echoed back → CorpusGenerationError if still
   unrepairable (caught per page by the compile loop → heuristic fallback,
   fixed-size chunker with empty aux fields, flagged needs_regen). The
   heuristic fallback now covers API failure only. Acceptance measured on 40
   sampled pages with the shipped config: 98% first-try valid, 100% usable,
   2.20 chunks/page, 0 verbatim violations (eval_corpus_quality.py).
4. **Checkpointing**: every 25 pages the manifest is atomically rewritten,
   merging prior entries. Kill the process any time; rerunning `compile`
   continues exactly where it left off. `corpus/failures.jsonl` logs pages that
   needed repair/fallback.
5. **Multi-provider sharding**: with several `--config` flags (or a `"providers"` list inside one config), pages are
   assigned deterministically (`md5(rel) % n_providers`), each provider gets
   `workers/n` concurrency. Adding a provider later rekeys every page once
   (expected-key mismatch), then settles.

## Index layout (all generated, all gitignored)

| Path | Content |
| --- | --- |
| `corpus/<rel>.rag.json` | per-page chunks + aux fields + html_md5 + gen_key |
| `corpus/manifest.json` | md5 registry + gen keys (the incrementality memory) |
| `corpus/failures.jsonl` | pages that needed repair/fallback, with errors |
| `index/chunks.msgpack` | the single flat chunk table (chunk_uid keys both indexes) |
| `index/bm25_word.pkl` | word-tokenizer BM25 over `to_index_text` + path terms ×3 |
| `index/vectors.f32` | BGE-M3 embeddings of `to_embed_text` (clean text ONLY) |
| `index/vector_meta.json` | dim/count/model/normalized |
| `index/manifest.json` | n_chunks, corpus gen_key guard, input sha1s |

Index text composition (the aux design):

- **BM25 indexes `title + heading_path + text` — aux fields are NOT indexed by
  default** (`"bm25_aux": false`). This is a measured decision, and it *reversed*
  as the corpus grew: aux helped at 3k pages (0.917 vs 0.906) but hurt badly at
  26k (0.826 vs **0.881**). Two mechanisms, both scaling with corpus size: every
  `MaterialPropertyBlock.SetX` sibling page's aux repeats the parent symbol (so
  aux destroys that term's idf), and aux lengthens documents (avgdl 124 → 169)
  which BM25's length normalization penalizes. Set `bm25_aux: true` to re-enable;
  `index/manifest.json` records which setting built the current index.
- **Dense embeds `title + heading_path + text` only.** Aux fields are synthetic
  and must not pollute the vector space — the old word+dense(hash) regression
  (MRR 0.875 → 0.792) is the cautionary tale.
- **Aux is still generated and stored** — it is not wasted. `qa` fields feed the
  extended gold set, `summary` is a good snippet source, and the fields remain
  available to a reranker. The default only excludes them from BM25's index text.
- Fusion is **RRF** (`Σ 1/(60+rank)`), not linear score fusion: BM25 scores are
  unbounded, cosine is bounded. The legacy linear-alpha path survives in
  `rag/index/fuse.py` for ablations.
- **There is no n-gram layer.** The plan framed the system as "BM25 + n-gram +
  dense", but the shipped index is BM25 + dense only. The legacy character-trigram
  back-end was retained solely for typo tolerance, and BGE-M3 does that job
  strictly better (8/8 vs 3/8 on typo queries) while the trigram model destroys
  exact identifiers (MRR 0.04–0.52 vs 0.875, and nondeterministic across runs).
  Evidence: `eval_typo_tolerance.py`, written up in `eval_results.md`.
- **The cross-encoder reranker is off by default** (`"rerank": false`), and should
  stay off: measured MRR 0.917 → 0.760 on base-24. It sees chunk text but not the
  source path, so it discards the `path_terms` signal that makes API lookups work.
  The code path exists and is tested (`tests/test_rerank.py`) for corpora whose
  documents are not identifiable by filename.

The index build refuses to mix generations: when the corpus gen_key changes,
`compile --steps index` demands a rebuild (it replaces all artefacts).

## Measured retrieval quality

The 24-query gold set lives in `eval_lib.py` / `eval_gold_extended.json`; the
RAG harness is `rag/eval/eval_rag.py` (ablation flags, gate check; run it as
`python -m rag.eval.eval_rag`). Results table:
`eval_results.md`; raw rows per run go to `eval_results_latest.md`, which
`emit()` writes so it can never clobber the curated findings.

Adoption gate (base-24, final-k=10, on the full corpus): MRR ≥ 0.875 /
hit@1 ≥ 0.833 / hit@10 ≥ 0.917, or a documented hit@10 win without MRR loss.

**FINAL (2026-09-20, full corpus 43,938/43,938 pages, 83,098 chunks): bm25
MRR 0.880 / hit@1 0.833 / hit@5 0.958 / hit@10 0.958 — GATE PASS** (gate:
0.875 / 0.833 / 0.917). `mode` in config.json is `"bm25"` and has been
since that measurement. Full table, per-query forensics, and the typo-rescue
analysis: `eval_results.md` → "Full-corpus final (2026-09-20)".

**hybrid FAILS the gate at full scale** (MRR 0.800 / hit@1 0.708; hit@10 only
ties bm25 at 0.958 — no win; dense alone 0.624). Cause: BGE-M3 dense drift
demotes rank-1 BM25 hits inside near-duplicate sibling families
(MaterialPropertyBlock.SetX, Rigidbody2D crowding); the RRF margins are
~0.0002–0.001, and the ablation loop (rrf_k 20/60/120, path_boost
0/1/3/5, aux on/off) found no config that rescues it. Residual value of
dense/hybrid: typo tolerance — at full scale bm25 rescues 0/6 typo queries
while hybrid/dense rescue 3/6 — so `--mode hybrid` is a targeted escape hatch,
not a default.

Do **not** trust numbers measured on a small subset: BM25-only scored 0.917 on a
3k-page probe and 0.826 on 26k pages. That gap is also what flipped the aux
default (see above).

Honesty note carried in `eval_results.md`: only `base-24` is an independent
gold set. `ext-16` / `probe` were harvested from corpus `qa` fields, and `qa.q`
is itself indexed in the BM25 aux text, so they measure a string the build was
handed. Use them for trends, not accuracy. n=24 is small; treat
second-decimal differences as noise.

Run the evals:
```
uv run python -m rag.eval.eval_rag --gold-set base --mode bm25   # the gate (now the default mode)
uv run python -m rag.eval.eval_rag --gold-set base --mode hybrid # reference: documents the MRR loss at scale
uv run python -m rag.eval.eval_rag --sweep-aux / --sweep-path-boost 0,1,3,5 / --sweep-rrf-k 20,60,120
```
```

## Provider configs

`--config` takes ONE JSON file that is BOTH the provider config and the
corpus config: the kosong/kimi-cli provider format (see
the fields below): `model`, `type`
the fields below): `model`, `type`
(`anthropic|kimi|openai_legacy|openai_responses`), `url` (→ base_url), `api_key`,
`max_tokens`, `capabilities` (`thinking`…), `thinking_effort`, `env`; unknown
keys are ignored with a warning) PLUS the corpus settings in the same
object: `dirs` (input dirs, relative to the config file — required for
the corpus step), `corpus_dir` (default `corpus`), `index_dir`
(default `index`) and the engine tuning (`mode`, `rrf_k`, `embed_model`,
`compile_workers`, `accept_legacy_models`, …). Extra provider shards go
in a top-level `"providers"` list. Relative `corpus_dir`/`index_dir`
anchor at the config file's directory, so config + documents form a
relocatable unit. `config.example.json` is the annotated template.
Vendored tool-free clients live in `rag/llm/`.
Hard-won gateway facts (measured 2026-09):

- **Always pass `--no-thinking` for corpus builds.** Anthropic-type gateways
  apply server-side thinking when the thinking field is absent: 8k output
  tokens and ~100 s/page instead of ~300 tokens / ~15 s. The clients now send
  `thinking: {type: disabled}` explicitly when thinking is off.
- **Per-provider concurrency > 4 triggers 429 throttling** on these gateways;
  `workers 4` per provider is the sweet spot (429/5xx/timeouts retry with
  exponential backoff, see `rag/llm/base.with_retry`).
- Use `--dry-run` before any big run: page counts, token estimates
  (ratio 1.0 without thinking, 3.0 with), optional cost via
  `--price-in/--price-out` per 1M tokens.
- **Several configs can be ONE quota pool.** A pool is `(host, api_key)`:
  `qwen_flash.json` / `qwen.json` / `ds_ali.json` all point at
  `token-plan.cn-beijing.maas.aliyuncs.com` with the same key, and
  `k27.json` / `k27-high.json` / `k3.json` all share `api.kimi.com` + one key.
  Passing them as "3 providers" buys zero redundancy — when kimi's 5-hour window
  ran out, all three died together and ~14k pages degraded to aux-less fallback
  before this was caught. List **one config per distinct pool**.
- Rolling quota windows kill whole fleets at once. When every circuit is open
  the run now pauses (up to provider_wait_budget_s, default 90 min) waiting
  for the window to reset instead of grinding the work list into heuristic
  chunks. If the budget expires it aborts, leaves those pages untouched, and
  exits 3 so an auto-resume wrapper retries rather than reporting success.
  Check `python -m rag status` → needs_regen for pages that did degrade.

Local inference (llama.cpp Qwen3.5-9B)

A local provider (`"type": "llama"`) serves corpus builds without any
gateway. `llama_cpp/` ships prebuilt CUDA binaries (gitignored, multi-100MB)
plus a ready config: `llama_cpp/provider-qwen35-local.json` (relative paths
anchor at the config file's directory). Managed mode:
`rag/llm/llama.py` spawns `llama-server.exe` with the GGUF from `models/`
(also gitignored), waits for `/health`, reuses the one warm server for every
page, and kills it on shutdown — atexit + finalizer, so a crashed run leaves
no GPU-resident orphan. Measured on this machine (RTX 4080 SUPER, `-ngl 99`,
ctx 8192): model load ≈5 s, ≈100–105 tok/s generation, <1 s warm turnaround.
Config keys + rebuild-from-source: `llama_cpp/USAGE.md`.

- Smoke: `uv run python .kimix_cache/_provider_smoke.py` — real server +
  real model through the provider; expect a coherent ~25-word reply and no
  `llama-server.exe` left in tasklist afterwards.
- `enable_thinking=false` (via `extra_body.chat_template_kwargs`): Qwen3.5 is
  a hybrid thinking model; skipping the thinking phase is ~10x faster and
  yields clean prose immediately. `--reasoning-format none` does NOT work for
  this model — its reasoning is emitted inside regular content, not as a
  separate segment.
- Known quirk: non-thinking Qwen3.5 sometimes emits EOS right after the last
  chunk object, dropping the JSON's trailing ]}. extract_json_object
  repairs exactly that (missing container closers only); json_repair then
  salvages broader hallucinated breakage (trailing commas, unterminated
  strings) before a self-repair session is spent.
- Fleet-switch safety: compile accepts a page only when its recorded
  `page_gen_key` matches a `--config` on the command line or a model in
  `config.json → accept_legacy_models`. Before pointing the build at the
  local model, add the outgoing fleet there first (currently includes
  `qwen3.8-flash` / `deepseek-v4.1-flash`) or ~40k pages look stale and a
  mass regeneration starts. Always `--dry-run` first and check
  `to process:` equals the expected small count.

Troubleshooting

- `search` says artefacts not found → run `compile` (corpus step, then index).
- Index "refusing to build / gen_key changed" → corpus was regenerated after
  the index; rerun `compile --steps index` (it rebuilds everything from the
  current corpus — BM25 ~10 s, dense ~7 min on GPU / a few hours on CPU, so
  stop any corpus build first to keep the CPU free).
- Slow compile → check `corpus/failures.jsonl` for the error mix. `429
  Throttling` means lower `--workers`; `403 ... usage limit` / `AccessDenied`
  means a provider's quota window or subscription is exhausted (see the gateway
  facts). The run pauses rather than degrading pages when every provider is down.
- Index/search reports a `vectors.f32` row-count or interrupted-embed mismatch →
  the dense build was killed or the corpus changed since; rebuild with
  `compile --steps index --force` (resumable, so this continues rather than
  restarting).
- Interrupted compile → just rerun the same command; the manifest diff resumes.
- **Does the auto-resume loop ever spin forever on a page that keeps failing?** No.
  Per-page fallbacks (bad JSON that survives one repair retry, an empty page) still
  exit **0**, so the loop terminates and the page stays flagged `needs_regen` for
  the *next* manual run. Only the all-providers-down abort exits **3**, which is the
  case where retrying genuinely helps (quota window resets). That split is
  deliberate: a permanently unchunkable page must not wedge the whole build.
  Check `python -m rag status` → `needs_regen` for the backlog, and
  `eval_corpus_quality.py` to confirm the first-try rate is healthy.
- A page's corpus looks wrong → `python -m rag compile --config config.json --config ... --only Manual/foo.html`
  regenerates exactly one page.
- Deleting corpus: `rm -rf corpus index` is safe; everything regenerates.

## Final engine numbers (full corpus, 2026-09-20)

The full build is complete: 43,938/43,938 pages generated, 83,098 chunks;
BM25 word tokenizer + path_boost 3, aux OFF; BGE-M3 fp16 on CUDA; RRF k=60;
rerank off; fuzziness 0; final-k=10. Base-24 (the independent gate set):

| config | MRR | hit@1 | hit@5 | hit@10 |
| --- | --- | --- | --- | --- |
| old baseline (historical record) | 0.875 | 0.833 | — | 0.917 |
| old baseline (re-measured 2026-09-20) | 0.833 | 0.750 | — | 0.917 |
| **new engine, bm25 (default)** | **0.880** | **0.833** | **0.958** | **0.958** |
| new engine, hybrid | 0.800 | 0.708 | 0.958 | 0.958 |
| new engine, dense | 0.624 | 0.542 | 0.750 | 0.875 |

Full details, the hybrid-failure forensics, the typo-rescue analysis, and the
contamination caveats: `eval_results.md` → "Full-corpus final (2026-09-20)".

## Testing

`uv run --extra dev python -m pytest tests/ -q` — **217 tests, all pass**.
pytest is a dev extra, so plain `uv run python -m pytest` fails; always use
`--extra dev`. Tests are network-free by design (httpx MockTransport for the
LLM wire format, scripted fake clients for corpus generation, tmpdir mirrors
for the file manager, interrupt/resume and provider-death/failover simulation
for the compile pipeline) and need no corpus/index artefacts — a fresh clone
runs green before anything is built.
