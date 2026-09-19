# Agent Usage: Unity Manual RAG (LLM-built corpus + hybrid retrieval)

A two-command RAG system over `Manual/` + `ScriptReference/` (43,938 HTML pages,
Unity 6.x era):

- `compile` — LLM-generated corpus (semantic chunks + aux fields per page) →
  BM25 + dense indexes. md5-incremental, checkpointed, resumable.
- `search` — BM25 (word tokenizer + path terms) ∪ BGE-M3 dense, fused by RRF.

Operational manual below; the legacy BM25-only retriever (`hybrid_retrieve.py`,
chunks.pkl/index_word.pkl) is still available and documented at the end.

## Daily use

Build (first run takes ~1 day on this corpus; afterwards it is incremental):

```
# single provider
uv run python rag.py compile --provider D:/qwen_flash.json --no-thinking --workers 4

# several providers: pass --provider once per DISTINCT quota pool (see the gateway
# facts below). This roughly doubles throughput only when the pools are independent;
# two configs sharing one host+api_key are one pool and add no redundancy.
uv run python rag.py compile --provider D:/qwen_flash.json --provider D:/glm.json \
    --steps corpus --no-thinking --workers 8

# cost/plan preview before spending anything
uv run python rag.py compile --provider D:/qwen_flash.json --dry-run
```

The full build is driven by `.kimix_cache/run_full_compile.sh`, an auto-resume
loop around `compile --steps corpus` (see the runbook at the bottom).

Search:

```
uv run python rag.py search --query "Rigidbody.AddForce"
uv run python rag.py search --query-file q.txt --k 10          # batch (one load)
uv run python rag.py search --mentions MaterialPropertyBlock   # literal enumeration
uv run python rag.py search --query "..." --explain            # term df + RRF breakdown
uv run python rag.py status                                    # freshness + counts
```

`--explain` shows which query terms the index knows (with document frequencies)
and, per hit, the BM25 score/rank, dense score/rank, and RRF contributions.

Defaults come from `rag_config.json` (new keys: `corpus_dir`, `index_dir`,
`embed_model`, `rrf_k`, `mode`, `dense_k`, `rerank`). `retriever_config.json`
carries the same new keys plus the legacy ones.

## How compile works (and why it is safe to interrupt)

1. **Scan + md5 diff** against `corpus/manifest.json` → added / changed /
   removed / unchanged. Only added+changed pages hit the LLM.
2. **Per-page gen_key** = sha1(prompt_version | model | extractor_version |
   schema_version). A bump of any component (or a page moving to a different
   provider shard) requeues that page. Per-page keys live in
   `manifest.page_gen_keys`.
3. **Generation** (tool-free, system+user prompt only): strict JSON → msgspec
   validation (chunk text MUST be a verbatim excerpt of the page markdown,
   checked whitespace/punctuation/quote-insensitively) → one repair retry with
   the validation errors echoed → heuristic fallback (fixed-size chunker, empty
   aux fields). Measured on samples: ~80–95% first-try valid, 100% usable.
4. **Checkpointing**: every 25 pages the manifest is atomically rewritten,
   merging prior entries. Kill the process any time; rerunning `compile`
   continues exactly where it left off. `corpus/failures.jsonl` logs pages that
   needed repair/fallback.
5. **Multi-provider sharding**: with several `--provider` flags, pages are
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

- **BM25** indexes `title + heading_path + text + summary + keywords + synonyms
  + qa.q` — synthetic aux text boosts lexical recall; hits map back to the
  clean chunk. Ablate with `eval_rag.py --sweep-aux`.
- **Dense embeds `title + heading_path + text` only.** Aux fields are synthetic
  and must not pollute the vector space — the old word+dense(hash) regression
  (MRR 0.875 → 0.792) is the cautionary tale.
- Fusion is **RRF** (`Σ 1/(60+rank)`), not linear score fusion: BM25 scores are
  unbounded, cosine is bounded. The legacy linear-alpha path survives in
  `rag/index/fuse.py` for ablations.

The index build refuses to mix generations: when the corpus gen_key changes,
`compile --steps index` demands a rebuild (it replaces all artefacts).

## Measured retrieval quality

The 24-query gold set and the old engine's numbers are in `eval_lib.py` /
`eval_retrieval.py`; the new engine's harness is `eval_rag.py` (same gold set +
an extended LLM-assisted set in `eval_gold_extended.json`, ablation flags,
gate check). Results table: `eval_results.md`.

Old engine, word tokenizer, fuzziness 0 (the baseline the new engine must meet
or beat): **MRR 0.875 / hit@1 0.833 / hit@10 0.917** (24 queries, final-k=10).

New engine numbers: see `eval_results.md` (regenerated by `eval_rag.py`).
**The gate has PASSED** on the independent base-24 set: MRR 0.917 / hit@1 0.875 /
hit@10 0.958, i.e. better than the old 0.875 / 0.833 / 0.917 on all three. Those
figures come from a PARTIAL corpus (~19k of 43,938 pages) and must be re-measured
after the full build — see the runbook.

Two honesty notes carried in `eval_results.md`:
* Only `base-24` is an independent gold set. `ext-16` / `probe` were harvested
  from corpus `qa` fields, and `qa.q` is itself indexed in the BM25 aux text, so
  they measure a string the build was handed. Use them for trends, not accuracy.
* `hybrid_retrieve.py` auto-selects the legacy engine until the RAG index covers
  ≥95% of the mirror, so a partial build never silently narrows retrieval.
  Override with `--rag` / `--legacy`; `rag.py status` reports the choice.

Run the evals:

```
uv run python eval_retrieval.py          # old engine sweep (needs legacy artefacts)
uv run python eval_rag.py --gold-set all --mode hybrid
uv run python eval_rag.py --sweep-aux / --sweep-path-boost 0,1,3,5 / --sweep-rrf-k 20,60,120
```

## Provider configs

`--provider` takes the kosong/kimi-cli provider JSON format (see
`D:/qwen_flash.json`, `D:/k27.json`): `model`, `type`
(`anthropic|kimi|openai_legacy|openai_responses`), `url` (→ base_url), `api_key`,
`max_tokens`, `capabilities` (`thinking`…), `thinking_effort`, `env`; unknown
keys are ignored with a warning. Vendored tool-free clients live in `rag/llm/`.

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
- **Rolling quota windows kill whole fleets at once.** When every circuit is open
  the run now **pauses** (up to `provider_wait_budget_s`, default 90 min) waiting
  for the window to reset instead of grinding the work list into heuristic
  chunks. If the budget expires it aborts, leaves those pages untouched, and
  exits **3** so an auto-resume wrapper retries rather than reporting success.
  Check `rag.py status` → `needs_regen` for pages that did degrade.

## Troubleshooting

- `search` says artefacts not found → run `compile` (corpus step, then index).
- Index "refusing to build / gen_key changed" → corpus was regenerated after
  the index; rerun `compile --steps index` (it rebuilds everything from the
  current corpus — BM25 minutes, dense hours on CPU).
- Slow compile → check `corpus/failures.jsonl` for 429 storms; lower `--workers`.
- Interrupted compile → just rerun the same command; the manifest diff resumes.
- A page's corpus looks wrong → `compile --only Manual/foo.html --provider ...`
  regenerates exactly one page.
- Deleting corpus: `rm -rf corpus index` is safe; everything regenerates.

## Runbook: completing a full build (state 2026-09-19)

The full-corpus compile runs via `.kimix_cache/run_full_compile.sh` (auto-resume
loop, `--no-thinking --workers 8`, ~0.6 pages/s). It is safe to interrupt: the
loop or a manual rerun resumes from the md5 manifest, skipping finished pages.
Watch progress with `uv run python rag.py status`.

When `tail .kimix_cache/full_compile.log` shows `COMPILE COMPLETE`:

```
# 0. repair the manifest if status reported phantom entries (do this ONLY when
#    no compile is running — it rewrites corpus/manifest.json)
uv run python rag.py status            # look for the phantom WARNING line
uv run python rag.py audit-corpus

# 1. build both index layers (BM25 minutes; BGE-M3 dense ~1.5-3 h on this CPU)
uv run python rag.py compile --steps index --force

# 2. regenerate the extended gold set from the finished corpus, then evaluate
uv run python eval_rag.py --generate-gold
uv run python eval_rag.py --gold-set base --mode hybrid   # the gate
uv run python eval_rag.py --gold-set base --mode bm25
uv run python eval_rag.py --gold-set base --sweep-aux
uv run python eval_smoke.py                               # integration smoke

# 3. gate: the new engine becomes the default only if eval_results.md shows
#    MRR >= 0.875 AND hit@10 >= 0.917 on the INDEPENDENT base-24 set, or a
#    documented hit@10 win without MRR loss. ALREADY PASSED on a partial corpus
#    (MRR 0.917 / hit@1 0.875 / hit@10 0.958); re-confirm on the full corpus.
#    Do NOT gate on ext/probe sets — they are contaminated (see eval_results.md).

# 4. once coverage >= 95%, hybrid_retrieve.py switches to the RAG engine by
#    itself; `rag.py status` prints which engine would serve a query and why.
uv run python rag.py search --query "MaterialPropertyBlock" --k 5 --text
uv run python rag.py search --mentions MaterialPropertyBlock | head
```

Then fill "New engine numbers" below with the full-corpus rows and delete this
runbook section.

### New engine numbers (PARTIAL corpus — replace after the full build)

Measured with ~19k of 43,938 pages generated, so treat as provisional:

| config | gold set | MRR | hit@1 | hit@10 |
| --- | --- | --- | --- | --- |
| old baseline (historical record) | base-24 | 0.875 | 0.833 | 0.917 |
| old baseline (re-measured today) | base-24 | 0.833 | 0.750 | 0.917 |
| **new engine, bm25** | base-24 | **0.917** | **0.875** | **0.958** |
| new engine, hybrid (default) | base-24 | 0.917 | 0.875 | 0.958 |

Full details, ablations, the hash-regression post-mortem, and the contamination
caveats: `eval_results.md`.

## Testing

`uv run python -m pytest tests/` — 91 tests, all network-free (httpx
MockTransport for the LLM wire format, scripted fake clients for corpus
generation, tmpdir mirrors for the file manager, interrupt/resume and
provider-death/failover simulation for the compile pipeline).

## Legacy retriever (still present, auto-selected only while the RAG index is partial)

The eval gate has PASSED, so the RAG engine is the intended default. But
`hybrid_retrieve.py` still picks the engine per invocation via
`rag/search/engine_select.py`: it uses the RAG engine only when `index/` is built
**and** covers ≥95% of the mirrored HTML pages, and otherwise falls back to the
legacy artefacts (which span the whole mirror) while printing why. `--rag` and
`--legacy` force either side. Once the full build finishes and the index is
rebuilt, the switch is automatic and permanent — no config edit needed.

The legacy engine (chunks.pkl + index_word.pkl, linear fusion, ollama/st/hash
embedders) is preserved behind `--legacy` and still powers `eval_retrieval.py`,
so the recorded baseline stays reproducible. Files: `hybrid_retrieve.py`,
`retrieval.py` (BM25 lib), `unity_tokenizer.py`, `doc_clean.py`, `dumpdoc.py`,
`build_word_index.py`, `eval_lib.py`.
