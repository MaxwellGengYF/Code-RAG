# Agent Usage: Unity Manual RAG (LLM-built corpus + hybrid retrieval)

A two-command RAG system over `Manual/` + `ScriptReference/` (43,938 HTML pages,
Unity 6.x era):

- `compile` — LLM-generated corpus (semantic chunks + aux fields per page) →
  BM25 + dense indexes. md5-incremental, checkpointed, resumable.
- `search` — BM25 (word tokenizer + path terms) ∪ BGE-M3 dense, fused by RRF.

Operational manual below; the legacy BM25-only retriever (`hybrid_retrieve.py`,
chunks.pkl/index_word.pkl) is still available and documented at the end.

## Daily use

Build (first run takes ~1–2 days on this corpus; afterwards it is incremental):

```
uv run python rag.py compile --provider D:/qwen_flash.json --no-thinking --workers 4
uv run python rag.py compile --provider D:/qwen_flash.json --provider D:/k27.json \
    --steps corpus --no-thinking --workers 6   # two providers = ~2x throughput
```

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
The switch of the default engine is gated on beating that row (or winning
hit@10 with no MRR loss).

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

## Testing

`uv run python -m pytest tests/` — 57 tests, all network-free (httpx
MockTransport for the LLM wire format, scripted fake clients for corpus
generation, tmpdir mirrors for the file manager, interrupt/resume simulation
for the compile pipeline).

## Legacy retriever (kept until the eval gate passes)

`hybrid_retrieve.py` now delegates to the RAG engine by default. The old engine
(chunks.pkl + index_word.pkl, linear fusion, ollama/st/hash embedders) is
preserved behind `--legacy` and still powers `eval_retrieval.py`. Files:
`hybrid_retrieve.py`, `retrieval.py` (BM25 lib), `unity_tokenizer.py`,
`doc_clean.py`, `dumpdoc.py`, `build_word_index.py`, `eval_lib.py`.
