# Agent Usage: Unity Manual Hybrid Retrieve

A BM25 (+ optional dense) retriever over Manual/ and ScriptReference/ (~44k HTML files,
~83k chunks with boilerplate stripped).

## Daily use

The only required argument is `--query`:

```bash
cd D:\unity_manual
uv run python hybrid_retrieve.py --query "Rigidbody.AddForce"
```

Defaults come from `retriever_config.json`. Useful flags:

```bash
uv run python hybrid_retrieve.py --query "Collider" --final-k 10 --text
uv run python hybrid_retrieve.py --query "MaterialPropertyBlock" --explain   # show per-term df
uv run python hybrid_retrieve.py --query-file q.txt --out hits.json          # batch, one index load
uv run python hybrid_retrieve.py --mentions MaterialPropertyBlock --text     # enumerate all mentions
```

## Two index back-ends

| Tokenizer | Index file | Term model | Notes |
| --- | --- | --- | --- |
| `word` (default) | `index_word.pkl` | identifier-aware words + path/title field terms | Use this. |
| `ngram` (legacy) | `index.pkl` | character 3-grams, `fuzziness=AUTO` | Kept for typo tolerance / comparison only. |

`--tokenizer ngram` automatically switches to `ngram_index_path` unless you pass
`--index-path` explicitly.

**Why the default changed.** The character-trigram model tokenizes
`MaterialPropertyBlock` into 19 overlapping trigrams (`mat ate ter eri ria ial alp lpr pro
rop ope per ert rty tyb ybl blo loc ock`). Over half are generic English trigrams present in
almost every page, so BM25 loses all discriminative power and `min_should_match` stops meaning
anything — `--query "MaterialPropertyBlock"` used to return `Random.html` and
`BillboardAsset.html`. The `word` model emits `materialpropertyblock` (df 146 / 113k) plus its
camelCase parts, and adds terms derived from the source filename, which matters because pages
like `Renderer.SetPropertyBlock.html` never mention their own symbol in the body prose.

## Batch queries: do not loop over `--query`

Loading `index.pkl` costs ~14 s and 180 MB. Always batch instead:

```bash
uv run python hybrid_retrieve.py --query-file queries.txt --out results.json
```

`queries.txt` is one query per line; blank lines and `#` comments are ignored.

## Choosing between retrieval and enumeration

BM25 cannot answer "which pages mention this symbol at all?" — a page that mentions a term once
is ranked nearly the same as one that mentions it fifty times, and common terms dominate. Use
`--mentions TERM`, which is a literal count grouped by file, ordered by occurrences:

```bash
uv run python hybrid_retrieve.py --mentions prepareMaterialPropertyBlockCallback --text
uv run python hybrid_retrieve.py --mentions MaterialPropertyBlock --mentions-context 120 --text
```

A productive research loop is: `--mentions` to enumerate the authoritative pages, then
`--query` variants to rank them, then read the HTML directly.

## Reading the source pages

Retrieval returns ~500-char snippets, which are not enough to write accurate API documentation.
Read the underlying HTML with `dumpdoc.py`, which converts a doc page to markdown preserving
signatures, parameter tables and full code samples:

```bash
uv run python dumpdoc.py -o out.md ScriptReference/MaterialPropertyBlock.html \
    ScriptReference/Renderer.SetPropertyBlock.html Manual/DrawCallBatching-Properties.html
```

## Measuring retrieval quality

`eval_retrieval.py` scores configurations against a 24-query gold set derived from a real
verified documentation task. Gold sets are intentionally strict, so the true accuracy of the
remaining "misses" is higher than the numbers imply.

```bash
uv run python eval_retrieval.py                          # compare back-ends
uv run python eval_retrieval.py --configs word --misses  # show non-rank-1 queries
uv run python eval_retrieval.py --show-gold
```

Last measured (24 queries, `final-k=10`):

| config | tokenizer | fuzziness | MRR | hit@1 | hit@10 |
| --- | --- | --- | --- | --- | --- |
| `legacy` | ngram | AUTO | 0.04 – 0.13 (**varies per run**) | 0.00 – 0.04 | 0.17 – 0.29 |
| `nofuzz` | ngram | 0 | 0.521 | 0.417 | 0.625 |
| `word-nopath` | word | 0 | 0.691 | 0.625 | 0.875 |
| **`word` / `default`** | **word** | **0** | **0.875** | **0.833** | **0.917** |
| `word+dense` | word + hash | 0 | 0.792 | 0.667 | 0.917 |

Two separate defects in the legacy configuration, both worth knowing about:

1. **Nondiscriminative terms** — the trigram problem described above.
2. **Nondeterminism** — `fuzziness="AUTO"` expands each term through
   `LevenshteinAutomaton.match`, which fills a `set` and then truncates it to
   `max_expansions=50`. Python randomises string hashing per process, so set iteration order
   differs between runs and *which* 50 expansions survive changes too. Repeatedly evaluating the
   `legacy` config yields different MRR (observed 0.041 / 0.063 / 0.083 / 0.103 / 0.121 / 0.214).
   Setting `fuzziness: 0` makes results bit-identical across runs (verified 3× at MRR 0.521) and
   is part of why the default changed.

If you change tokenization, chunking, or fusion weights, re-run this. Adding gold queries from
each new research task keeps it honest.

## Embedders

`embedder` may be `none` (default), `ollama`, `st`, or `hash`.

`none` is the default because the offline `hash` embedder is a character-signature model, not a
semantic one, and was measured to **degrade** ranking on this corpus (MRR 0.875 → 0.792). It also
no longer silently substitutes for an unavailable backend — `get_embedder` falls back to
BM25-only and says so on stderr.

To use real dense retrieval:

1. `ollama pull nomic-embed-text`, start Ollama.
2. Set `"embedder": "ollama", "alpha": 0.7` in `retriever_config.json` (or pass on the CLI).

## Configuration

```json
{
  "tokenizer": "word",
  "path_boost": 3,
  "fuzziness": 0,
  "min_should_match": 0.6,
  "per_file": 1,
  "embedder": "none",
  "ollama_model": "nomic-embed-text",
  "bm25_k": 200,
  "final_k": 8,
  "alpha": 1.0,
  "dirs": ["Manual", "ScriptReference"],
  "index_path": "index_word.pkl",
  "ngram_index_path": "index.pkl",
  "chunks_path": "chunks.pkl"
}
```

- `tokenizer`: `word` | `ngram`.
- `path_boost`: how many times to repeat the filename/title field terms when indexing (0–5 are
  equivalent above 3; `3` is the default).
- `fuzziness`: keep `0` for the `word` index — edit-distance expansion of long identifiers is what
  made the legacy index unusable, and it is also the source of the legacy back-end's run-to-run
  nondeterminism (see the table above).
- `min_should_match`: `0.6` measured best; `0.7`+ starts dropping valid hits, and with the `ngram`
  index `0.9` returns **nothing**.
- `per_file`: max chunks kept per source document, for result diversity. `1` raised distinct files
  in the top-10 from 6.58 to 9.08 at no cost in hit rate. `0` disables.
- `bm25_k`: candidates pulled from BM25 before fusion.

CLI flags override config for a single run.

## Rebuilding

```bash
cd D:\unity_manual
uv run python hybrid_retrieve.py --build        # full corpus -> chunks.pkl + index_word.pkl
```

Takes a few minutes. To rebuild a single back-end without touching the other:

```bash
uv run python hybrid_retrieve.py --build --tokenizer ngram   # -> index.pkl
uv run python build_word_index.py --path-boost 3             # word index only, reuses chunks.pkl
```

`--build --max-files N` is refused unless you also pass `--force`, because it would otherwise
silently overwrite the full-corpus `chunks.pkl` with a partial one. For a quick smoke build,
redirect both artefacts:

```bash
uv run python hybrid_retrieve.py --build --max-files 3000 \
    --chunks-path /tmp/c_smoke.pkl --index-path /tmp/i_smoke.pkl --force
```

## Files

- hybrid_retrieve.py — main retrieval script (build, query, --mentions, --explain)
- doc_clean.py — shared HTML cleaning: content_root selector, page_title, strip_boiler/BOILER
  (site chrome lives in plain divs inside #content-wrap, so decomposing semantic tags is
  not enough; the footer/feedback lines are removed by pattern)
- `unity_tokenizer.py` — identifier-aware `WordTokenizer`, `split_identifier`, `path_terms`
- `build_word_index.py` — standalone word-index builder with `--path-boost` ablation support
- `eval_retrieval.py` / `eval_lib.py` — gold-set retrieval evaluation harness
- `dumpdoc.py` — HTML doc page → markdown (signatures, tables, code samples)
- `retrieval.py` — copied BM25 library from `D:/KimiX/src/kimix/retrieval.py`
- `retriever_config.json` — default settings
- `index_word.pkl` / `index.pkl` / `chunks.pkl` — generated indices
