# doc-rag

A two-command RAG system over any set of HTML documents: **compile** builds a
LLM-generated corpus (semantic chunks per page) + BM25/dense indexes;
**search** queries them. md5-incremental, per-page checkpointed, resumable.

## Install

```bash
uv sync                # default: HTML->markdown, BM25, LLM clients
uv sync --extra local  # + torch/BGE-M3 dense embeddings + reranker (local inference)
```

Missing extras are a supported state, never a traceback: dense/hybrid search
degrades to BM25 with a warning.

## Build

```bash
python -m rag compile --config config.json --steps deps    # HTML -> markdown corpus (LLM)
python -m rag compile --config config.json --steps index   # corpus -> BM25 (+dense) indexes
python -m rag compile --clean                              # wipe ALL artefacts, rebuild from scratch
```

- Incremental: rerun the same command after an interrupt; only added/changed
  pages hit the LLM. Ctrl-C is graceful (exit 130, in-flight pages checkpointed).
- `--dry-run` first on big builds: page counts + token estimates.
- Index refuses to mix corpus generations: after a corpus regen, rerun
  --steps index (BM25 ~seconds, dense minutes on GPU).

Provider gateway facts (hard-won, measured)

- Transient failures (429 / 5xx / timeouts / connection resets) are NOT
  fatal: each LLM call waits them out on a fixed schedule — 2 s → 4 s →
  1 min → 10 min → 1 h → 2 h → 4 h (RETRY_DELAYS in rag/llm/base.py) —
  before the page is demoted to the aux-less heuristic fallback. The short
  steps absorb ordinary throttling; the long tail rides out an exhausted
  ROLLING QUOTA WINDOW (these gateways reset on 5-hour windows), so a rate
  limit costs latency, not corpus quality. Waits ≥ 1 min print a
  [retry] <model>: ... waiting 10min line, so a stalled run is visibly
  waiting, not hung. Override per config with "retry_delays": [seconds, ...]
  (an empty list is rejected; the list length IS the retry count).
- A Ctrl-C during a long backoff drops that page unfinished (nothing sent,
  nothing written — RetryAborted) instead of degrading it: the
  graceful-interrupt contract still costs zero pages. 403 quota/access errors
  are NOT retried here; they trip the circuit breaker and fail over.
- Per-provider concurrency > 4 triggers 429 throttling on these gateways;
  workers 4 per provider is the sweet spot. Always pass --no-thinking for
  corpus builds: server-side thinking costs ~8k output tokens / ~100 s per
  page instead of ~300 tokens / ~15 s (the clients send
  thinking: {type: disabled} explicitly when thinking is off).
- One quota pool = (host, api_key): passing several configs that share both
  buys zero redundancy — when the window runs out they all die together.
  List one config per distinct pool.
- When every provider circuit is open the run pauses (up to
  provider_wait_budget_s, default 90 min) waiting for the quota window to
  reset instead of grinding pages into heuristic chunks; if the budget
  expires it aborts, leaves the pages untouched, and exits 3 so an
  auto-resume wrapper retries. Check python -m rag status → needs_regen for
  pages that did degrade.

Search

```bash
python -m rag search --query "Rigidbody.AddForce"     # markdown out (agent-friendly)
python -m rag search --query "..." --json | --text    # raw JSON / compact text
python -m rag search --mentions Rigidbody             # literal term -> files
python -m rag repl                                    # persistent JSONL session on stdin/stdout
python -m rag status                                  # build state, needs_regen backlog
```

Defaults come from config.json (`mode`, `final_k`, `embed_model`, ...). No
index yet → the error prints the exact build command.

## Server mode (warm index, ~100 ms/query)

```bash
python -m rag.server
```

- Binds 127.0.0.1:8642; if busy, rebinds an **ephemeral port** automatically.
- Announces its actual port in `$TMP/rag_server_<hash>.json` and removes it on
  shutdown. Clients discover the port from that file — **no port argument
  needed** on the agent side. Resolution: `--port` > `RAG_SERVER_PORT` > port
  file (liveness-probed, stale files ignored) > config `server_port` > 8642.
- API: `GET /health`, `POST /search`, `/mentions`, `/read` (JSON).
- `rag search` / `rag repl` try the server first, fall back to a local engine
  load with one warning line. `RAG_NO_SERVER=1` forces the local path.

## Config

One JSON = provider + corpus + tuning (see `config.example.json` for the
annotated template): `dirs` (input HTML, relative to the config), `corpus_dir`,
`index_dir`, `mode` (bm25|hybrid|dense), `rrf_k`, `embed_model`, `final_k`,
rerank, plus provider keys (model, type, url, api_key, ...). Relative
paths anchor at the config file, so config + documents form a relocatable unit.
Extra provider shards go in a top-level "providers" list; with several configs
pages are sharded deterministically (md5(rel) % n_providers) and adding or
swapping a provider later changes nothing for finished pages.

Local provider (type "llama")

A local provider serves corpus builds without any gateway:
rag/llm/llama.py spawns llama-server.exe with the GGUF named in the config
(relative paths anchor at the config file's directory), waits for /health,
reuses the one warm server for every page, and kills it on shutdown
(atexit + finalizer — a crashed run leaves no GPU-resident orphan). Measured
on RTX 4080 SUPER (-ngl 99, ctx 8192): model load ≈5 s, ≈100–105 tok/s
generation. For hybrid-thinking models pass enable_thinking=false via
extra_body.chat_template_kwargs (~10x faster); --reasoning-format none does
NOT work for those — the reasoning is emitted inside regular content.

Troubleshooting

- Slow compile → check corpus/failures.jsonl for the error mix. 429
  throttling is retried on the schedule above (the [retry] lines say so);
  403 usage limit / AccessDenied means the provider's quota window or
  subscription is exhausted.
- Interrupted compile → just rerun the same command; the md5 diff resumes.
  Ctrl-C is a supported exit: friendly notice + report + exit 130, never a
  traceback; the first Ctrl-C waits for and checkpoints in-flight pages.
- Index/search reports a vectors.f32 row-count or interrupted-embed mismatch
  → the dense build was killed or the corpus changed; rebuild with
  compile --steps index --force (resumable).
- A page's corpus looks wrong → python -m rag compile --config config.json
  --only Manual/foo.html regenerates exactly one page.

Tests

```bash
uv run --extra dev python -m pytest tests/ -q
[code block: 2 lines]

Network-free, needs no corpus/index artefacts — a fresh clone runs green.
279 pass (measured 2026-09-22 after the merge of the retry/failover work and
server mode). Tests simulate the missing optional stack (None in sys.modules),
so the suite is green both on a plain BM25 checkout and on a full GPU one.
pytest is a dev extra: always uv run --extra dev python -m pytest, never bare
uv run pytest (it can silently resolve the system Python and exercise
differently-pinned SDKs).
