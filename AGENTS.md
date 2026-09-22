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
  `--steps index` (BM25 ~seconds, dense minutes on GPU).

## Search

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
`rerank`, plus provider keys (`model`, `type`, `url`, `api_key`, ...). Relative
paths anchor at the config file, so config + documents form a relocatable unit.

## Tests

```bash
uv run --extra dev python -m pytest tests/ -q
```

Network-free, needs no corpus/index artefacts — a fresh clone runs green.
