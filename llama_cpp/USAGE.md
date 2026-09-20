# llama.cpp local inference build (Windows, CUDA + Vulkan)

Pre-built **Release** binaries of [llama.cpp](https://github.com/ggml-org/llama.cpp)
for local inference with the RAG pipeline. Built natively on Windows (no WSL).

## Build provenance

| | |
|---|---|
| Source | `ggml-org/llama.cpp` @ `1af554f8f` (master, 2026-09-19) |
| Build type | `Release`, static CRT-linked (`BUILD_SHARED_LIBS=OFF`) |
| Generator | Ninja + MSVC 19.x (Visual Studio 2022 Enterprise) |
| Backends | CUDA 13.0 (`GGML_CUDA=ON`), Vulkan 1.4.321 (`GGML_VULKAN=ON`) |
| CUDA arch | `sm_89` (RTX 4080 SUPER, Ada) |
| Machine | NVIDIA GeForce RTX 4080 SUPER 16 GB + AMD Radeon iGPU (Vulkan only) |

Binaries are fully **static** — no DLLs needed; copy the folder anywhere.

`llama-server.exe --list-devices` output on this machine:

```
CUDA0:    NVIDIA GeForce RTX 4080 SUPER (16375 MiB)
Vulkan0:  NVIDIA GeForce RTX 4080 SUPER (16045 MiB)
Vulkan1:  AMD Radeon(TM) Graphics (48600 MiB)
```

## Binaries

| Binary | Purpose |
|---|---|
| `llama-server.exe` | OpenAI-compatible HTTP server (used by the RAG `llama` provider) |
| `llama-cli.exe` | Interactive / one-shot CLI chat |
| `llama-bench.exe` | Throughput benchmark (`-ngl 99` etc.) |
| `llama-quantize.exe` | GGUF quantization (e.g. re-quantize to Q4_K_M) |
| `llama-embedding.exe` | Text embeddings (RAG dense-retrieval experiments) |
| `llama-gguf-split.exe` | Split / reassemble large GGUF files |

Measured on this machine (Qwen3.5-9B Q4_K_M, CUDA, `-ngl 99`):
model load ≈ 3 s, generation ≈ **100–105 tokens/s**.

## Quick start

## Quick start
Commands below assume the repo root as CWD. In the RAG provider config,
relative `server_bin` / `model_path` values anchor at the config file's
directory, so the config ships next to its binary and just works from any CWD.
### One-shot chat
```bat
llama-cli.exe -m models\Qwen3.5-9B\Qwen3.5-9B-Q4_K_M.gguf ^
    -ngl 99 -c 8192 -n 256 --single-turn ^
    -p "In one sentence: what is Unity?"
```

(`--single-turn` exits after the reply instead of waiting for more stdin.)

### HTTP server

```bat
llama-server.exe -m models\Qwen3.5-9B\Qwen3.5-9B-Q4_K_M.gguf ^
    -ngl 99 -c 8192 --port 8080
```

Then:

```bat
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" ^
  -d "{\"model\":\"Qwen3.5-9B-Q4_K_M.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}"
```

Server UI: <http://127.0.0.1:8080>. GPU layer offload: `-ngl 99` = all layers.

## Thinking-model note (Qwen3.5-9B)

Qwen3.5-9B is a **hybrid thinking model**: by default it emits a long
`<think>…</think>` planning phase before the answer. Two ways to control it:

* **Skip thinking** (recommended for RAG batch work — ~10x faster):

  ```json
  "extra_body": { "chat_template_kwargs": { "enable_thinking": false } }
  ```
  (Works with `rag.llm` `llama` provider; also sendable as a request field.)

* **Keep thinking** in the reply: send nothing; thinking text arrives inside
  `content`. `--reasoning-format none` does **not** strip it for this model
  (its reasoning is emitted as ordinary content, not a separate segment).

## Using with the RAG pipeline (`rag.llm` provider `type: "llama"`)

`rag/llm/llama.py` implements the tool-free `LLMClient`
protocol against `llama-server`. Two modes:

* **Managed** (no `base_url`): the client spawns `llama-server.exe` on first
  `generate()`, waits for `/health`, reuses the warm server for every page,
  and kills it on `aclose()` / process exit.
* **External** (`base_url` set): connect to a server you started yourself.

A ready provider config ships here: [`provider-qwen35-local.json`](provider-qwen35-local.json)

RAG corpus compile with the local model:
RAG corpus compile with the local model:
```bat
  cd <repo root>
  .venv\Scripts\python.exe -m rag compile --config config.json --config llama_cpp/provider-qwen35-local.json
```

Minimal Python usage (pure conversation: prompt in → answer out):

```python
import asyncio
from rag.llm import create_llm
from rag.llm.config import ProviderConfig

  client = create_llm(ProviderConfig.from_file(
      "llama_cpp/provider-qwen35-local.json"))  # relative paths anchor at the config
async def main():
    res = await client.generate("You are a helpful assistant.",
                                "In one sentence: what is a Rigidbody?")
    print(res.text)          # answer
    print(res.input_tokens, res.output_tokens)
    await client.aclose()    # shuts the managed server down
asyncio.run(main())
```

### Provider config keys (llama-specific)

| Key | Default | Meaning |
|---|---|---|
| `server_bin` | `llama-server` (PATH) | Path to `llama-server.exe` |
| `model_path` | value of `model` | GGUF file to serve (managed mode) |
| `server_cmd` | — | Full custom argv template; replaces the default `-m …` invocation; `{model}`/`{host}`/`{port}` placeholders |
| `host` | `127.0.0.1` | Bind/connect host |
| `port` | `0` (auto) | Managed mode only; pick a free port |
| `ngl` | `99` | GPU layers to offload |
| `ctx_size` | `8192` | Context size |
| `extra_args` | `[]` | Extra `llama-server` CLI flags |
| `extra_body` | `{}` | Merged into every `/v1/chat/completions` request |
| `start_timeout` | `600` | Seconds to wait for model load |

Errors map onto the shared `rag.llm` hierarchy (`RateLimitError`,
`APIStatusError`, …) so the compile pipeline's retry/circuit-breaker behaviour
is unchanged.

## Rebuild from source

```bat
:: in D:\llama_cpp_official, from a VS 2022 x64 dev prompt:
set VULKAN_SDK=C:\VulkanSDK\1.4.321.1
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DGGML_CUDA=ON -DGGML_VULKAN=ON -DCMAKE_CUDA_ARCHITECTURES=89 ^
      -DBUILD_SHARED_LIBS=OFF
cmake --build build -j %NUMBER_OF_PROCESSORS%
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ggml_cuda_init failed` / no CUDA device | Driver too old for CUDA 13 runtime — update NVIDIA driver |
| Model loads but generation is slow | Check `-ngl 99` is applied; verify with `llama-bench.exe -m model.gguf -ngl 99` |
| `CUDA out of memory` | Lower `-c` (context) or `-ngl`; close other GPU apps |
| Server port busy | Managed mode auto-picks a port; for manual server use `--port <other>` |
| `llama-server binary not found` | Set `server_bin` in the provider config |
