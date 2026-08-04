# llama.cpp router mode

jishubench has first-class support for llama.cpp [router mode](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#using-multiple-models), which loads and unloads models via API without restarting the server.

## Start llama-server (no `-m`)

```bash
~/.llama-app/llama serve \
  --host 0.0.0.0 --port 8080 \
  --models-dir /path/to/models \
  -np 4 --cache-ram 0 --no-cache-prompt \
  --reasoning off --seed 42 \
  --slots
```

In particular, for the two agent tasks (tau-bench, terminal-bench), we recommend serial execution (np = 1, with unlimited context).

### Multimodal layout

Place the mmproj file in a subdirectory named after the model:

```
/path/to/models/
  Qwen3.5-4B-Q4_K_M/
    Qwen3.5-4B-Q4_K_M.gguf
    Qwen3.5-4B-mmproj-BF16.gguf   # filename must start with "mmproj"
```

## Config

Enable router control in the device profile (e.g. `configs/devices/macstudio.yaml`):

```yaml
target:
  inference_base_url: http://user.local:8080/v1
  model: Qwen3.5-4B-Q4_K_M

  llama_cpp:
    mode: router
    lifecycle:
      ensure_loaded: true        # POST /models/load before each run
      unload_after_run: false    # keep loaded across sequential benchmarks
    preflight:
      check_health: true
      check_model_catalog: true
    # model_aliases:
    #   Qwen3.5-4B-Q4_K_M: "Qwen/Qwen3.5-4B-GGUF:Q4_K_M"
```

See [Configuration](../configuration/overview.md) for all `llama_cpp` keys. If llama-server requires auth, add `--api-key` at startup and set `TARGET__INFERENCE_API_KEY` on the Host — see [API keys](../configuration/api-keys.md).

## What jishubench controls via API

| Action | API call |
|--------|----------|
| Load model before run | `POST /models/load {"model": "Qwen3.5-4B-Q4_K_M"}` |
| Poll until loaded | `GET /models` (status: loading → loaded) |
| Unload after run | `POST /models/unload {"model": "Qwen3.5-4B-Q4_K_M"}` |
| Preflight health | `GET /health` |
| Verify ctx/slots | `GET /props?model=...` |

When loading a VLM, jishubench rescans the Target catalog, resolves the companion mmproj, verifies the router preset includes it, loads the main model, then checks `/props` reports `image` input.

Parameters `-np`, `-c`, `--cache-ram`, `--reasoning`, and `--mmproj` cannot be changed per request via `POST /models/load`; set them at server startup or in a preset file. jishubench relies on the router associating mmproj from the models-dir layout (or `presets.ini`) before load.

## CLI helpers

```bash
uv run jishubench target models
uv run jishubench target status
uv run jishubench target load --model Qwen3.5-4B-Q4_K_M
uv run jishubench target unload --model Qwen3.5-4B-Q4_K_M
```

## edge-eval-agent with router

Run the agent alongside the router:

```bash
edge-eval-agent --bind 0.0.0.0:9090
```
