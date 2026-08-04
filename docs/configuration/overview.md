# Configuration

Runtime settings are assembled from two YAML layers:

1. **`configs/base.yaml`** — shared defaults (`judge`, `eval`, `logging`, benchmark configs, `target.llama_cpp`)
2. **Device profile** (e.g. `configs/devices/macstudio.yaml`) — machine-specific overrides (`target.*` URLs, model name, personal cache paths)

When you pass `--config`, set it to the device profile. `base.yaml` is merged automatically.

```bash
uv run jishubench preflight --config configs/devices/macstudio.yaml
```

An optional repo-root `.env` file (gitignored) and process environment variables are applied on top. Nested keys use `__`, e.g. `TARGET__INFERENCE_BASE_URL`. Environment overrides take precedence over YAML values.

See [API keys](api-keys.md) for how credentials are injected and where each key is consumed.

## `target`

| Key | Default | Description |
|-----|---------|-------------|
| `inference_base_url` | `http://127.0.0.1:8080/v1` | OpenAI-compatible API base URL |
| `inference_api_key` | `EMPTY` | Target API key; see [API keys](api-keys.md) |
| `model` | `Qwen3.5-4B-Q4_K_M` | Default model name for runs and `target status` |
| `monitor_base_url` | `http://127.0.0.1:9090` | `edge-eval-agent` base URL |
| `monitor_timeout_s` | `30` | HTTP timeout for monitor client |
| `llama_cpp` | — | Router control plane (see below) |

## `target.llama_cpp`

Enable when the Target runs llama.cpp in [router mode](../deployment/llama-cpp-router.md).

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `router` | Router mode (only supported value today) |
| `api_key` | `""` | Optional control-plane override; falls back to `inference_api_key` — see [API keys](api-keys.md) |
| `model_aliases` | `{}` | Map friendly names to router catalog IDs |

### `lifecycle`

| Key | Default | Description |
|-----|---------|-------------|
| `ensure_loaded` | `true` | `POST /models/load` before each run |
| `autoload_fallback` | `true` | Fallback autoload behaviour |
| `wait_load_timeout_s` | `300` | Max wait for model to reach `loaded` |
| `poll_interval_s` | `2` | Poll interval while loading |
| `unload_after_run` | `false` | Unload model after each run |
| `unload_on_model_switch` | `true` | Unload when switching models |

### `preflight`

| Key | Default | Description |
|-----|---------|-------------|
| `check_health` | `true` | `GET /health` |
| `check_model_catalog` | `true` | Verify model appears in `GET /models` |

## `judge`

Judge LLM credentials for lmms-eval scoring on the Host (isolated from Target inference env).

| Key | Default | Description |
|-----|---------|-------------|
| `openai_api_base` | `https://api.openai.com/v1` | Judge API base |
| `openai_api_key` | `""` | Judge API key; see [API keys](api-keys.md) |

## `eval`

| Key | Default | Description |
|-----|---------|-------------|
| `work_dir` | `./outputs` | Run outputs directory |
| `api_nproc` | `4` | Concurrent API workers |
| `enable_traces` | `true` | Enable monitor traces when agent is configured |
| `save_samples` | `false` | Save per-sample outputs |

## `logging`

Host application logs for **`jishubench run` only** (console via Rich, optional global rotating file). Other subcommands do not call `setup_logging` or write log files.

Each run also writes a per-run log at `<run_dir>/logs/run.log` under `eval.work_dir/runs/` (independent of `save_dir`).

| Key | Default | Description |
|-----|---------|-------------|
| `level` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `save_dir` | `null` | Optional directory for global session log files (relative to repo root); each CLI invocation writes `YYYYMMDD-HHMMSS.log` when set |
| `max_bytes` | `10485760` | Max size per log file before rotation (10 MiB) |
| `backup_count` | `5` | Number of rotated backup files to keep |

Environment overrides: `LOGGING__LEVEL`, `LOGGING__SAVE_DIR`, etc.

## `lmms_eval`

| Key | Default | Description |
|-----|---------|-------------|
| `model_backend` | `async_openai` | lmms-eval model backend |
| `batch_size` | `1` | Batch size |
| `adaptive_concurrency` | `true` | Adaptive concurrency |
| `hub_cache` | — | Hugging Face hub cache path |
| `datasets_cache` | — | Hugging Face datasets cache path |

Both paths are used by `jishubench download` and `jishubench run`. For offline edge deployment, pre-download on a connected Host, copy both directories to the edge machine, then run `jishubench download BENCHMARK --check` before evaluation.

Example rsync (adjust paths and host):

```bash
rsync -av ~/datasets/huggingface/hub/ edge-host:~/datasets/huggingface/hub/
rsync -av ~/datasets/huggingface/datasets/ edge-host:~/datasets/huggingface/datasets/
```

## Agent benchmarks

Agent benchmark adapters share `target.*` for the evaluated model. Each has its own config namespace and optional install extra:

| Benchmark | Config prefix | Doc |
|-----------|---------------|-----|
| Terminal-Bench | `terminal_bench.*` | [terminal-bench.md](terminal-bench.md) |
| Claw-Eval | `claw_eval.*` | [claw-eval.md](claw-eval.md) |
| Tau-Bench | `tau_bench.*` | [tau-bench.md](tau-bench.md) |

## Example

```yaml
target:
  inference_base_url: http://user.local:8080/v1
  model: Qwen3.5-4B-Q4_K_M
  monitor_base_url: http://user.local:9090

  llama_cpp:
    mode: router
    lifecycle:
      ensure_loaded: true
      unload_after_run: false
    preflight:
      check_health: true

judge:
  openai_api_base: https://api.openai.com/v1
  openai_api_key: ""

eval:
  work_dir: ./outputs
  api_nproc: 16

logging:
  level: INFO
  save_dir: null
```

See `configs/base.yaml` and `configs/devices/` in the repository for the default settings.
