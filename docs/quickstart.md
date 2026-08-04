# Quick start

End-to-end path from clone to a smoke-test benchmark on the Target.

## 1. Clone and submodules

```bash
git clone https://github.com/x-aijishu/jishubench.git --recursive
cd jishubench
```

Submodule contents:

| Submodule | Path | Purpose |
|-----------|------|---------|
| lmms-eval | `submodules/lmms-eval/` | Vision-language benchmark tasks and scoring |
| tau2-bench | `submodules/tau2-bench/` | Tool-agent-user interaction benchmark (tau-bench) |
| claw-eval | `submodules/claw-eval/` | Autonomous agent harness, tasks, fixtures, sandbox |

## 2. Install the Host

Install [uv](https://docs.astral.sh/uv/) if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the virtual environment and install everything with a single command:

```bash
# Core + dev tools + lmms-eval + terminal-bench (tb) + tau-bench + claw-eval (excludes harbor)
uv sync --extra essentials

# Or install individual extras only:
uv sync --extra lmms-eval          # vision-language benchmarks
uv sync --extra terminal           # Terminal-Bench (tb CLI)
uv sync --extra harbor             # Terminal-Bench 2 (harbor CLI)
uv sync --extra tau                # tau-bench (requires TAU_USER_API_KEY/DEEPSEEK_API_KEY)
uv sync --extra claw-eval          # Claw-Eval agent benchmark + docker (sandbox; included in essentials)
```

!!! note
    `essentials` bundles lmms-eval, terminal-bench, tau-bench, and claw-eval. The `harbor` extra (not-tested) conflicts with `tau` / `essentials` due to incompatible `litellm` versions — do not combine them.

`lmms-eval` is declared as a local path dependency via `[tool.uv.sources]` in
`pyproject.toml`, so uv installs it and all its dependencies (torch,
transformers, datasets, etc.) automatically — no separate `pip install` needed.

Enable git hooks so checks run automatically before each commit (one-time per clone):

```bash
uv run pre-commit install
```

Hooks are defined in `.pre-commit-config.yaml` (formatting, lint, type-check, Rust checks).
To run them manually on the whole tree: `uv run pre-commit run --all-files`.

## 3. Configure the Target

Edit `configs/devices/macstudio.yaml` or `.env` (see [Configuration](configuration/overview.md), optional [API keys](configuration/api-keys.md)):

| Setting | Purpose |
|---------|---------|
| `target.inference_base_url` | (optional) OpenAI-compatible API on the Target (llama.cpp, Ollama, …) |
| `target.model` | Default model name for runs |
| `target.monitor_base_url` | (optional) `edge-eval-agent` at `http://<edge-ip>:9090`  |
| `judge.openai_api_*` | (optional) Judge LLM on the Host for scoring |
| `target.llama_cpp` | llama.cpp router control plane (see [llama.cpp router](deployment/llama-cpp-router.md)) |

## 4. Deploy monitoring (optional)

`edge-eval-agent` exposes hardware metrics at `/v1/eval/*`. See [edge-eval-agent deployment](edge-eval-agent/index.md).

Set `target.monitor_base_url` after the agent is reachable from the Host.

## 5. Preflight and benchmarks

```bash
uv run jishubench preflight
```

List available benchmarks (no Target required):

```bash
uv run jishubench benchmarks
uv run jishubench benchmarks --kind groups
uv run jishubench benchmarks --search mmmu
uv run jishubench benchmarks --json
```

## 6. Run evaluations

```bash
uv run jishubench run mmmu_val --model Qwen3.5-4B-Q4_K_M
uv run jishubench run mmmu_val realworldqa --limit 10
```

`run` checks submodule availability and Target connectivity before starting. When `target.llama_cpp` is configured, it also ensures the model is loaded on the Target before the benchmark begins.

## 6b. Pre-download datasets (offline / edge)

On a network-connected Host, download datasets before moving caches to an air-gapped edge:

```bash
uv run jishubench download mmmu_val realworldqa
```

Use the same `--config` as `run` so `lmms_eval.hub_cache` and `lmms_eval.datasets_cache` match. After copying both cache directories to the edge Host, verify offline:

```bash
uv run jishubench download mmmu_val --check
```

For Claw-Eval fixtures:

```bash
uv run jishubench download claw-eval
uv run jishubench download claw-eval --check
```

## 7. Manage Target models (router mode)

```bash
uv run jishubench target models
uv run jishubench target models --reload
uv run jishubench target models --json
uv run jishubench target status
uv run jishubench target load --model Qwen3.5-4B-Q4_K_M
uv run jishubench target unload --model Qwen3.5-4B-Q4_K_M
```

## 8. Run tau-bench (tool-agent-user simulation)

Tau-bench evaluates the Agent (on Target) interacting with a remote user-simulator LLM across multi-turn tool-call conversations.

```bash
# Set user-simulator API key (DeepSeek or any OpenAI-compatible remote LLM)
export DEEPSEEK_API_KEY=sk-...

# Install tau extra
uv sync --extra tau

# Preflight: check submodule, tau2 install, data directory, and API key
uv run jishubench preflight --benchmark tau-bench --config configs/devices/user-local.yaml

# Run 2 tasks across the airline domain
uv run jishubench run tau-bench --limit 2 --config configs/devices/user-local.yaml

# Run specific domains
uv run jishubench run tau-bench --domain airline --domain retail --limit 5
```

Results are written under `eval.work_dir/runs/tau-bench/<run_id>/` (same layout as other adapters) and archived in a manifest JSON.

See [Configuration — Tau-Bench](configuration/tau-bench.md) for all settings.

## 9. Run Claw-Eval (autonomous agent harness)

Claw-Eval runs its upstream Host-side sandbox harness and sends the evaluated agent model to the Target OpenAI-compatible endpoint.

```bash
# Prepare fixtures (requires essentials or --extra claw-eval)
uv run jishubench download claw-eval

# Preflight: Target, package, tasks, fixtures, Docker/sandbox, judge key
uv run jishubench preflight --benchmark claw-eval --config configs/devices/user-local.yaml

# Smoke a few text-only tasks
uv run jishubench run claw-eval --limit 3 --config configs/devices/user-local.yaml

# Run one task without judge scoring
uv run jishubench run claw-eval --task-id T001zh_email_triage --no-judge
```

By default `claw_eval.sandbox=true`. The `claw-eval` extra installs `docker` via `claw-eval[sandbox]`. Build the sandbox image from the repo root (uses the jishubench venv):

```bash
uv run claw-eval build-image --config submodules/claw-eval/config_general.yaml
```

See [Configuration — Claw-Eval](configuration/claw-eval.md) for filters, text-only selection, and run settings.

## Edge deployment checklist

1. Target inference server exposes `http://<edge-ip>:8080/v1/chat/completions`
2. `target.inference_base_url` points to the edge IP (not `127.0.0.1` unless co-located)
3. (Optional) `edge-eval-agent` at `~/.jishubench/bin/edge-eval-agent`, listening on `0.0.0.0:9090`
4. `git submodule update --init submodules/lmms-eval`
5. Pre-download datasets on a connected Host: `uv run jishubench download mmmu_val` (rsync `hub_cache` + `datasets_cache` to edge)
6. On edge: `uv run jishubench download mmmu_val --check`, then `uv run jishubench preflight` and `uv run jishubench run mmmu_val --limit 10`
