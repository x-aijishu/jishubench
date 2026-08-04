# Terminal-Bench 2.0 (Harbor)

Terminal-Bench 2.0 (TB 2.0) is a **harder, better-verified** successor to
Terminal-Bench v0.1.1. It is run by **Harbor** — the official harness from the
same team, a ground-up rewrite of the legacy `tb` harness for improved
reliability, observability, scalability, and performance.

## Install

```bash
uv sync --extra harbor
```

The `harbor` extra installs the Harbor package. It **conflicts** with the `tau`
and `essentials` extras (litellm version incompatibility, enforced in
`pyproject.toml`) — do not combine them in one environment.

## Configuration

Settings live under `terminal_bench_2:` in `configs/base.yaml` and any device
profile. The defaults are:

```yaml
terminal_bench_2:
  agent: terminus-2
  dataset: terminal-bench@2.0
  global_timeout_multiplier: 3.0
  n_concurrent: 1
  n_attempts: 1
  agent_kwargs:
    temperature: 0.0
```

| Key | Default | Description |
|-----|---------|-------------|
| `terminal_bench_2.agent` | `terminus-2` | Agent scaffold passed to `harbor run --agent` |
| `terminal_bench_2.dataset` | `terminal-bench@2.0` | Harbor registry dataset id (`name@version`) |
| `terminal_bench_2.n_concurrent` | `1` | Concurrent Docker trials on Host |
| `terminal_bench_2.n_attempts` | `1` | Attempts per task; resolved if any attempt passes (pass@k) |
| `terminal_bench_2.max_episodes` | `null` | Optional cap on agent episodes per trial |
| `terminal_bench_2.global_timeout_multiplier` | `1.0` | Scale per-task agent/test timeouts |
| `terminal_bench_2.global_agent_timeout_sec` | `null` | Fixed agent timeout in seconds (overrides multiplier) |
| `terminal_bench_2.global_test_timeout_sec` | `null` | Fixed test timeout in seconds (overrides multiplier) |
| `terminal_bench_2.cache_dir` | `null` | Optional dataset/registry cache directory |
| `terminal_bench_2.agent_kwargs` | `{temperature: 0.0}` | Passed as `--agent-kwarg`; `api_base` defaults to `target.inference_base_url` |

`api_base` and `model_name` are injected automatically: `api_base` falls back to
`target.inference_base_url`, and `model_name` becomes
`openai/<resolved-target-model>` (litellm format) so Harbor's litellm layer
reaches the Target llama-server.

## Commands

```bash
# Preflight (Docker + harbor CLI + harbor package + Target)
jishubench preflight --benchmark terminal-bench-2

# Pre-download dataset cache
jishubench download terminal-bench-2

# Run (Host Docker + Target llama-server)
jishubench run terminal-bench-2 --limit 5
jishubench run terminal-bench-2 --task-id <task-name> --model Qwen3.5-4B-Q4_K_M
```

Under the hood, `run` shells out to:

```bash
harbor run \
  --dataset terminal-bench@2.0 \
  --agent terminus-2 \
  --model openai/<resolved-target-model> \
  --jobs-dir <runs_dir> \
  --job-name <run_id> \
  --n-concurrent <n> \
  --n-attempts <k> \
  --agent-kwarg api_base=<target.inference_base_url> \
  --agent-kwarg temperature=0.0 \
  --agent-kwarg model_name=openai/<resolved-target-model>
```

### Headline metric

Stdout / `summary.json` report `metric: accuracy` (`n_resolved / n_total`),
matching the Harbor trial resolve rate. A task counts as resolved if **any**
attempt resolves it (pass@k).

### Output & cache

- Run artifacts: `outputs/runs/terminal-bench-2/<model>__<timestamp>/`
  (harbor `--jobs-dir` + `--job-name`, plus jishubench `summary.json` / `manifest.json`).
- Result cache: `outputs/shared/cache/terminal_bench_2_cache.jsonl`
  (one JSONL record per resolved/unresolved task, append-only).

## Caveats

- **x86-64 only.** TB 2.0 task Docker images are `linux/amd64` only and are not
  supported under QEMU emulation on ARM64 hosts. Preflight and `run` both
  reject ARM64.
- **Not tested in CI.** Harbor integration is provided for parity with the
  upstream harness but is not exercised by the jishubench test suite. Prefer
  `terminal-bench` (TB v0.1.1) for validated runs.
- **Extra conflict.** `harbor` cannot be installed alongside `tau` /
  `essentials` (litellm). Use a dedicated environment.

## Migration from `terminal-bench`

If you previously set `terminal_bench.harness: harbor`, that field is removed.
Switch to the `terminal-bench-2` backend and the `terminal_bench_2:` config
section — the same Target endpoint and model work via `--model` /
`target.inference_base_url`.

See also: [Terminal-Bench v0.1.1 configuration](terminal-bench.md),
[CLI reference](../cli.md), [Configuration overview](overview.md).
