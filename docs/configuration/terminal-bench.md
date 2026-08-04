# Terminal-Bench v0.1.1

Terminal-Bench v0.1.1 runs on the **Host** (Docker sandbox + `tb` Python API).
The Terminus agent calls the **Target** OpenAI-compatible inference endpoint
configured under `target.*`.

Install the optional extra (requires Python ≥3.12 for upstream `terminal-bench`):

```bash
uv pip install 'jishubench[terminal]'
```

> Terminal-Bench **2.0** (Harbor) is a separate, fully-decoupled adapter — see
> [Terminal-Bench 2.0 (Harbor)](terminal-bench-2.md). It has its own config
> section (`terminal_bench_2:`), CLI benchmark (`terminal-bench-2`), and
> shares no code with this adapter.

| Key | Default | Description |
|-----|---------|-------------|
| `terminal_bench.agent` | `terminus-2` | Agent scaffold passed to the harness |
| `terminal_bench.dataset` | `terminal-bench-core==0.1.1` | Registry dataset id |
| `terminal_bench.n_concurrent` | `1` | Concurrent Docker trials on Host |
| `terminal_bench.n_attempts` | `1` | Attempts per task; a task counts as resolved if any attempt passes (pass@k) |
| `terminal_bench.max_episodes` | `null` | Optional cap on agent episodes per trial |
| `terminal_bench.global_timeout_multiplier` | `1.0` | Scale per-task agent/test timeouts (e.g. `3.0` → 360s agent default becomes 1080s) |
| `terminal_bench.global_agent_timeout_sec` | `null` | Fixed agent timeout in seconds (overrides multiplier when set) |
| `terminal_bench.global_test_timeout_sec` | `null` | Fixed test timeout in seconds (overrides multiplier when set) |
| `terminal_bench.cache_dir` | `null` | Optional dataset/registry cache directory |
| `terminal_bench.agent_kwargs` | see base.yaml | Passed as `--agent-kwarg` (e.g. `temperature`, `parser_name`); `api_base` defaults to `target.inference_base_url` |

Commands:

```bash
# Preflight (Docker + tb CLI + Target)
jishubench preflight --benchmark terminal-bench

# Pre-download dataset cache
jishubench download terminal-bench

# Run agent eval (Host Docker + Target llama-server)
jishubench run terminal-bench --limit 5
```

### Headline metric

Stdout / `summary.json` report the upstream Terminal-Bench field
`metric: accuracy` (`n_resolved / n_total`). This matches the harness
`results.json` resolve rate.

### Output & cache

- Run artifacts: `outputs/runs/terminal-bench/<model>__<timestamp>/`
  (harness `results.json`, plus jishubench `summary.json` / `manifest.json`).
- Result cache: `outputs/shared/cache/terminal_bench_cache.jsonl`
  (one JSONL record per resolved/unresolved task, append-only).

### ARM64

Terminal-Bench runs require an x86-64 Host. ARM64 Hosts are unsupported because
the task Docker images are `linux/amd64` only and are not supported under QEMU
emulation. Preflight and `run` both reject ARM64.

See also: [Terminal-Bench 2.0 (Harbor)](terminal-bench-2.md),
[Configuration overview](overview.md) for shared `target`, `eval`, and logging
settings.
