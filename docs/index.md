# jishubench

**Host–Target edge AI evaluation infrastructure** — evaluate models on edge
devices without running the evaluation harness on the device itself.

The **Host** runs the evaluation harness (dataset loading, task orchestration,
scoring); the **Target** runs only the inference service (e.g.
llama.cpp) plus the optional `edge-eval-agent` for hardware metric monitoring.

## Supported benchmarks

| Backend | Description | Submodule |
|---------|-------------|-----------|
| **lmms-eval** | Vision-language benchmarks (MMMU, RealWorldQA, …) | `submodules/lmms-eval` |
| **tau-bench** | Tool-agent-user interaction simulation | `submodules/tau2-bench` |
| **terminal-bench** | CLI agent evaluation with Docker sandbox | — (pip package) |
| **claw-eval** | Agent benchmark with Docker sandbox & granular task filtering | `submodules/claw-eval` |

## Quick start

```bash
git clone https://github.com/x-aijishu/jishubench.git --recursive
cd jishubench
uv sync --extra essentials  # everything except harbor
uv run jishubench preflight
```

The `essentials` extra installs all backends plus dev tooling. Use more
targeted extras to reduce dependencies:

```bash
uv sync --extra lmms-eval          # vision-language only
uv sync --extra tau                # tool-agent-user only
uv sync --extra terminal           # terminal-bench 1 only
uv sync --extra harbor             # terminal-bench 2 only
uv sync --extra claw-eval          # claw-eval only
uv sync --extra dev                # dev tooling only (pytest, ruff, mkdocs)
```

> **Note:** `harbor` conflicts with `tau` / `essentials` — do not combine them.

### Configure a device profile

Pick or create a YAML profile in `configs/devices/`. Set the Target inference
URL and model, then run:

```bash
uv run jishubench run mmmu_val --config macstudio --limit 10
```

Or set a persistent default so `--config` is no longer needed:

```bash
uv run jishubench config use   # user-friendly choosing
uv run jishubench run mmmu_val --limit 10
```

## Documentation

Full guides are published from the `docs/` tree via [MkDocs](https://www.mkdocs.org/):

| Topic | Link |
|-------|------|
| Quick start | [quickstart.md](quickstart.md) |
| Architecture | [architecture.md](architecture.md) |
| Configuration overview | [configuration/overview.md](configuration/overview.md) |
| CLI reference | [cli.md](cli.md) |
| Agent benchmarks config | [tau-bench.md](configuration/tau-bench.md) · [terminal-bench.md](configuration/terminal-bench.md) · [claw-eval.md](configuration/claw-eval.md) |
| API keys | [configuration/api-keys.md](configuration/api-keys.md) |
| edge-eval-agent | [edge-eval-agent/index.md](edge-eval-agent/index.md) |
| llama.cpp router | [deployment/llama-cpp-router.md](deployment/llama-cpp-router.md) |

Preview the doc site locally:

```bash
uv run mkdocs serve    # http://127.0.0.1:8000
uv run mkdocs build    # output in site/
```

## Security

See the repository's `SECURITY.md` for supported versions and how to report
vulnerabilities.

## Acknowledgments

This project builds on several open-source evaluation frameworks:

- **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)** — dataset loading, task definition, and scoring for vision-language benchmarks
- **[tau2-bench (tau-bench)](https://github.com/sierra-research/tau2-bench)** — tool-agent-user interaction simulation
- **[terminal-bench](https://github.com/harbor-framework/terminal-bench)** — CLI agent evaluation harness with Docker containers
- **[claw-eval](https://github.com/ByteDance-BTTE/claw-eval)** — agent benchmark with Docker sandbox

We thank the maintainers and contributors of these projects for their excellent work.
