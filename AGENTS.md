# AGENTS.md

> Instructions for AI coding agents working on the jishubench codebase.

## 1. Environment & Dependency Management

This project uses **uv** (not pip). The virtual environment is already configured; do NOT re-create it or run `pip install`.

### First-time setup (for reference, already done)

```bash
git submodule update --init --recursive
uv sync --extra essentials
```

### All commands go through `uv`

| Do this | Not this |
|---------|----------|
| `uv sync` / `uv sync --extra <name>` | `pip install` / `pip install -r ...` |
| `uv run <command>` | Direct invocation of scripts (may use wrong interpreter) |
| `uv add <package>` | `pip install <package>` |
| `uv remove <package>` | `pip uninstall <package>` |
| `uv pip install <package>` | Only for packages not in pyproject.toml (rare) |

### Extras quick reference

| Extra | Contents | When to use |
|-------|----------|-------------|
| `dev` | pytest, ruff, pre-commit, mkdocs | Always needed for development |
| `lmms-eval` | lmms-eval + torch/transformers | Vision-language benchmark tasks |
| `tau` | tau2-bench runner | Tool-agent-user simulation |
| `terminal` | terminal-bench harness | Terminal-bench Docker tasks |
| `claw-eval` | claw-eval + docker (sandbox) | Agent benchmark with Docker sandbox |
| `essentials` | dev + lmms-eval + tau + terminal + claw-eval | All-in-one (excludes harbor) |
| `harbor` | harbor CLI (not-tested) | Conflicts with tau/essentials |

**Important:** `harbor` conflicts with `tau` and `essentials` — do not combine them (enforced in `pyproject.toml`).

### How to add a new dependency

Tell the user the package name and why it's needed. Do NOT add it without approval.

---

## 2. File & Directory Modification Policy

**Before creating any new file or directory, inform the user and wait for approval.** This includes:
- New Python modules, test files, scripts
- Configuration files (yaml, toml)
- Documentation files
- Any other new file

The user needs to assess whether the new file fits the project's directory structure.

**Modifying existing files** is generally OK without prior approval, but exercise judgment:
- Small, targeted changes (fix a bug, add a parameter, refactor a function) → go ahead
- Large restructurings or changes that touch many files → inform first

**Existing ignored working documents** are OK to edit directly when the user asks. In particular, `.gitignore` currently ignores `docs/design` and `docs/experiments`; existing files under those directories may not appear in `git status` or `git diff`. Do not ask for extra confirmation just because such a file is untracked/ignored. Preserve any existing data carefully, and validate changes by reading/checking the file contents instead of relying only on git-tracked diffs.

**Deleting files** also requires prior approval.

---

## 3. Project Structure

```
jishubench/
├── host/                        # Python: CLI, adapters, clients, config (main codebase)
│   ├── __init__.py
│   ├── main.py                  # CLI entry point (jishubench command)
│   ├── controller.py            # Run lifecycle and Target management
│   ├── adapters/                # One file per evaluation backend
│   │   ├── __init__.py          # Adapter registry
│   │   ├── lmms_eval.py         # Vision-language benchmarks
│   │   ├── tau_bench.py         # Tool-agent-user simulation
│   │   ├── terminal_bench.py    # Terminal-Bench v0.1.1 (tb harness)
│   │   ├── terminal_bench_2.py  # Terminal-Bench 2.0 (Harbor harness)
│   │   ├── claw_eval.py         # Agent benchmark with Docker sandbox
│   │   └── utils.py             # Shared adapter preflight helpers
│   ├── client/                  # HTTP clients
│   │   ├── llama_cpp_client.py  # llama.cpp data-plane health + control plane
│   │   └── eval_monitor_client.py  # edge-eval-agent metrics API
│   ├── config/                  # Pydantic settings, env, logging
│   │   ├── settings.py          # Settings model (single source of truth)
│   │   ├── profiles.py          # load_settings(): merges base.yaml + device yaml
│   │   ├── env.py               # Environment variable helpers
│   │   ├── submodules.py        # Submodule path resolution
│   │   └── logging_setup.py     # Loguru configuration
│   └── cli/                     # CLI command implementations
│       ├── __init__.py
│       ├── commands.py          # All subcommand logic
│       └── parser.py            # Argument parser
├── target/                      # Rust: edge-eval-agent sidecar
│   ├── agent/                   # HTTP API for hardware metric traces
│   ├── hal/                     # Hardware abstraction layer
│   └── monitor/                 # Power / utilization probes
├── submodules/
│   ├── lmms-eval/               # Git submodule: dataset loading and scoring
│   ├── tau2-bench/              # Git submodule: tool-agent-user simulation
│   └── claw-eval/               # Git submodule: agent benchmark with Docker sandbox
├── configs/
│   ├── base.yaml                # Shared default configuration
│   └── devices/                 # Machine-specific profiles
│       ├── macstudio.yaml
│       ├── local.yaml
│       ├── dgx-spark.yaml
│       └── ...
├── docs/                        # MkDocs documentation source
├── tests/                       # pytest test suite
│   ├── test_claw_eval.py
│   ├── test_cli.py
│   ├── test_controller.py
│   ├── test_llama_cpp.py
│   ├── test_lmms_eval.py
│   ├── test_logging.py
│   ├── test_profiles.py
│   ├── test_settings.py
│   ├── test_tau_bench.py
│   ├── test_terminal_bench.py
│   └── test_terminal_bench_2.py
└── outputs/                     # Run results (auto-generated, do not edit manually)
```

---

## 4. Configuration System

Configuration has **three layers**; understand the precedence:

1. **`configs/base.yaml`** — shared defaults for all machines (eval settings, logging, adapter defaults)
2. **`configs/devices/<machine>.yaml`** — device-specific overrides (target URL, model, cache paths)
3. **`.env`** — secrets and API keys (gitignored, never commit)

Loading (in `jishubench/host/config/profiles.py`):
```python
settings = load_settings("configs/devices/macstudio.yaml")  # merges base.yaml automatically
```

CLI override: `uv run jishubench run mmmu_val --config configs/devices/local.yaml`

**Key config paths** (dot-separated, used in YAML and CLI):
- `target.inference_base_url` — Target inference server URL
- `target.model` — Default model name
- `target.monitor_base_url` — edge-eval-agent URL
- `target.llama_cpp.*` — llama.cpp router settings
- `judge.openai_api_key` / `judge.openai_api_base` — Judge LLM
- `eval.work_dir` — Output directory
- `lmms_eval.*` — lmms-eval-specific settings
- `tau_bench.*` / `terminal_bench.*` — Adapter-specific settings

**`.env` file** (already exists, contains real keys):
```
DEEPSEEK_API_KEY=sk-...
JUDGE__OPENAI_API_KEY=sk-...
JUDGE__OPENAI_API_BASE=...
MODEL_NAME=...
```

---

## 5. Common Commands

```bash
# ----- Setup & verification -----
uv sync --extra dev --extra lmms-eval    # Install dependencies
uv run jishubench preflight              # Check submodule + Target readiness

# ----- Running benchmarks -----
uv run jishubench benchmarks             # List available tasks
uv run jishubench run mmmu_val --limit 10   # Smoke test
uv run jishubench run mmmu_val realworldqa --limit 10
uv run jishubench run tau-bench --limit 2
uv run jishubench run terminal-bench --limit 5

# ----- Testing -----
uv run pytest                              # Full test suite
uv run pytest tests/test_cli.py -v         # Single file
uv run pytest --cov=jishubench --cov-report=term-missing
cd jishubench/target && cargo test         # Rust tests

# ----- Linting & formatting -----
uv run pre-commit run --all-files          # All checks (ruff, mypy, Rust fmt/clippy)
# Focused mypy checks must use the pre-commit hook; `uv run mypy ...` is not available.
uv run pre-commit run mypy --files jishubench/host/controller.py  # Focused mypy via hook
uv run ruff check --fix .                  # Python lint + auto-fix
uv run ruff format .                       # Python format
# ----- Documentation -----
uv run mkdocs serve                        # Preview doc site (http://127.0.0.1:8000)
uv run mkdocs build                        # Build static HTML
```

---

## 6. Key Design Patterns

### Adapter Pattern

Each evaluation backend is a self-contained adapter in `jishubench/host/adapters/`. The `controller.py` dispatches to the right adapter based on the benchmark name:

```
controller.run("mmmu_val")          → lmms_eval.py adapter
controller.run("tau-bench")         → tau_bench.py adapter
controller.run("terminal-bench")   → terminal_bench.py adapter (TB v0.1.1)
controller.run("terminal-bench-2")  → terminal_bench_2.py adapter (TB 2.0 / Harbor)
```

To add a new backend, create a new adapter file and register it in `adapters/__init__.py`.

### Settings Loading

```python
# profiles.py — how config is assembled:
# 1. Read configs/base.yaml
# 2. Read device-specific yaml (overrides base)
# 3. Read .env (overrides yaml for matching keys)
settings = load_settings("configs/devices/macstudio.yaml")
```

The result is a Pydantic `Settings` model defined in `settings.py`. All code reads from this model; do not parse YAML directly.

### Submodule Path Resolution

`jishubench/host/config/submodules.py` provides helpers to locate submodule paths at runtime (e.g. `lmms_eval` package root). Import these rather than hardcoding paths.

---

## 7. Code Style

- **Python**: 3.12+, ruff (line-length=100, rules: E, F, I)
- **Type hints**: Encouraged for all public APIs and complex functions
- **Import sorting**: Handled by ruff (no manual `isort` needed)
- **Rust**: `cargo fmt` + `cargo clippy -- -D warnings` (both run in pre-commit)
- **Models**: Use Pydantic `BaseModel` for data classes (see `settings.py` for reference)

---

## 8. Commit Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/). The pre-commit hook enforces this.

```
<type>(<scope>): <short summary>

[optional body]
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

Examples:
```
feat(adapters): add tau-bench runner
fix(client): retry on 503 during model load
docs(quickstart): add macOS Metal setup steps
```

Breaking changes: append `!` after the type and add `BREAKING CHANGE:` footer.

---

## 9. Things to Watch Out For

- **`.env` file contains real API keys** — never commit it, never echo its contents, never include it in diffs shown to the user.
- **`outputs/` is auto-generated** — run results land here. Do not manually create or edit files in `outputs/`. They follow the `outputs/experiments-*/runs/<benchmark>/<model>__<timestamp>/` convention.
- **`submodules/` are git submodules** — changes to `submodules/lmms-eval/` or `submodules/tau2-bench/` must be committed to their own repos. Make sure to inform the user if you need to touch them.
- **Never install system packages** — the project has no system dependencies beyond Python, uv, Rust, and Docker (Docker only needed for terminal-bench). Do not suggest `apt install` or `brew install` without user approval.
- **`harbor` conflicts with `tau` / `essentials`** — `litellm` version incompatibility is enforced in `pyproject.toml`. Do not try to work around it.
- **Pre-commit runs ruff + mypy + Rust checks** — always run `uv run pre-commit run --all-files` before committing. Fix all issues.
- **Device configs reference real edge hardware** — `target.inference_base_url` points to actual edge devices (not localhost in production). Do not change device configs unless the user asks.
- **Offline edge deployments** — datasets can be pre-downloaded via `uv run jishubench download <task>`. The `--check` flag verifies local cache without network access.
