# Contributing to jishubench

Thank you for taking the time to contribute. This document covers how to set up
a development environment, run the test suite, and submit changes.

---

## Table of contents

- [Prerequisites](#prerequisites)
- [Getting started](#getting-started)
- [Project layout](#project-layout)
- [Running tests](#running-tests)
- [Code style](#code-style)
- [Commit conventions](#commit-conventions)
- [Submitting changes](#submitting-changes)
- [Reporting issues](#reporting-issues)

---

## Prerequisites

| Tool | Minimum version | Install |
|------|----------------|---------|
| [uv](https://docs.astral.sh/uv/) | 0.4 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Rust / Cargo | 1.78 | [rustup.rs](https://rustup.rs/) |
| Docker | 24 | required for terminal-bench / claw-eval tasks |

---

## Getting started

```bash
# 1. Fork the repository on GitLab/GitHub, then clone your fork
git clone git@<host>/<your-fork>/jishubench.git
cd jishubench

# 2. Add the upstream remote
git remote add upstream https://github.com/x-aijishu/jishubench.git

# 3. Pull in all submodules (lmms-eval, tau2-bench, claw-eval)
git submodule update --init --recursive

# 4. Install Python dependencies (dev + all backends)
uv sync --extra essentials

# 5. Install pre-commit hooks
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg

# 6. Verify the setup
uv run jishubench preflight
```

Keep your fork up to date before starting new work:

```bash
git fetch upstream
git rebase upstream/main
```

---

## Project layout

```
jishubench/
  host/                # Python: CLI, adapters, clients, config
    adapters/          # one module per backend, plus shared runner types/helpers
    client/            # llama_cpp_client.py, eval_monitor_client.py
    config/            # Pydantic settings, logging, env
    cli/               # commands.py, parser.py
    controller.py      # Run and Target lifecycle controller
  target/              # Rust: edge-eval-agent sidecar
    agent/             # HTTP API for hardware metric traces
    hal/               # Hardware abstraction layer
    monitor/           # Power / utilization probes
submodules/
  lmms-eval/           # git submodule – vision-language benchmark tasks
  tau2-bench/          # git submodule – tool-agent-user simulation
  claw-eval/           # git submodule – agent benchmark suite
docs/                  # MkDocs source
tests/                 # pytest suite
configs/               # base.yaml (shared defaults) + devices/ (machine profiles)
```

---

## Running tests

### Python tests

```bash
# Full suite
uv run pytest

# A single file
uv run pytest tests/test_cli.py -v

# With coverage
uv run pytest --cov=jishubench --cov-report=term-missing
```

Tests that require a live Target endpoint are skipped automatically when
`JISHU_TARGET__INFERENCE_BASE_URL` is not set.

### Rust tests

```bash
cd jishubench/target
cargo test
```

### Linting (all checks pre-commit runs)

```bash
uv run pre-commit run --all-files
```

---

## Code style

- **Python**: [ruff](https://docs.astral.sh/ruff/) for formatting and linting.
  Line length is 100. Run `uv run ruff format .` and `uv run ruff check --fix .`
  before committing, or let pre-commit do it.
- **Type checking**: mypy with `--ignore-missing-imports` on `jishubench/host/`.
- **Rust**: `cargo fmt` and `cargo clippy -- -D warnings`. Both run in pre-commit.
- **Docstrings**: Not required for internal helpers, appreciated for public API.

---

## Commit conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/).
The `commit-msg` hook enforces this automatically after `pre-commit install`.

**Format**

```
<type>(<scope>): <short summary>

[optional body]

[optional footer(s)]
```

**Allowed types**

| Type | When to use |
|------|-------------|
| `feat` | New user-visible feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, whitespace (no logic change) |
| `refactor` | Code restructuring without behaviour change |
| `perf` | Performance improvement |
| `test` | Adding or fixing tests |
| `build` | Build system, dependencies |
| `ci` | CI configuration |
| `chore` | Maintenance tasks |
| `revert` | Reverting a previous commit |

**Examples**

```
feat(adapters): add tau-bench runner
fix(client): retry on 503 during model load
docs(quickstart): add macOS Metal setup steps
```

Breaking changes: append `!` after the type and add a `BREAKING CHANGE:` footer.

---

## Submitting changes

1. Create a branch from `main`, for example:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make your changes. Keep commits small and focused.
3. Ensure the full test suite and pre-commit checks pass.
4. Open a Merge Request (GitLab) or Pull Request (GitHub) against `main`.
   - Fill in the MR/PR template completely.
   - Link to any related issues with `Closes #<number>`.
5. A maintainer will review within a few business days. Address review comments
   with fixup commits; squash before merge if requested.

---

## Reporting issues

See [SECURITY.md](SECURITY.md) for vulnerability disclosures.
For bugs and feature requests, use the issue templates in `.github/ISSUE_TEMPLATE/`.
