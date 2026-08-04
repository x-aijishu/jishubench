# CLI reference

Entry point: `jishubench` (installed via `uv sync`).

Common leaf option:

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `configs/devices/macstudio.yaml` | Path to device profile YAML (`configs/base.yaml` is merged automatically) |

When used explicitly, place `--config` after the concrete command and any required
positional arguments. It can appear before or after other options:

```bash
uv run jishubench target status --config configs/devices/user-local.yaml
uv run jishubench run mmmu_val --config configs/devices/user-local.yaml --model Qwen
uv run jishubench run mmmu_val --model Qwen --config configs/devices/user-local.yaml
```

Config path resolution accepts any of these forms:

- full or relative path to a YAML file, e.g. `configs/devices/user-local.yaml`
- shared profile name under `configs/devices/`, e.g. `user-local`
- shared profile filename, e.g. `user-local.yaml`

When `--config` is omitted, `jishubench` resolves the active profile in this order:

1. `JISHU_BENCH_CONFIG` environment variable
2. personal default selected by `jishubench config use`
3. repository fallback `configs/devices/macstudio.yaml`

## `config`

Select and inspect device profiles. These commands manage the personal default
profile used when other commands omit `--config`.

```bash
uv run jishubench config SUBCOMMAND
```

`config` management commands do not accept `--config`; they operate on the shared
profiles in `configs/devices/` and the personal selection file
`~/.jishubench/profile`.

### `config list`

List shared device profiles discovered under `configs/devices/`.

```bash
uv run jishubench config list
```

Each line prints the profile name, an active marker when applicable, the target
inference URL, and the configured model:

```text
macstudio *	http://127.0.0.1:8080/v1  model=Qwen3.5-4B-Q4_K_M
user-local	http://127.0.0.1:8080/v1  model=Qwen
```

The `*` marks the active profile. If no personal default has been selected, the
fallback profile can still appear as active because it is the profile that would
be used by commands that omit `--config`.

### `config show`

Print the resolved active device profile path.

```bash
uv run jishubench config show
```

This uses the same default resolution as other commands when `--config` is
omitted: `JISHU_BENCH_CONFIG`, then `~/.jishubench/profile`, then
`configs/devices/macstudio.yaml`. It prints the absolute path to the selected
YAML file and exits non-zero if the selected profile cannot be found.

### `config use`

Save a shared device profile as the personal default.

```bash
uv run jishubench config use PROFILE
```

`PROFILE` is the name of a YAML file under `configs/devices/`, without the
`.yaml` suffix:

```bash
uv run jishubench config use macstudio
uv run jishubench config use user-local
```

On success, the command writes only the profile name to `~/.jishubench/profile`
and prints the resolved path. It does not copy or modify the YAML file. After
that, commands can omit `--config`:

```bash
uv run jishubench preflight
uv run jishubench run mmmu_val --limit 10
uv run jishubench target status
```

If `PROFILE` is omitted in an interactive terminal, `config use` shows a numbered
menu of available profiles and prompts for a selection:

```bash
uv run jishubench config use
```

In non-interactive mode, `PROFILE` is required. Passing an unknown profile exits
non-zero and prints the available profile names.

## `preflight`

Check submodule readiness and Target connectivity.

```bash
uv run jishubench preflight
```

Prints JSON status: submodules, inference reachability, monitor reachability, warnings.

Benchmark-specific preflight checks are available for agent backends:

```bash
uv run jishubench preflight --benchmark tau-bench
uv run jishubench preflight --benchmark terminal-bench
uv run jishubench preflight --benchmark terminal-bench-2
uv run jishubench preflight --benchmark claw-eval
```

## `benchmarks`

List lmms-eval benchmark names (no Target required).

```bash
uv run jishubench benchmarks [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--kind` | `subtasks` (default), `groups`, `tags`, or `all` |
| `--search QUERY` | Case-insensitive substring filter |
| `--json` | Full catalog as JSON (default: one name per line) |

## `download`

Pre-download benchmark datasets to configured cache directories (no Target required).

```bash
uv run jishubench download BENCHMARK [BENCHMARK ...] [OPTIONS]
```

| Argument / option | Description |
|-------------------|-------------|
| `--backend NAME` | Dataset backend override (default: infer agent backend objects, otherwise `lmms-eval`) |
| `BENCHMARK` | Task names (same as `run`, e.g. `mmmu_val`, `realworldqa`) or an agent backend object (`claw-eval`, `tau-bench`, `terminal-bench`, `terminal-bench-2`) |
| `--hub-cache PATH` | Override `lmms_eval.hub_cache` (`lmms-eval` backend only) |
| `--datasets-cache PATH` | Override `lmms_eval.datasets_cache` (`lmms-eval` backend only) |
| `--check` | Verify local cache only; do not download from the network |
| `--force` | Force re-download where the backend supports it |
| `-v`, `--verbose` | Full traceback on failure |

Examples:

```bash
uv run jishubench download mmmu_val realworldqa
uv run jishubench download --backend lmms-eval mmmu_val
uv run jishubench download tau-bench
uv run jishubench download terminal-bench
uv run jishubench download terminal-bench-2
uv run jishubench download claw-eval
uv run jishubench download claw-eval --check
uv run jishubench download mmmu_val --config configs/devices/user-local.yaml
uv run jishubench download mmmu_val --check
```

Gated Hugging Face datasets require `HF_TOKEN` in the environment. Use the same `--config` as `run` so cache paths stay aligned.

For agent backends, the backend name is the download object: `tau-bench` verifies submodule data, `terminal-bench` prepares the configured Terminal-Bench v0.1.1 dataset, `terminal-bench-2` prepares the Terminal-Bench 2.0 (Harbor) dataset, and `claw-eval` fetches `data/*` from the Hugging Face dataset and extracts fixtures into `submodules/claw-eval/tasks`. `--check` is offline where supported.

## `run`

Run one or more benchmarks against the Target inference server. Supports lmms-eval tasks and agent benchmarks (`terminal-bench`, `terminal-bench-2`, `tau-bench`, `claw-eval`).

```bash
uv run jishubench run BENCHMARK [BENCHMARK ...] [OPTIONS]
```

| Argument / option | Description |
|-------------------|-------------|
| `BENCHMARK` | Task names (e.g. `mmmu_val`, `realworldqa`) or agent benchmarks (`terminal-bench`, `terminal-bench-2`, `tau-bench`, `claw-eval`) |
| `--model` | Model on Target (default: config `target.model`) |
| `--limit N` | Limit samples per task / number of agent tasks |
| `--task-id ID` | Run specific task ID(s) only; repeatable (agent benchmarks) |
| `--domain NAME` | Tau-bench domain(s) to run (e.g. `airline`, `retail`); repeatable |
| `--no-judge` | Disable judge scoring where supported (`claw-eval`) |
| `--claw-filter QUERY` | Claw-Eval task id/path substring filter |
| `--claw-tag TAG` | Claw-Eval task tag filter |
| `--claw-range L-R` | Claw-Eval numeric T-task range, e.g. `1-104` |
| `--claw-language LANG` | Claw-Eval prompt language filter, e.g. `zh` or `en` |
| `--claw-category CATEGORY` | Claw-Eval task category filter |
| `-v`, `--verbose` | Full traceback on failure |

Examples (lmms-eval):

```bash
uv run jishubench run mmmu_val --model Qwen3.5-4B-Q4_K_M
uv run jishubench run mmmu_val realworldqa --limit 10
```

Examples (Claw-Eval):

```bash
uv run jishubench run claw-eval --limit 3
uv run jishubench run claw-eval --task-id T001zh_email_triage --no-judge
uv run jishubench run claw-eval --claw-language zh --claw-category finance --limit 5
uv run jishubench run claw-eval --claw-tag multimodal
```

Claw-Eval runs the upstream batch handler in-process on the Host. Run artifacts stay under the jishubench run directory, including `claw_eval_config.yaml`, stdout/stderr logs, `traces/**/batch_summary.json`, `batch_results.json`, and JSONL traces.

### Run output

Each successful `run` writes `summary.json` under the run directory and prints the same headline fields to stdout. Metrics keep the **upstream name and formula**; they are not renamed to a generic `accuracy`.

| Field | Meaning |
|-------|---------|
| `metric` | Headline metric name when `value` is a scalar; `null` for multi-task runs |
| `value` | Scalar score, or multi-task map `{task: {"metric": name, "value": score}}` |

| Engine | Typical `metric` | How `value` is computed |
|--------|------------------|-------------------------|
| `lmms_eval` | Task YAML headline (e.g. `exact_match`, `average`, `acc`) | lmms-eval primary metric for that task |
| `terminal_bench` | `accuracy` | Upstream resolve rate (`n_resolved / n_total`) |
| `terminal_bench_2` | `accuracy` | Harbor trial resolve rate (`n_resolved / n_total`), pass@k across attempts |
| `tau_bench` | `pass_hat` | Pass^k: fraction of tasks where every trial succeeds |
| `claw_eval` | `pass_hat_{trials}` (e.g. `pass_hat_3`) | Upstream `pass_hat_k / tasks` from `batch_summary.json` |

Example (single lmms-eval task):

```json
{
  "run_id": "20260716_030419",
  "success": true,
  "engine": "lmms_eval",
  "metric": "exact_match",
  "value": 0.77
}
```

Example (multi-task lmms-eval):

```json
{
  "metric": null,
  "value": {
    "mmmu_val": {"metric": "exact_match", "value": 0.72},
    "mmstar": {"metric": "average", "value": 0.62}
  }
}
```

Supported agent benchmarks:

| `AGENT` | Description |
|---------|-------------|
| `terminal-bench` | Terminal-Bench v0.1.1 Docker harness via `tb` Python API (requires Docker). |
| `terminal-bench-2` | Terminal-Bench 2.0 via Harbor CLI (`jishubench[harbor]`); not-tested. See [Terminal-Bench 2.0 (Harbor)](configuration/terminal-bench-2.md). |
| `tau-bench` | Tau-bench in-process simulation (agent → Target, user → remote LLM) |

### `run terminal-bench`

> Terminal-Bench v0.1.1 Docker harness via `tb` (requires `--extra terminal` and Docker).

> Terminal-Bench Docker images are built for `linux/amd64` (x86-64).

```bash
uv run jishubench run terminal-bench --limit 5
```

### `run terminal-bench-2`

> Terminal-Bench 2.0 via the Harbor CLI (requires `--extra harbor` and Docker). Not-tested.

> Harbor conflicts with `tau` / `essentials` (litellm). Use a dedicated environment.

```bash
uv run jishubench run terminal-bench-2 --limit 5
uv run jishubench run terminal-bench-2 --task-id <task-name>
```

See [Terminal-Bench 2.0 (Harbor)](configuration/terminal-bench-2.md) for task-format and reward differences from TB v0.1.1.

### `run tau-bench`

> Requires `--extra tau` and a user-simulator API key (`TAU_USER_API_KEY` or `DEEPSEEK_API_KEY`).

```bash
export DEEPSEEK_API_KEY=sk-...
uv run jishubench run tau-bench --limit 2
uv run jishubench run tau-bench --domain airline --domain retail --limit 5
uv run jishubench run tau-bench --task-id 42 --model Qwen3.5-4B-Q4_K_M
```

## `target`

Manage the Target llama.cpp router server.

### `target status`

Model status, slot availability, and readiness for the configured (or given) model.

```bash
uv run jishubench target status [--model NAME]
```

### `target models`

List models from `GET /models`.

```bash
uv run jishubench target models [--reload] [--json]
```

| Option | Description |
|--------|-------------|
| `--reload` | Rescan models-dir / cache (`?reload=1`) |
| `--json` | Full catalog as JSON |

### `target load` / `target unload`

```bash
uv run jishubench target load [--model NAME]
uv run jishubench target unload [--model NAME]
```

`target load` defaults to `target.model` from the active device profile.

`target unload` without `--model` infers the target from the live Target
router state:

- if exactly one model is `loaded` on the Target, that model is unloaded;
- if no model is `loaded`, the command exits with a message and suggests
  `target load --model NAME`;
- if more than one model is `loaded`, the command lists them and asks you to
  pass `--model NAME` to disambiguate.

Pass `--model NAME` to unload a specific model regardless of current state.
