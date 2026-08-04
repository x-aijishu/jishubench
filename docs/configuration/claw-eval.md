# Claw-Eval

Claw-Eval runs the upstream agent harness on the **Host** and points the evaluated agent model at the **Target** OpenAI-compatible endpoint. jishubench generates a run-local Claw-Eval config, calls the upstream batch handler in-process, and archives Claw-Eval traces under the jishubench run directory.

Install (includes `docker` for sandbox mode via `claw-eval[sandbox]`):

```bash
uv sync --extra claw-eval
# or: uv sync --extra essentials
```

Prepare fixtures before the first run:

```bash
uv run jishubench download claw-eval
uv run jishubench download claw-eval --check
```

| Key | Default | Description |
|-----|---------|-------------|
| `claw_eval.tasks_dir` | `submodules/claw-eval/tasks` | Upstream task directory |
| `claw_eval.config_template` | `null` | Optional Claw-Eval config template; when null, uses `config_general.yaml` |
| `claw_eval.trials` | `3` | Trials per task; headline metric is Pass^k as `pass_hat_{k}` (e.g. `pass_hat_3`) |
| `claw_eval.parallel` | `1` | Batch workers on the Host |
| `claw_eval.sandbox` | `true` | Run Claw-Eval sandbox tools inside Docker containers |
| `claw_eval.sandbox_image` | `claw-eval-agent:latest` | Docker sandbox image |
| `claw_eval.sandbox_tools` | `false` | Inject sandbox tools without Docker, for development only |
| `claw_eval.no_judge` | `false` | Disable judge scoring |
| `claw_eval.text_only` | `true` | Apply Claw-Eval text-only split filtering |
| `claw_eval.filter` | `null` | Task id/path substring filter |
| `claw_eval.tag` | `null` | Task tag filter |
| `claw_eval.range` | `null` | Numeric T-task range, e.g. `1-104` |
| `claw_eval.language` | `null` | Prompt language filter, e.g. `zh` or `en` |
| `claw_eval.category` | `null` | Task category filter |
| `claw_eval.continue_existing` | `false` | Continue from the current run trace directory |
| `claw_eval.rerun_errors` | `null` | Existing trace directory for Claw-Eval `rerun-errors` |
| `claw_eval.api_key_env` | `JISHU_TARGET_API_KEY` | Temporary env var used in run-local config for Target key expansion |
| `claw_eval.user_agent_model_url` | `""` | User-simulator LLM base URL; empty falls back to `judge.openai_api_base` |
| `claw_eval.user_agent_model` | `""` | User-simulator model id; empty falls back to `judge.model` |

Judge credentials come from the shared `judge` block (`judge.openai_api_key` / `judge.openai_api_base` / `judge.model`), not from `claw_eval`.

## User-agent channel (multi-turn C* tasks)

Claw-Eval multi-turn tasks (`C*`) use a simulated user agent — a separate LLM that plays the user role. By default this agent shares the judge credentials (`judge.openai_api_key` / `judge.openai_api_base` / `judge.model`). When the fallback is active, the run log prints:

```
user_agent_model: using judge fallback  hint: set CLAW_USER_API_KEY or ...
```

To use a dedicated endpoint for the user simulator, set `CLAW_USER_API_KEY` (or `DEEPSEEK_API_KEY`) in the environment and optionally configure `claw_eval.user_agent_model_url` / `claw_eval.user_agent_model`:

```bash
# .env — dedicated user-agent key (read from environment, not YAML)
CLAW_USER_API_KEY=sk-...
```

```yaml
# configs/devices/<machine>.yaml — dedicated user-agent endpoint
claw_eval:
  user_agent_model_url: https://openrouter.ai/api/v1
  user_agent_model: google/gemini-3-flash-preview
```

Key resolution order (same pattern as tau-bench): `CLAW_USER_API_KEY` > `DEEPSEEK_API_KEY` > `judge.openai_api_key`.

Commands:

```bash
uv run jishubench preflight --benchmark claw-eval
uv run jishubench run claw-eval --limit 3
uv run jishubench run claw-eval --task-id T001zh_email_triage --no-judge
```

When `sandbox=true`, preflight checks Docker availability and the configured sandbox image. Build from the repo root:

```bash
uv run claw-eval build-image --config submodules/claw-eval/config_general.yaml
```

See also: [Configuration overview](overview.md) for shared `target`, `eval`, and logging settings.
