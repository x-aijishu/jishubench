# Tau-Bench

Tau-bench runs entirely **in-process** on the Host. The LLM **agent** is called against the **Target** inference endpoint; the **user simulator** is called against a remote API (DeepSeek, OpenAI, or any compatible provider).

Install the optional extra:

```bash
uv sync --extra tau
```

## API keys for tau-bench

Tau-bench uses its own key resolution chain — **separate from** the Target and Judge channels:

| Priority | Source | Description |
|----------|--------|-------------|
| 1 | `TAU_USER_API_KEY` env var | Preferred; tau-bench-specific key |
| 2 | `DEEPSEEK_API_KEY` env var | Convenient for DeepSeek default |
| 3 | `judge.openai_api_key` (config) | Fallback — reuses judge key |

The key is **never stored in YAML**. Inject it via `.env` or the shell:

```bash
export DEEPSEEK_API_KEY=sk-...
# or
export TAU_USER_API_KEY=sk-...
```

## `tau_bench` settings

| Key | Default | Description |
|-----|---------|-------------|
| `tau_bench.domains` | `[airline]` | Domain(s) to evaluate; available: `airline`, `retail`, `telecom`, `banking_knowledge` |
| `tau_bench.split` | `base` | Task split name within the domain |
| `tau_bench.num_tasks` | `10` | Max tasks per domain (`null` = all) |
| `tau_bench.repeats` | `1` | Trials per task for Pass^k; headline metric is `pass_hat` |
| `tau_bench.max_steps` | `100` | Max simulation steps before termination |
| `tau_bench.n_concurrent` | `1` | Parallel simulations on Host |
| `tau_bench.agent_temperature` | `0.0` | Temperature for agent calls to Target |
| `tau_bench.agent_max_tokens` | `2048` | Max tokens for agent calls to Target |
| `tau_bench.user_model_url` | `https://api.deepseek.com/v1` | User-simulator API base URL |
| `tau_bench.user_model` | `deepseek-chat` | User-simulator model name |
| `tau_bench.user_temperature` | `0.0` | Temperature for user-simulator calls |
| `tau_bench.user_max_tokens` | `4096` | Max tokens for user-simulator calls |

Commands:

```bash
# Preflight (submodule + tau2 import + data directory + user API key)
jishubench preflight --benchmark tau-bench

# Ensure submodule data is present
jishubench download tau-bench

# Run agent eval (Target agent + remote user simulator)
jishubench run tau-bench --limit 5

# Run specific domains
jishubench run tau-bench --domain airline --domain retail --limit 5

# Run a specific task
jishubench run tau-bench --task-id 42
```

See also: [Configuration overview](overview.md) for shared `target`, `eval`, and logging settings. For judge credentials used as fallback, see [API keys](api-keys.md).
