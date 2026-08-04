# API keys

jishubench uses separate credential channels. Target inference credentials should not be committed to YAML; judge/user-simulator credentials are Host-side and isolated from Target model serving.

| Channel | Config keys | Used for |
|---------|-------------|----------|
| **Target** | `target.inference_api_key`, optional `target.llama_cpp.api_key` | Benchmark inference (`/v1/*`) and llama.cpp control plane (`/health`, `/models`, …) |
| **Judge** | `judge.openai_api_key` (+ `judge.openai_api_base`) | LLM-as-judge scoring on the Host during lmms-eval and Claw-Eval runs |
| **User-agent** | `CLAW_USER_API_KEY` / `DEEPSEEK_API_KEY` env, optional `claw_eval.user_agent_model_url` / `claw_eval.user_agent_model` | Simulated user LLM for Claw-Eval multi-turn (C*) tasks; falls back to Judge when not set |

Do **not** commit secrets in YAML. Omit key fields from personal config files and inject them via `.env` or the shell instead.

## Injection methods

All three methods map to the same settings fields. **Environment variables and `.env` override defaults, but not an explicit value already present in the YAML file.** To inject a secret while using `--config`, leave the key out of that YAML file (as in `configs/devices/user-local.yaml`).

**1. Repo-root `.env` (recommended for local secrets)**

```bash
# .env — gitignored; loaded automatically by pydantic-settings
TARGET__INFERENCE_API_KEY=your-llama-server-key
JUDGE__OPENAI_API_KEY=sk-...
```

**2. Shell environment variables**

```bash
export TARGET__INFERENCE_API_KEY=your-llama-server-key
export JUDGE__OPENAI_API_KEY=sk-...
uv run jishubench preflight --config configs/devices/user-local.yaml
```

**3. YAML config file**

```yaml
target:
  inference_api_key: your-llama-server-key

judge:
  openai_api_key: sk-...
```

Use this only for non-secret placeholders (e.g. `EMPTY`) or locked-down deployment configs.

## Environment variable names

| Setting | Env var |
|---------|---------|
| `target.inference_api_key` | `TARGET__INFERENCE_API_KEY` |
| `target.llama_cpp.api_key` | `TARGET__LLAMA_CPP__API_KEY` |
| `judge.openai_api_key` | `JUDGE__OPENAI_API_KEY` |
| `judge.openai_api_base` | `JUDGE__OPENAI_API_BASE` |
| `judge.model` | `JUDGE__MODEL` |

## Target inference key

Default: `EMPTY` (no auth — typical for a trusted LAN llama-server).

| Consumer | How the key is applied |
|----------|------------------------|
| lmms-eval `async_openai` | Passed per-run as `api_key=…` in model args — **not** via global `OPENAI_API_KEY` |
| Claw-Eval agent model | Temporarily exposed as `JISHU_TARGET_API_KEY` (or `claw_eval.api_key_env`) only while calling the upstream batch handler; the run-local config stores `${JISHU_TARGET_API_KEY}`, not the secret |
| `LlamaCppClient` | OpenAI SDK `api_key=` for the `/v1/models` reachability check |
| `LlamaCppControlPlane` | `Authorization: Bearer …` on control-plane requests when the resolved key is non-empty and not `EMPTY` |

Resolution order for the control plane: `target.llama_cpp.api_key` if set, otherwise `target.inference_api_key`.

## Judge key

When `judge.openai_api_key` is set, jishubench temporarily applies judge credentials inside the relevant Host-side run context, then restores the previous process environment afterward. If the key is empty, no judge env vars are set.

For lmms-eval, this means `OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_API_URL`, and `MODEL_VERSION` inside `judge_env` / `lmms_eval_env`.

For Claw-Eval, the adapter maps `judge.openai_api_key` to `OPENROUTER_API_KEY` / `OPENAI_API_KEY` for the duration of the in-process batch call, unless judge scoring is disabled with `--no-judge` or `claw_eval.no_judge: true`.

For DeepSeek as judge (e.g. HallusionBench GPT scoring), set the API base and model explicitly — DeepSeek rejects the default `gpt-4` model id:

```bash
JUDGE__OPENAI_API_KEY=sk-...
JUDGE__OPENAI_API_BASE=https://api.deepseek.com
JUDGE__MODEL=deepseek-v4-flash
```

## User-agent key (Claw-Eval multi-turn)

Claw-Eval multi-turn tasks (`C*`) need a user-simulator LLM. By default it reuses the judge channel. To use a dedicated endpoint, set `CLAW_USER_API_KEY` (or `DEEPSEEK_API_KEY`) in the environment and optionally `claw_eval.user_agent_model_url` / `claw_eval.user_agent_model` in YAML.

Key resolution order (same pattern as tau-bench `TAU_USER_API_KEY`):

1. `CLAW_USER_API_KEY` (env / `.env`)
2. `DEEPSEEK_API_KEY` (env / `.env`)
3. `judge.openai_api_key` (fallback)

When the fallback is active, the run log prints a reminder. See [Claw-Eval](claw-eval.md) for the config fields.

## llama-server with `--api-key`

When the Target runs llama-server with authentication:

```bash
llama-server --host 0.0.0.0 --port 8080 --api-key your-llama-server-key ...
```

Configure the Host to send the same value (prefer `.env`):

```bash
TARGET__INFERENCE_API_KEY=your-llama-server-key
```

The same key covers both `/v1/*` inference and native control-plane endpoints on the same server.
