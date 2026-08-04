"""Runtime configuration for Host-Target evaluation."""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    InitSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Placeholder for local OpenAI-compatible servers (llama.cpp, Ollama, vLLM).
LOCAL_TARGET_API_KEY = "EMPTY"
_YAML_SETTINGS_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "yaml_settings_context", default=None
)


# ---------------------------------------------------------------------------
# llama.cpp Router config sub-models
# ---------------------------------------------------------------------------


class LlamaCppLifecycleConfig(BaseModel):
    ensure_loaded: bool = True
    autoload_fallback: bool = True
    wait_load_timeout_s: float = 300.0
    poll_interval_s: float = 2.0
    unload_after_run: bool = False
    unload_on_model_switch: bool = True


class LlamaCppPreflightConfig(BaseModel):
    check_health: bool = True
    check_model_catalog: bool = True


class LlamaCppConfig(BaseModel):
    mode: Literal["router"] = "router"
    api_key: str = ""

    lifecycle: LlamaCppLifecycleConfig = Field(default_factory=LlamaCppLifecycleConfig)
    preflight: LlamaCppPreflightConfig = Field(default_factory=LlamaCppPreflightConfig)

    # Friendly name → real router model id (e.g. "Qwen3.5-4B-Q4_K_M" →
    # "Qwen/Qwen3.5-4B-GGUF:Q4_K_M").  Optional; identity if absent.
    model_aliases: dict[str, str] = Field(default_factory=dict)
    # Router model id → mmproj catalog id or filesystem path for VLM pairing.
    model_mmproj: dict[str, str] = Field(default_factory=dict)

    def resolve_model_id(self, name: str) -> str:
        return self.model_aliases.get(name, name)

    def resolved_api_key(self, fallback: str = "") -> str:
        return self.api_key.strip() or fallback


# ---------------------------------------------------------------------------
# Main config models
# ---------------------------------------------------------------------------


class TargetConfig(BaseModel):
    inference_base_url: str = "http://127.0.0.1:8080/v1"
    inference_api_key: str = LOCAL_TARGET_API_KEY
    model: str = "Qwen3.5-4B"
    monitor_base_url: str = "http://127.0.0.1:9090"
    monitor_timeout_s: float = 5.0
    llama_cpp: LlamaCppConfig | None = None

    def resolved_inference_api_key(self) -> str:
        key = self.inference_api_key.strip()
        return key or LOCAL_TARGET_API_KEY

    def inference_root_url(self) -> str:
        """llama.cpp control-plane root (inference_base_url without /v1)."""
        return self.inference_base_url.rstrip("/").removesuffix("/v1")


class JudgeConfig(BaseModel):
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    # LLM-as-judge model id (exported as MODEL_VERSION for lmms-eval tasks).
    model: str = "gpt-4o"


class EvalConfig(BaseModel):
    work_dir: Path = REPO_ROOT / "outputs"
    runs_subdir: str = "runs"  # relative subdir under work_dir for per-run dirs
    shared_subdir: str = "shared"  # relative subdir under work_dir for shared data
    api_nproc: int = 4
    enable_traces: bool = True
    save_samples: bool = False


class LmmsEvalConfig(BaseModel):
    model_backend: str = "async_openai"
    batch_size: int = 1
    adaptive_concurrency: bool = True
    datasets_cache: Path | None = None
    hub_cache: Path | None = None


def _default_terminal_bench_agent_kwargs() -> dict[str, object]:
    return {
        "temperature": 0.0,
        "parser_name": "json",
    }


def _default_terminal_bench_2_agent_kwargs() -> dict[str, object]:
    return {"temperature": 0.0}


class TerminalBenchConfig(BaseModel):
    """Terminal-Bench v0.1.1 agent benchmark settings (Host-side `tb` Docker harness)."""

    agent: str = "terminus-2"
    dataset: str = "terminal-bench-core==0.1.1"
    n_concurrent: int = 1
    n_attempts: int = 1
    max_episodes: int | None = None
    global_timeout_multiplier: float = 1.0
    global_agent_timeout_sec: float | None = None
    global_test_timeout_sec: float | None = None
    cache_dir: Path | None = None
    agent_kwargs: dict[str, object] = Field(default_factory=_default_terminal_bench_agent_kwargs)


class TerminalBench2Config(BaseModel):
    """Terminal-Bench 2.0 agent benchmark settings (Host-side Harbor harness).

    Harbor is the official harness for Terminal-Bench 2.0 (TB 2.0+). It is a
    ground-up rewrite of the legacy ``tb`` harness with a different task format
    (``task.toml`` + ``instruction.md`` + ``environment/``), reward scheme
    (``reward.txt`` instead of a parser), and CLI (``harbor run``). TB 2.0 is
    fully decoupled from TB v0.1.1 — see ``docs/configuration/terminal-bench-2.md``.
    """

    agent: str = "terminus-2"
    dataset: str = "terminal-bench@2.0"
    n_concurrent: int = 1
    n_attempts: int = 1
    max_episodes: int | None = None
    global_timeout_multiplier: float = 1.0
    global_agent_timeout_sec: float | None = None
    global_test_timeout_sec: float | None = None
    cache_dir: Path | None = None
    agent_kwargs: dict[str, object] = Field(default_factory=_default_terminal_bench_2_agent_kwargs)


class TauBenchConfig(BaseModel):
    """Tau-Bench agent benchmark settings (in-process tau2 runner)."""

    domains: list[str] = Field(default_factory=lambda: ["airline", "retail", "telecom", "mock"])
    split: str = "base"
    num_tasks: int | None = None
    repeats: int = 1
    max_steps: int = 100
    n_concurrent: int = 1
    verbose_logs: bool = True
    # Agent (被测模型) — 默认继承 target 温度 0
    agent_temperature: float = 0.0
    agent_max_tokens: int = 2048
    # User simulator — 远程 LLM, key via env
    user_model_url: str = "https://api.deepseek.com"
    user_model: str = "deepseek-v4-flash"
    user_temperature: float = 0.0
    user_max_tokens: int = 4096


class ClawEvalConfig(BaseModel):
    """Claw-Eval agent benchmark settings (external Host sandbox harness)."""

    tasks_dir: Path = REPO_ROOT / "submodules" / "claw-eval" / "tasks"
    config_template: Path | None = None
    trials: int = 3
    parallel: int = 1
    sandbox: bool = True
    sandbox_image: str = "claw-eval-agent:latest"
    sandbox_tools: bool = False
    no_judge: bool = False
    text_only: bool = True
    filter: str | None = None
    tag: str | None = None
    range: str | None = None
    language: str | None = None
    category: str | None = None
    continue_existing: bool = False
    rerun_errors: Path | None = None
    api_key_env: str = "JISHU_TARGET_API_KEY"
    # User-agent (simulated user for C* multi-turn tasks) — empty falls back to judge.
    user_agent_model_url: str = ""
    user_agent_model: str = ""


class LoggingConfig(BaseModel):
    level: str = "INFO"
    save_dir: Path | None = None
    max_bytes: int = 10_485_760
    backup_count: int = 5

    def resolved_save_dir(self) -> Path | None:
        if self.save_dir is None:
            return None
        resolved = self.save_dir.expanduser()
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        return resolved


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    target: TargetConfig = Field(default_factory=TargetConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    lmms_eval: LmmsEvalConfig = Field(default_factory=LmmsEvalConfig)
    terminal_bench: TerminalBenchConfig = Field(default_factory=TerminalBenchConfig)
    terminal_bench_2: TerminalBench2Config = Field(default_factory=TerminalBench2Config)
    tau_bench: TauBenchConfig = Field(default_factory=TauBenchConfig)
    claw_eval: ClawEvalConfig = Field(default_factory=ClawEvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_settings = _YAML_SETTINGS_CONTEXT.get()
        if yaml_settings is None:
            return init_settings, env_settings, dotenv_settings, file_secret_settings
        yaml_source = InitSettingsSource(settings_cls, yaml_settings)
        return init_settings, env_settings, dotenv_settings, file_secret_settings, yaml_source

    @classmethod
    def from_yaml(cls, path: Path | str) -> Settings:
        config_path = Path(path)
        base = _read_yaml_dict(BASE_CONFIG)
        device = _read_yaml_dict(config_path)
        merged = _deep_merge(base, device)
        token = _YAML_SETTINGS_CONTEXT.set(merged)
        try:
            return cls()
        finally:
            _YAML_SETTINGS_CONTEXT.reset(token)

    def ensure_work_dir(self) -> Path:
        self.eval.work_dir.mkdir(parents=True, exist_ok=True)
        return self.eval.work_dir

    def runs_dir(self) -> Path:
        """Return the root directory for all per-run output dirs (``work_dir/runs/``)."""
        path = self.ensure_work_dir() / self.eval.runs_subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def shared_dir(self) -> Path:
        """Return the directory for cross-run shared data (``work_dir/shared/``)."""
        path = self.ensure_work_dir() / self.eval.shared_subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_dir(self, benchmark: str, model_dir_name: str) -> Path:
        """Create and return the self-contained run directory: runs/<benchmark>/<model_name>/."""
        path = self.runs_dir() / benchmark / model_dir_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def build_lmms_model_args(self, model_name: str | None = None) -> str:
        """Build non-secret async_openai arguments safe to persist in results."""
        model = model_name or self.target.model
        parts = [
            f"model_version={model}",
            f"base_url={self.target.inference_base_url.rstrip('/')}",
            f"num_cpus={self.eval.api_nproc}",
        ]
        if self.lmms_eval.adaptive_concurrency:
            parts.append("adaptive_concurrency=true")
        return ",".join(parts)

    def build_lmms_model_kwargs(self, model_name: str | None = None) -> dict[str, object]:
        """Build runtime async_openai arguments, including the in-memory credential."""
        model = model_name or self.target.model
        return {
            "model_version": model,
            "base_url": self.target.inference_base_url.rstrip("/"),
            "api_key": self.target.resolved_inference_api_key(),
            "num_cpus": self.eval.api_nproc,
            "adaptive_concurrency": self.lmms_eval.adaptive_concurrency,
        }

    def datasets_cache_env_for_upstream(self) -> dict[str, str]:
        """Env vars for lmms-eval Hugging Face datasets cache (processed Arrow files)."""
        cache = self.lmms_eval.datasets_cache
        if cache is None:
            return {}
        return {"LMMS_EVAL_DATASETS_CACHE": str(cache.expanduser().resolve())}

    def hub_cache_env_for_upstream(self) -> dict[str, str]:
        """Env vars for Hugging Face Hub downloads (raw parquet/zip from Hub)."""
        cache = self.lmms_eval.hub_cache
        if cache is None:
            return {}
        return {"HF_HUB_CACHE": str(cache.expanduser().resolve())}

    def judge_env_for_upstream(self) -> dict[str, str]:
        """Env vars for LLM-as-judge on Host (separate from Target inference)."""
        key = self.judge.openai_api_key.strip()
        if not key:
            return {}
        base = self.judge.openai_api_base.rstrip("/")
        env: dict[str, str] = {
            "OPENAI_API_KEY": key,
            "OPENAI_API_BASE": base,
            "OPENAI_API_URL": f"{base}/chat/completions",
        }
        model = self.judge.model.strip()
        if model:
            env["MODEL_VERSION"] = model
        return env

    def lmms_eval_env_for_upstream(self) -> dict[str, str]:
        """Env vars applied for the full lmms-eval run (HF caches + judge)."""
        return {
            **self.hub_cache_env_for_upstream(),
            **self.datasets_cache_env_for_upstream(),
            **self.judge_env_for_upstream(),
        }

    def resolve_terminal_bench_model_id(self, model_name: str | None = None) -> str:
        """LiteLLM model id for Terminus (openai/ prefix for compatible endpoints)."""
        model = model_name or self.target.model
        llama = self.target.llama_cpp
        if llama is not None:
            model = llama.resolve_model_id(model)
        return model

    def terminal_bench_litellm_model(self, model_name: str | None = None) -> str:
        return f"openai/{self.resolve_terminal_bench_model_id(model_name)}"

    def resolved_terminal_bench_api_base(self) -> str:
        override = self.terminal_bench.agent_kwargs.get("api_base")
        if isinstance(override, str) and override.strip():
            return override.rstrip("/")
        return self.target.inference_base_url.rstrip("/")

    def terminal_bench_runs_dir(self) -> Path:
        """Root runs directory (engine layer removed; benchmark dirs are direct children)."""
        return self.runs_dir()

    def terminal_bench_cache_path(self) -> Path:
        cache = self.terminal_bench.cache_dir
        if cache is not None:
            resolved = cache.expanduser()
            if not resolved.is_absolute():
                resolved = REPO_ROOT / resolved
            return resolved
        return self.shared_dir() / "cache" / "terminal_bench_cache.jsonl"

    def terminal_bench_env_for_subprocess(self) -> dict[str, str]:
        """Env vars for tb subprocess only (not process-wide)."""
        env: dict[str, str] = {}
        key = self.target.resolved_inference_api_key()
        if key:
            # litellm requires a non-empty key even for local OpenAI-compatible
            # servers; the LOCAL_TARGET_API_KEY placeholder satisfies that.
            env["OPENAI_API_KEY"] = key
        return env

    def merged_terminal_bench_agent_kwargs(
        self, model_name: str | None = None
    ) -> dict[str, object]:
        kwargs = dict(self.terminal_bench.agent_kwargs)
        kwargs.setdefault("api_base", self.resolved_terminal_bench_api_base())
        kwargs["model_name"] = self.terminal_bench_litellm_model(model_name)
        if self.terminal_bench.max_episodes is not None:
            kwargs["max_episodes"] = self.terminal_bench.max_episodes
        return kwargs

    # ------------------------------------------------------------------
    # Terminal-Bench 2.0 (Harbor) helpers
    # ------------------------------------------------------------------

    def resolve_terminal_bench_2_model_id(self, model_name: str | None = None) -> str:
        """LiteLLM model id for Harbor/Terminus-2 (openai/ prefix for compatible endpoints)."""
        model = model_name or self.target.model
        llama = self.target.llama_cpp
        if llama is not None:
            model = llama.resolve_model_id(model)
        return model

    def terminal_bench_2_litellm_model(self, model_name: str | None = None) -> str:
        return f"openai/{self.resolve_terminal_bench_2_model_id(model_name)}"

    def resolved_terminal_bench_2_api_base(self) -> str:
        override = self.terminal_bench_2.agent_kwargs.get("api_base")
        if isinstance(override, str) and override.strip():
            return override.rstrip("/")
        return self.target.inference_base_url.rstrip("/")

    def terminal_bench_2_runs_dir(self) -> Path:
        """Root runs directory for Terminal-Bench 2.0 (Harbor) runs."""
        return self.runs_dir()

    def terminal_bench_2_cache_path(self) -> Path:
        cache = self.terminal_bench_2.cache_dir
        if cache is not None:
            resolved = cache.expanduser()
            if not resolved.is_absolute():
                resolved = REPO_ROOT / resolved
            return resolved
        return self.shared_dir() / "cache" / "terminal_bench_2_cache.jsonl"

    def terminal_bench_2_env_for_subprocess(self) -> dict[str, str]:
        """Env vars for harbor subprocess only (not process-wide)."""
        env: dict[str, str] = {}
        key = self.target.resolved_inference_api_key()
        if key:
            env["OPENAI_API_KEY"] = key
        return env

    def merged_terminal_bench_2_agent_kwargs(
        self, model_name: str | None = None
    ) -> dict[str, object]:
        kwargs = dict(self.terminal_bench_2.agent_kwargs)
        kwargs.setdefault("api_base", self.resolved_terminal_bench_2_api_base())
        kwargs["model_name"] = self.terminal_bench_2_litellm_model(model_name)
        if self.terminal_bench_2.max_episodes is not None:
            kwargs["max_episodes"] = self.terminal_bench_2.max_episodes
        return kwargs

    # ------------------------------------------------------------------
    # Tau-Bench helpers
    # ------------------------------------------------------------------

    def resolve_tau_user_api_key(self) -> str:
        """Resolve user-simulator API key: TAU_USER_API_KEY > DEEPSEEK_API_KEY > judge key."""
        import os

        for env_var in ("TAU_USER_API_KEY", "DEEPSEEK_API_KEY"):
            val = os.environ.get(env_var, "").strip()
            if val:
                return val
        return self.judge.openai_api_key.strip()

    def tau_bench_env_for_upstream(self) -> dict[str, str]:
        """Env vars for tau2 fallback evaluator/judge calls."""
        key = self.resolve_tau_user_api_key()
        if not key:
            return {}
        return {"OPENAI_API_KEY": key}

    # ------------------------------------------------------------------
    # Claw-Eval helpers
    # ------------------------------------------------------------------

    def resolve_claw_eval_model_id(self, model_name: str | None = None) -> str:
        model = model_name or self.target.model
        llama = self.target.llama_cpp
        if llama is not None:
            model = llama.resolve_model_id(model)
        return model

    def resolve_claw_eval_user_agent_api_key(self) -> str:
        """Resolve user-agent API key: CLAW_USER_API_KEY > DEEPSEEK_API_KEY > judge key."""
        import os

        for env_var in ("CLAW_USER_API_KEY", "DEEPSEEK_API_KEY"):
            val = os.environ.get(env_var, "").strip()
            if val:
                return val
        return self.judge.openai_api_key.strip()

    def claw_eval_env_for_upstream(self) -> dict[str, str]:
        env: dict[str, str] = {}
        key = self.target.resolved_inference_api_key()
        if key:
            env[self.claw_eval.api_key_env] = key
        judge_key = self.judge.openai_api_key.strip()
        if judge_key:
            env.setdefault("OPENAI_API_KEY", judge_key)
            env.setdefault("OPENROUTER_API_KEY", judge_key)
        ua_key = self.resolve_claw_eval_user_agent_api_key()
        if ua_key:
            env["CLAW_USER_API_KEY"] = ua_key
        return env


# ---------------------------------------------------------------------------
# YAML deep merge
# ---------------------------------------------------------------------------

BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* and return a new dict.

    Leaf values in *override* win.  Lists and other non-dict values in
    *override* replace the base value entirely.
    """
    merged = {}
    keys = set(base) | set(override)
    for k in keys:
        if k in override and k in base:
            bv, ov = base[k], override[k]
            if isinstance(bv, dict) and isinstance(ov, dict):
                merged[k] = _deep_merge(bv, ov)
            else:
                merged[k] = ov  # override wins for non-dict or mixed types
        elif k in override:
            merged[k] = override[k]
        else:
            merged[k] = base[k]
    return merged


def _read_yaml_dict(path: Path) -> dict:
    """Read a YAML file into a dict, returning ``{}`` if the file doesn't exist."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings(config_path: Path | str | None = None) -> Settings:
    from host.config.profiles import resolve_config_path

    try:
        resolved = resolve_config_path(config_path)
    except FileNotFoundError:
        if config_path is None:
            return Settings()
        raise
    return Settings.from_yaml(resolved)
