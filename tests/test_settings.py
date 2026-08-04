"""Tests: Settings 配置解析 + 环境变量辅助函数。"""

from __future__ import annotations

import os

from host.config import REPO_ROOT, Settings, load_settings
from host.config.settings import LlamaCppConfig, TargetConfig

# ---------------------------------------------------------------------------
# load_settings / from_yaml
# ---------------------------------------------------------------------------


def test_load_default_settings():
    """从真实的默认配置文件加载，验证关键字段存在。"""
    settings = load_settings(REPO_ROOT / "configs" / "base.yaml")
    assert settings.target.model  # non-empty
    assert settings.logging.level == "INFO"
    assert settings.lmms_eval.model_backend == "async_openai"
    assert settings.eval.work_dir  # non-empty path


def test_settings_from_yaml_malformed_or_missing_fields(tmp_path):
    """验证 from_yaml 能处理缺失可选字段和多余字段 (pydantic extra='ignore')。"""
    yaml_content = """
    target:
      inference_base_url: "http://custom:8080/v1"
      model: "custom-model"
    extra_field_should_be_ignored: true
    """
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    settings = Settings.from_yaml(yaml_path)
    assert settings.target.inference_base_url == "http://custom:8080/v1"
    assert settings.target.model == "custom-model"
    # extra_field 不存在于 Settings 模型中，应被忽略
    assert hasattr(settings, "extra_field_should_be_ignored") is False


# ---------------------------------------------------------------------------
# deep_merge
# ---------------------------------------------------------------------------


def test_deep_merge_leaf_override():
    """override 的 leaf 值覆盖 base。"""
    from host.config.settings import _deep_merge

    base = {"target": {"model": "default-model", "inference_base_url": "http://default:8080/v1"}}
    override = {"target": {"inference_base_url": "http://custom:9090/v1"}}
    merged = _deep_merge(base, override)
    assert merged["target"]["model"] == "default-model"  # 来自 base
    assert merged["target"]["inference_base_url"] == "http://custom:9090/v1"  # 来自 override


def test_deep_merge_nested_preservation():
    """override 只改嵌套 dict 的部分 key，其他 key 保留。"""
    from host.config.settings import _deep_merge

    base = {
        "target": {
            "llama_cpp": {
                "lifecycle": {"ensure_loaded": True, "unload_after_run": False},
                "preflight": {"check_health": True},
            }
        }
    }
    override = {"target": {"inference_base_url": "http://edge:8080/v1"}}
    merged = _deep_merge(base, override)
    assert merged["target"]["llama_cpp"]["lifecycle"]["ensure_loaded"] is True
    assert merged["target"]["llama_cpp"]["preflight"]["check_health"] is True
    assert merged["target"]["inference_base_url"] == "http://edge:8080/v1"


def test_deep_merge_list_replaced():
    """list 字段整体替换，不做 element-wise merge。"""
    from host.config.settings import _deep_merge

    base = {"tau_bench": {"domains": ["airline", "retail"]}}
    override = {"tau_bench": {"domains": ["banking_knowledge"]}}
    merged = _deep_merge(base, override)
    assert merged["tau_bench"]["domains"] == ["banking_knowledge"]


def test_deep_merge_new_keys():
    """override 新增 base 中没有的 key。"""
    from host.config.settings import _deep_merge

    base = {"target": {"model": "m1"}}
    override = {"lmms_eval": {"hub_cache": "/custom/path"}}
    merged = _deep_merge(base, override)
    assert merged["target"]["model"] == "m1"
    assert merged["lmms_eval"]["hub_cache"] == "/custom/path"


def test_load_settings_merges_base_and_device(tmp_path, monkeypatch):
    """实际 base.yaml + device override 验证集成合并。"""
    from host.config.settings import load_settings

    # Point BASE_CONFIG to a temp base.yaml
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        "target:\n  model: base-model\n  monitor_timeout_s: 30\nlogging:\n  level: INFO\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("host.config.settings.BASE_CONFIG", base_path)

    device_path = tmp_path / "device.yaml"
    device_path.write_text(
        "target:\n  model: device-model\n",
        encoding="utf-8",
    )
    settings = load_settings(device_path)
    assert settings.target.model == "device-model"  # 来自 device override
    assert settings.target.monitor_timeout_s == 30  # 来自 base
    assert settings.logging.level == "INFO"  # 来自 base


def test_nested_config_direct_instantiation_ignores_environment(monkeypatch):
    """子配置是纯 BaseModel，直接实例化不读取环境变量。"""
    monkeypatch.setenv("TARGET__MODEL", "env-model")
    cfg = TargetConfig()
    assert cfg.model == "Qwen3.5-4B"


def test_settings_from_yaml_environment_overrides_yaml(tmp_path, monkeypatch):
    """顶层 Settings 统一处理 env，优先级高于 base/device YAML。"""
    base_path = tmp_path / "base.yaml"
    base_path.write_text("target:\n  model: base-model\n", encoding="utf-8")
    monkeypatch.setattr("host.config.settings.BASE_CONFIG", base_path)

    device_path = tmp_path / "device.yaml"
    device_path.write_text("target:\n  model: device-model\n", encoding="utf-8")
    monkeypatch.setenv("TARGET__MODEL", "env-model")

    settings = Settings.from_yaml(device_path)
    assert settings.target.model == "env-model"


# ---------------------------------------------------------------------------
# build_lmms_model_args
# ---------------------------------------------------------------------------


def test_build_lmms_model_args_excludes_api_key():
    settings = Settings(
        target={
            "inference_base_url": "http://edge:8080/v1",
            "inference_api_key": "target-secret-sentinel",
            "model": "Qwen3.5-4B-Q4_K_M",
        },
    )
    args = settings.build_lmms_model_args()
    assert "api_key" not in args
    assert "target-secret-sentinel" not in args


def test_build_lmms_model_args_includes_adaptive_concurrency():
    settings = Settings(
        target={
            "inference_base_url": "http://edge:8080/v1",
            "inference_api_key": "test-key",
            "model": "Qwen3.5-4B-Q4_K_M",
        },
        eval={"api_nproc": 8},
        lmms_eval={"adaptive_concurrency": True},
    )
    args = settings.build_lmms_model_args()
    assert "model_version=Qwen3.5-4B-Q4_K_M" in args
    assert "base_url=http://edge:8080/v1" in args
    assert "api_key" not in args
    assert "test-key" not in args
    assert "num_cpus=8" in args
    assert "adaptive_concurrency=true" in args


# ---------------------------------------------------------------------------
# inference_root_url 字符串裁剪
# ---------------------------------------------------------------------------


def test_inference_root_url_strips_v1():
    """inference_root_url 应去掉 /v1 后缀。"""
    settings = Settings(target={"inference_base_url": "http://edge:8080/v1"})
    assert settings.target.inference_root_url() == "http://edge:8080"


def test_inference_root_url_no_v1():
    """没有 /v1 后缀时返回原 URL。"""
    settings = Settings(target={"inference_base_url": "http://edge:8080"})
    assert settings.target.inference_root_url() == "http://edge:8080"


def test_inference_root_url_trailing_slash():
    """尾部有斜杠时正确去掉 /v1。"""
    settings = Settings(target={"inference_base_url": "http://edge:8080/v1/"})
    assert settings.target.inference_root_url() == "http://edge:8080"


# ---------------------------------------------------------------------------
# judge_env
# ---------------------------------------------------------------------------


def test_judge_env_context_restores_environment():
    """judge_env context manager 应正确设置和还原环境变量。"""
    settings = Settings(
        judge={
            "openai_api_base": "https://api.openai.com/v1",
            "openai_api_key": "sk-judge",
        }
    )
    from host.config import judge_env

    os.environ["OPENAI_API_KEY"] = "original"
    with judge_env(settings):
        assert os.environ["OPENAI_API_KEY"] == "sk-judge"
        assert "OPENAI_API_URL" in os.environ
    assert os.environ["OPENAI_API_KEY"] == "original"
    assert "OPENAI_API_URL" not in os.environ
    del os.environ["OPENAI_API_KEY"]


def test_tau_bench_env_context_overrides_and_restores_environment(monkeypatch):
    """tau_bench_env context manager 应覆盖旧 key 并在退出后还原。"""
    from host.config import tau_bench_env

    monkeypatch.setenv("OPENAI_API_KEY", "old-key")
    monkeypatch.setenv("TAU_USER_API_KEY", "sk-tau")
    settings = Settings()

    with tau_bench_env(settings):
        assert os.environ["OPENAI_API_KEY"] == "sk-tau"

    assert os.environ["OPENAI_API_KEY"] == "old-key"


def test_claw_eval_env_context_overrides_and_restores_environment(monkeypatch):
    """claw_eval_env context manager 应注入 Target/judge key 并在退出后还原。"""
    from host.config import claw_eval_env

    monkeypatch.setenv("OPENROUTER_API_KEY", "old-router")
    monkeypatch.setenv("OPENAI_API_KEY", "old-openai")
    monkeypatch.delenv("CLAW_USER_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = Settings(
        target={"inference_api_key": "target-key", "model": "m"},
        judge={"openai_api_key": "judge-key"},
        claw_eval={"api_key_env": "OPENROUTER_API_KEY"},
    )

    with claw_eval_env(settings):
        assert os.environ["OPENROUTER_API_KEY"] == "target-key"
        assert os.environ["OPENAI_API_KEY"] == "judge-key"
        assert os.environ["CLAW_USER_API_KEY"] == "judge-key"

    assert os.environ["OPENROUTER_API_KEY"] == "old-router"
    assert os.environ["OPENAI_API_KEY"] == "old-openai"
    assert "CLAW_USER_API_KEY" not in os.environ


# ---------------------------------------------------------------------------
# resolve_claw_eval_user_agent_api_key priority
# ---------------------------------------------------------------------------


def test_resolve_claw_user_agent_key_claw_env(monkeypatch):
    monkeypatch.setenv("CLAW_USER_API_KEY", "sk-claw")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
    settings = Settings(judge={"openai_api_key": "sk-judge"})
    assert settings.resolve_claw_eval_user_agent_api_key() == "sk-claw"


def test_resolve_claw_user_agent_key_deepseek_env(monkeypatch):
    monkeypatch.delenv("CLAW_USER_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
    settings = Settings(judge={"openai_api_key": "sk-judge"})
    assert settings.resolve_claw_eval_user_agent_api_key() == "sk-ds"


def test_resolve_claw_user_agent_key_judge_fallback(monkeypatch):
    monkeypatch.delenv("CLAW_USER_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = Settings(judge={"openai_api_key": "sk-judge"})
    assert settings.resolve_claw_eval_user_agent_api_key() == "sk-judge"


def test_claw_eval_env_uses_dedicated_user_agent_key(monkeypatch):
    """claw_eval_env should inject CLAW_USER_API_KEY from env, not judge."""
    from host.config import claw_eval_env

    monkeypatch.setenv("CLAW_USER_API_KEY", "sk-claw-ua")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = Settings(
        target={"inference_api_key": "target-key", "model": "m"},
        judge={"openai_api_key": "judge-key"},
    )
    with claw_eval_env(settings):
        assert os.environ["CLAW_USER_API_KEY"] == "sk-claw-ua"
        assert os.environ["OPENROUTER_API_KEY"] == "judge-key"


def test_claw_eval_env_injects_judge_fallback_as_claw_user_key(monkeypatch):
    """When no dedicated key is set, claw_eval_env injects judge key as CLAW_USER_API_KEY."""
    from host.config import claw_eval_env

    monkeypatch.delenv("CLAW_USER_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = Settings(
        target={"inference_api_key": "target-key", "model": "m"},
        judge={"openai_api_key": "judge-key"},
    )
    with claw_eval_env(settings):
        assert os.environ["CLAW_USER_API_KEY"] == "judge-key"
    assert "CLAW_USER_API_KEY" not in os.environ


# ---------------------------------------------------------------------------
# terminal_bench_env_for_subprocess
# ---------------------------------------------------------------------------


def test_terminal_bench_env_for_subprocess_keys():
    """验证 env 合并：API key。"""
    settings = Settings(
        target={"inference_api_key": "secret", "model": "m"},
    )
    env = settings.terminal_bench_env_for_subprocess()
    assert env["OPENAI_API_KEY"] == "secret"


def test_terminal_bench_env_for_subprocess_no_docker_default_platform():
    """Terminal-Bench subprocess env 不设置 Docker platform。"""
    settings = Settings(
        target={"inference_api_key": "secret", "model": "m"},
    )
    env = settings.terminal_bench_env_for_subprocess()
    assert "DOCKER_DEFAULT_PLATFORM" not in env
    assert env["OPENAI_API_KEY"] == "secret"


def test_terminal_bench_env_for_subprocess_empty_key():
    """空的 inference_api_key 使用 'EMPTY' 占位符设置 OPENAI_API_KEY。"""
    settings = Settings(
        target={"inference_api_key": "", "model": "m"},
    )
    env = settings.terminal_bench_env_for_subprocess()
    assert "DOCKER_DEFAULT_PLATFORM" not in env
    assert env["OPENAI_API_KEY"] == "EMPTY"


# ---------------------------------------------------------------------------
# LlamaCppConfig 子模型
# ---------------------------------------------------------------------------


def test_llama_cpp_config_defaults():
    """验证 LlamaCppConfig 默认值。"""
    cfg = LlamaCppConfig()
    assert cfg.mode == "router"
    assert cfg.lifecycle.ensure_loaded is True
    assert cfg.lifecycle.unload_after_run is False


def test_llama_cpp_config_model_alias_resolution():
    cfg = LlamaCppConfig(model_aliases={"Qwen3.5-4B-Q4_K_M": "Qwen/Qwen3.5-4B-GGUF:Q4_K_M"})
    assert cfg.resolve_model_id("Qwen3.5-4B-Q4_K_M") == "Qwen/Qwen3.5-4B-GGUF:Q4_K_M"
    assert cfg.resolve_model_id("unknown-model") == "unknown-model"


# ---------------------------------------------------------------------------
# LoggingConfig
# ---------------------------------------------------------------------------


def test_logging_resolved_save_dir_default():
    settings = Settings()
    assert settings.logging.resolved_save_dir() is None


def test_logging_resolved_save_dir_none():
    settings = Settings(logging={"save_dir": None})
    assert settings.logging.resolved_save_dir() is None


# ---------------------------------------------------------------------------
# resolved_inference_api_key
# ---------------------------------------------------------------------------


def test_resolved_inference_api_key_empty_uses_placeholder():
    settings = Settings(target={"inference_api_key": ""})
    assert settings.target.resolved_inference_api_key() == "EMPTY"


def test_resolved_inference_api_key_present():
    settings = Settings(target={"inference_api_key": "sk-real"})
    assert settings.target.resolved_inference_api_key() == "sk-real"


# ---------------------------------------------------------------------------
# resolved_terminal_bench_api_base override
# ---------------------------------------------------------------------------


def test_resolved_terminal_bench_api_base_override():
    """agent_kwargs 中的 api_base 应覆盖 target.inference_base_url。"""
    settings = Settings(
        target={"inference_base_url": "http://edge:8080/v1", "model": "m"},
        terminal_bench={"agent_kwargs": {"api_base": "http://custom:9090/v1"}},
    )
    assert settings.resolved_terminal_bench_api_base() == "http://custom:9090/v1"


def test_resolved_terminal_bench_api_base_fallback():
    """没有 api_base override 时应回退到 target.inference_base_url。"""
    settings = Settings(
        target={"inference_base_url": "http://edge:8080/v1", "model": "m"},
    )
    assert settings.resolved_terminal_bench_api_base() == "http://edge:8080/v1"
