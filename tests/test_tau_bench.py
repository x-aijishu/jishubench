"""Tests for the tau-bench adapter (no real tau2 simulation required)."""

from __future__ import annotations

import os
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from host.config import REPO_ROOT, Settings, load_settings
from host.config.submodules import TAU2_BENCH_DIR, tau2_data_dir

# ---------------------------------------------------------------------------
# Settings & config
# ---------------------------------------------------------------------------


def test_load_default_settings_includes_tau_bench():
    settings = load_settings(REPO_ROOT / "configs" / "base.yaml")
    assert settings.tau_bench.domains == ["airline", "retail", "telecom"]
    assert settings.tau_bench.split == "base"
    assert settings.tau_bench.num_tasks is None
    assert settings.tau_bench.user_model == "deepseek-v4-flash"


def test_tau_bench_run_dir_created(tmp_path):
    settings = Settings(eval={"work_dir": tmp_path})
    out = settings.run_dir("tau-bench", "run-test")
    assert out.is_dir()
    assert "tau-bench" in str(out)
    assert "run-test" in str(out)


# ---------------------------------------------------------------------------
# resolve_tau_user_api_key priority
# ---------------------------------------------------------------------------


def test_resolve_tau_user_api_key_tau_env(monkeypatch):
    monkeypatch.setenv("TAU_USER_API_KEY", "sk-tau-test")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = Settings()
    assert settings.resolve_tau_user_api_key() == "sk-tau-test"


def test_resolve_tau_user_api_key_deepseek_env(monkeypatch):
    monkeypatch.delenv("TAU_USER_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
    settings = Settings()
    assert settings.resolve_tau_user_api_key() == "sk-ds-test"


def test_resolve_tau_user_api_key_judge_fallback(monkeypatch):
    monkeypatch.delenv("TAU_USER_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = Settings(judge={"openai_api_key": "sk-judge"})
    assert settings.resolve_tau_user_api_key() == "sk-judge"


def test_resolve_tau_user_api_key_tau_takes_precedence(monkeypatch):
    monkeypatch.setenv("TAU_USER_API_KEY", "sk-tau")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
    settings = Settings()
    assert settings.resolve_tau_user_api_key() == "sk-tau"


# ---------------------------------------------------------------------------
# pass@k helper
# ---------------------------------------------------------------------------


def test_pass_hat_k_all_fail():
    from host.adapters.tau_bench import _pass_hat_k

    assert _pass_hat_k([0.0, 0.5, 0.9]) == 0.0


def test_pass_hat_k_one_success():
    from host.adapters.tau_bench import _pass_hat_k

    assert _pass_hat_k([0.0, 1.0]) == 1.0


def test_pass_hat_k_empty():
    from host.adapters.tau_bench import _pass_hat_k

    assert _pass_hat_k([]) == 0.0


def test_pass_hat_k_custom_threshold():
    from host.adapters.tau_bench import _pass_hat_k

    assert _pass_hat_k([0.7, 0.8], threshold=0.75) == 1.0
    assert _pass_hat_k([0.5, 0.6], threshold=0.75) == 0.0


# ---------------------------------------------------------------------------
# TauBenchRunner.preflight() mocked
# ---------------------------------------------------------------------------


def test_preflight_submodule_missing(monkeypatch, tmp_path):
    from host.adapters.tau_bench import TauBenchRunner

    settings = Settings(eval={"work_dir": tmp_path})
    runner = TauBenchRunner(settings)

    with patch("host.adapters.tau_bench._check_tau2_importable", return_value=True):
        with patch("host.config.submodules.TAU2_BENCH_DIR", tmp_path / "missing"):
            from host.config import submodules as sm

            orig = sm.TAU2_BENCH_DIR
            sm.TAU2_BENCH_DIR = tmp_path / "missing"
            try:
                ready = runner.preflight()
            finally:
                sm.TAU2_BENCH_DIR = orig

    assert not ready


def test_preflight_tau2_not_importable(monkeypatch, tmp_path):
    from host.adapters.tau_bench import TauBenchRunner

    settings = Settings(eval={"work_dir": tmp_path})
    runner = TauBenchRunner(settings)

    # Make submodule appear present
    fake_tau2 = tmp_path / "src" / "tau2"
    fake_tau2.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "airline").mkdir()

    with patch("host.config.submodules.TAU2_BENCH_DIR", tmp_path):
        with patch("host.adapters.tau_bench._check_tau2_importable", return_value=False):
            from host.config import submodules as sm

            orig = sm.TAU2_BENCH_DIR
            sm.TAU2_BENCH_DIR = tmp_path
            try:
                ready = runner.preflight()
            finally:
                sm.TAU2_BENCH_DIR = orig

    assert not ready


def test_preflight_user_key_missing(monkeypatch, tmp_path):
    from host.adapters.tau_bench import TauBenchRunner

    monkeypatch.delenv("TAU_USER_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    settings = Settings(eval={"work_dir": tmp_path}, judge={"openai_api_key": ""})
    runner = TauBenchRunner(settings)

    fake_tau2 = tmp_path / "src" / "tau2"
    fake_tau2.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "airline").mkdir()

    with patch("host.config.submodules.TAU2_BENCH_DIR", tmp_path):
        with patch("host.adapters.tau_bench._check_tau2_importable", return_value=True):
            from host.config import submodules as sm

            orig = sm.TAU2_BENCH_DIR
            sm.TAU2_BENCH_DIR = tmp_path
            try:
                ready = runner.preflight()
            finally:
                sm.TAU2_BENCH_DIR = orig

    assert not ready


# ---------------------------------------------------------------------------
# TauBenchRunner.run() mocked
# ---------------------------------------------------------------------------


def test_run_returns_failure_when_no_tasks_selected(monkeypatch, tmp_path):
    from host.adapters.tau_bench import TauBenchRunArgs, TauBenchRunner

    simulation_module = types.ModuleType("tau2.data_model.simulation")
    simulation_module.TextRunConfig = MagicMock()
    batch_module = types.ModuleType("tau2.runner.batch")
    batch_module.run_single_task = MagicMock()
    helpers_module = types.ModuleType("tau2.runner.helpers")

    def fake_get_tasks(**kwargs):
        assert os.environ["OPENAI_API_KEY"] == "old-key"
        return []

    helpers_module.get_tasks = MagicMock(side_effect=fake_get_tasks)

    monkeypatch.setitem(sys.modules, "tau2", types.ModuleType("tau2"))
    monkeypatch.setitem(sys.modules, "tau2.data_model", types.ModuleType("tau2.data_model"))
    monkeypatch.setitem(sys.modules, "tau2.data_model.simulation", simulation_module)
    monkeypatch.setitem(sys.modules, "tau2.runner", types.ModuleType("tau2.runner"))
    monkeypatch.setitem(sys.modules, "tau2.runner.batch", batch_module)
    monkeypatch.setitem(sys.modules, "tau2.runner.helpers", helpers_module)
    monkeypatch.setattr("host.adapters.tau_bench._ensure_llm_patch", MagicMock())

    monkeypatch.setenv("OPENAI_API_KEY", "old-key")
    monkeypatch.setenv("TAU_USER_API_KEY", "sk-tau-run")
    settings = Settings(tau_bench={"domains": ["missing-domain"], "split": "base"})
    runner = TauBenchRunner(settings)

    result = runner.run(
        TauBenchRunArgs(domains=["missing-domain"], run_id="empty-run"),
        output_dir=tmp_path,
    )

    assert result.success is False
    assert result.n_total == 0
    assert result.n_passed == 0
    assert result.metric is None
    assert result.value is None
    assert result.summary["tasks"] == []
    assert result.summary["parse_error"] == "no tau-bench tasks selected"
    assert os.environ["OPENAI_API_KEY"] == "old-key"
    batch_module.run_single_task.assert_not_called()


# ---------------------------------------------------------------------------
# Download backend
# ---------------------------------------------------------------------------


def test_tau_bench_download_backend_registered():
    from host.adapters.tau_bench import TauBenchRunner
    from host.cli.commands import _BACKEND_RUNNERS

    assert _BACKEND_RUNNERS.get("tau-bench") is TauBenchRunner
    assert callable(TauBenchRunner.download_tasks)


def test_tau_bench_download_check_only_success(tmp_path):
    from host.adapters.tau_bench import TauBenchRunner

    settings = Settings(eval={"work_dir": tmp_path})

    fake_root = tmp_path / "tau2-bench"
    fake_data = fake_root / "data"
    fake_data.mkdir(parents=True)
    (fake_data / "airline").mkdir()

    with patch("host.config.submodules.TAU2_BENCH_DIR", fake_root):
        with patch("host.config.submodules.tau2_data_dir", return_value=fake_data):
            report = TauBenchRunner.download_tasks(
                ["tau2-bench"], settings=settings, check_only=True
            )

    assert "tau2-bench" in report.succeeded


def test_tau_bench_download_check_only_missing(tmp_path):
    from host.adapters.tau_bench import TauBenchRunner

    settings = Settings(eval={"work_dir": tmp_path})

    with patch("host.config.submodules.TAU2_BENCH_DIR", tmp_path / "missing"):
        with patch(
            "host.config.submodules.tau2_data_dir", return_value=tmp_path / "missing" / "data"
        ):
            report = TauBenchRunner.download_tasks(
                ["tau2-bench"], settings=settings, check_only=True
            )

    assert "tau2-bench" in report.failed


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


def test_cli_run_tau_bench_registered():
    from host.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(["run", "tau-bench", "--limit", "2"])
    assert args.command == "run"
    assert args.benchmarks == ["tau-bench"]
    assert args.limit == 2


def test_cli_run_tau_bench_domain_flag():
    from host.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(["run", "tau-bench", "--domain", "airline", "--domain", "retail"])
    assert args.domain == ["airline", "retail"]


def test_cli_preflight_tau_bench_registered():
    from host.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(["preflight", "--benchmark", "tau-bench"])
    assert args.benchmark == "tau-bench"


# ---------------------------------------------------------------------------
# Submodule helpers
# ---------------------------------------------------------------------------


def test_submodule_status_includes_tau2_bench():
    from host.config.submodules import submodule_status

    status = submodule_status()
    assert "tau2-bench" in status
    # Should be True since we added the submodule
    assert isinstance(status["tau2-bench"], bool)


def test_tau2_data_dir_path():
    assert tau2_data_dir() == TAU2_BENCH_DIR / "data"


def test_tau_llm_patch_uses_explicit_agent_model(monkeypatch):
    import host.adapters.tau_bench as tb

    monkeypatch.setattr(tb, "_patch_tau2_generate", MagicMock())
    monkeypatch.setattr(
        tb,
        "_build_model_clients",
        MagicMock(return_value=(object(), object(), "explicit-model", "user-model")),
    )
    monkeypatch.setattr(tb, "_patch_applied", False)

    route = tb._ensure_llm_patch(
        Settings(target={"model": "yaml-model"}), agent_model="explicit-model"
    )

    tb._build_model_clients.assert_called_once()  # type: ignore[attr-defined]
    assert tb._build_model_clients.call_args.kwargs["agent_model"] == "explicit-model"  # type: ignore[attr-defined]
    assert route.agent_model == "explicit-model"


def test_tau_llm_patch_refreshes_agent_model_after_initial_patch(monkeypatch):
    import host.adapters.tau_bench as tb

    patch_generate = MagicMock()
    build_clients = MagicMock(
        side_effect=[
            (object(), object(), "first-model", "user-model"),
            (object(), object(), "second-model", "user-model"),
        ]
    )
    monkeypatch.setattr(tb, "_patch_tau2_generate", patch_generate)
    monkeypatch.setattr(tb, "_build_model_clients", build_clients)
    monkeypatch.setattr(tb, "_patch_applied", False)

    first_route = tb._ensure_llm_patch(Settings(), agent_model="first-model")
    second_route = tb._ensure_llm_patch(Settings(), agent_model="second-model")

    assert build_clients.call_count == 2
    patch_generate.assert_called_once()
    assert first_route.agent_model == "first-model"
    assert second_route.agent_model == "second-model"


def test_patched_generate_isolates_concurrent_run_routes():
    from host.adapters.tau_bench import _LLMRoute, _patched_generate, _use_llm_route

    barrier = threading.Barrier(2)

    class FakeCompletions:
        def __init__(self, calls):
            self.calls = calls

        def create(self, **kwargs):
            self.calls.append(kwargs)
            msg = MagicMock(content="ok", tool_calls=[])
            return MagicMock(choices=[MagicMock(message=msg)])

    class FakeClient:
        def __init__(self, calls):
            self.chat = MagicMock()
            self.chat.completions = FakeCompletions(calls)

    calls_a: list[dict] = []
    calls_b: list[dict] = []
    route_a = _LLMRoute(FakeClient(calls_a), FakeClient([]), "model-a", "user-a")
    route_b = _LLMRoute(FakeClient(calls_b), FakeClient([]), "model-b", "user-b")

    def generate(route):
        with _use_llm_route(route):
            barrier.wait()
            _patched_generate("agent", messages=[])

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(generate, route) for route in (route_a, route_b)]
        for future in futures:
            future.result()

    assert calls_a[0]["model"] == "model-a"
    assert calls_b[0]["model"] == "model-b"


def test_tau_llm_patch_installs_once_during_concurrent_initialization(monkeypatch):
    import host.adapters.tau_bench as tb

    barrier = threading.Barrier(2)
    patch_generate = MagicMock()

    def build_clients(*args, **kwargs):
        barrier.wait()
        return object(), object(), kwargs["agent_model"], "user-model"

    monkeypatch.setattr(tb, "_patch_tau2_generate", patch_generate)
    monkeypatch.setattr(tb, "_build_model_clients", build_clients)
    monkeypatch.setattr(tb, "_patch_applied", False)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(tb._ensure_llm_patch, Settings(), agent_model=model)
            for model in ("model-a", "model-b")
        ]
        routes = [future.result() for future in futures]

    patch_generate.assert_called_once()
    assert {route.agent_model for route in routes} == {"model-a", "model-b"}


def test_patched_generate_requires_bound_route():
    from host.adapters.tau_bench import _patched_generate

    with pytest.raises(RuntimeError, match="route not bound"):
        _patched_generate("agent", messages=[])


def test_patched_generate_tool_schema_not_double_wrapped():
    """Tool.openai_schema is already a full OpenAI tool dict; do not re-wrap it."""
    from host.adapters.tau_bench import _patched_generate
    from tau2.registry import registry

    env = registry.get_env_constructor("airline")()
    tools = env.get_tools()[:1]
    tool = tools[0]

    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = MagicMock()
            msg.content = "ok"
            msg.tool_calls = []
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            return resp

    class FakeClient:
        chat = MagicMock()
        chat.completions = FakeCompletions()

    from host.adapters.tau_bench import _LLMRoute, _use_llm_route

    route = _LLMRoute(FakeClient(), FakeClient(), "test-model", "user-model")
    with _use_llm_route(route):
        _patched_generate(
            "agent",
            messages=[],
            tools=tools,
        )

    schema = captured["tools"][0]
    assert schema["type"] == "function"
    assert "name" in schema["function"]
    assert schema["function"]["name"] == tool.name


def test_patched_generate_routes_tau2_default_judge_models_to_user_client():
    """tau2 judge defaults should use the configured remote user client, not OpenAI fallback."""
    from host.adapters.tau_bench import _patched_generate

    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = MagicMock()
            msg.content = "ok"
            msg.tool_calls = []
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            return resp

    class FakeClient:
        chat = MagicMock()
        chat.completions = FakeCompletions()

    from host.adapters.tau_bench import _LLMRoute, _use_llm_route

    route = _LLMRoute(FakeClient(), FakeClient(), "agent-model", "deepseek-v4-flash")
    with _use_llm_route(route):
        for model in ("gpt-4.1-2025-04-14", "claude-opus-4-5"):
            captured.clear()
            _patched_generate(
                model,
                messages=[],
                temperature=0,
            )
            assert captured["model"] == "deepseek-v4-flash"
            assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
