import json
from unittest.mock import patch

import pytest
from host.adapters.terminal_bench import TerminalBenchRunner
from host.config import Settings, load_settings, terminal_bench_env
from host.config.settings import REPO_ROOT
from loguru import logger


def test_load_default_settings_includes_terminal_bench():
    settings = load_settings(REPO_ROOT / "configs" / "base.yaml")
    assert settings.terminal_bench.agent == "terminus-2"
    assert settings.terminal_bench.dataset == "terminal-bench-core==0.1.1"


def test_terminal_bench_litellm_model():
    settings = Settings(
        target={
            "inference_base_url": "http://edge:8080/v1",
            "model": "Qwen3.5-4B-Q4_K_M",
            "llama_cpp": {
                "model_aliases": {"Qwen3.5-4B-Q4_K_M": "Qwen/Qwen3.5-4B-GGUF:Q4_K_M"},
            },
        }
    )
    assert settings.terminal_bench_litellm_model() == "openai/Qwen/Qwen3.5-4B-GGUF:Q4_K_M"


def test_terminal_bench_env_for_subprocess():
    settings = Settings(
        target={"inference_api_key": "secret", "model": "m"},
    )
    env = settings.terminal_bench_env_for_subprocess()
    assert "DOCKER_DEFAULT_PLATFORM" not in env
    assert env["OPENAI_API_KEY"] == "secret"


def test_terminal_bench_env_restores(monkeypatch):
    import os

    settings = Settings(target={"inference_api_key": "secret"})
    os.environ.pop("OPENAI_API_KEY", None)
    with terminal_bench_env(settings):
        assert os.environ["OPENAI_API_KEY"] == "secret"
    assert "OPENAI_API_KEY" not in os.environ


def test_terminal_bench_timeout_settings_and_harness_kwargs():
    settings = Settings(
        terminal_bench={
            "global_timeout_multiplier": 3.0,
            "global_agent_timeout_sec": 1800.0,
            "global_test_timeout_sec": 300.0,
        }
    )
    cfg = settings.terminal_bench
    assert cfg.global_timeout_multiplier == 3.0
    kwargs = TerminalBenchRunner._harness_timeout_kwargs(cfg)
    assert kwargs == {
        "global_timeout_multiplier": 3.0,
        "global_agent_timeout_sec": 1800.0,
        "global_test_timeout_sec": 300.0,
    }


def test_parse_tb_results(tmp_path):
    settings = Settings()
    runner = TerminalBenchRunner(settings)
    run_path = tmp_path / "run-1"
    run_path.mkdir()
    payload = {
        "results": [
            {"task_id": "a", "is_resolved": True, "failure_mode": "unset"},
            {"task_id": "b", "is_resolved": False, "failure_mode": "agent_timeout"},
        ]
    }
    (run_path / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    result = runner._parse_tb_results(tmp_path, "run-1")
    assert result.success is True
    assert result.n_total == 2
    assert result.n_passed == 1
    assert result.metric == "accuracy"
    assert result.value == pytest.approx(0.5)
    assert result.summary["n_resolved"] == 1
    assert result.summary["resolved_ids"] == ["a"]
    assert result.summary["unresolved_ids"] == ["b"]
    assert result.summary["parse_error"] is None


def test_preflight_arm64_not_supported():
    import io

    settings = Settings()
    runner = TerminalBenchRunner(settings)
    buf = io.StringIO()
    sink_id = logger.add(buf, format="{level} {message}", level="DEBUG")

    with patch("host.adapters.terminal_bench.check_docker_available", return_value=True):
        with patch("host.adapters.terminal_bench.check_harness_cli", return_value="/usr/bin/tb"):
            with patch("host.adapters.terminal_bench.platform.machine", return_value="arm64"):
                ready = runner.preflight()

    logger.remove(sink_id)
    assert not ready
    log_output = buf.getvalue()
    assert "ARM64" in log_output
    assert "unsupported" in log_output


def test_preflight_log_lists_errors_separately():
    import io

    settings = Settings()
    runner = TerminalBenchRunner(settings)
    buf = io.StringIO()
    sink_id = logger.add(buf, format="{level} {message}", level="DEBUG")

    with patch("host.adapters.terminal_bench.check_docker_available", return_value=False):
        with patch("host.adapters.terminal_bench.check_harness_cli", return_value=None):
            with patch("host.adapters.terminal_bench.platform.machine", return_value="x86_64"):
                ready = runner.preflight()

    logger.remove(sink_id)
    output = buf.getvalue()
    assert not ready
    assert "docker: not reachable" in output
    assert "tb CLI: not found" in output
    assert "Terminal-Bench preflight: not ready (2 error(s))" in output


def test_download_backend_registered():
    from host.adapters.terminal_bench import TerminalBenchRunner
    from host.cli.commands import _BACKEND_RUNNERS

    assert _BACKEND_RUNNERS.get("terminal-bench") is TerminalBenchRunner
    assert callable(TerminalBenchRunner.download_tasks)


def test_cli_run_terminal_bench_registered():
    from host.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(["run", "terminal-bench", "--limit", "1"])
    assert args.command == "run"
    assert args.benchmarks == ["terminal-bench"]
    assert args.limit == 1


def test_cli_preflight_benchmark_no_json(capsys, monkeypatch):
    from host.main import main

    class _FakeController:
        def preflight(self, *args, **kwargs):
            return {
                "submodules": {"lmms-eval": True},
                "target_reachable": True,
                "ready": True,
                "adapter_ready": True,
                "warnings": ["monitor unreachable (non-blocking): test"],
            }

    monkeypatch.setattr("host.cli.commands.HostController", lambda _s: _FakeController())

    assert main(["preflight", "--benchmark", "terminal-bench"]) == 0
    out = capsys.readouterr().out
    assert '"submodules"' not in out
    assert '"ready"' not in out


class TestParseDatasetSpec:
    def test_double_equal(self):
        from host.adapters.terminal_bench import _parse_dataset_spec

        assert _parse_dataset_spec("foo==1.0") == ("foo", "1.0")

    def test_at_sign(self):
        from host.adapters.terminal_bench import _parse_dataset_spec

        assert _parse_dataset_spec("bar@v2") == ("bar", "v2")

    def test_bare_name(self):
        from host.adapters.terminal_bench import _parse_dataset_spec

        assert _parse_dataset_spec("baz") == ("baz", None)

    def test_empty_string(self):
        from host.adapters.terminal_bench import _parse_dataset_spec

        assert _parse_dataset_spec("") == ("", None)

    def test_whitespace(self):
        from host.adapters.terminal_bench import _parse_dataset_spec

        assert _parse_dataset_spec("  spaced-name  ") == ("spaced-name", None)


class TestStringify:
    def test_none(self):
        from host.adapters.terminal_bench import _stringify

        assert _stringify(None) is None

    def test_string(self):
        from host.adapters.terminal_bench import _stringify

        assert _stringify("hello") == "hello"

    def test_non_string(self):
        from host.adapters.terminal_bench import _stringify

        assert _stringify(42) == "42"
        assert _stringify(True) == "True"


class TestDedupeTaskResults:
    def test_dedupe_simple(self):
        from host.adapters.terminal_bench import (
            TerminalBenchTaskResult,
            _dedupe_task_results,
        )

        tasks = [
            TerminalBenchTaskResult(task_id="a", is_resolved=True),
            TerminalBenchTaskResult(task_id="b", is_resolved=False),
            TerminalBenchTaskResult(task_id="a", is_resolved=False),
        ]
        deduped = _dedupe_task_results(tasks)
        assert len(deduped) == 2
        assert next(task for task in deduped if task.task_id == "a").is_resolved is True

    def test_dedupe_all_unique(self):
        from host.adapters.terminal_bench import (
            TerminalBenchTaskResult,
            _dedupe_task_results,
        )

        tasks = [
            TerminalBenchTaskResult(task_id="a", is_resolved=True),
            TerminalBenchTaskResult(task_id="b", is_resolved=False),
        ]
        assert len(_dedupe_task_results(tasks)) == 2

    def test_dedupe_empty(self):
        from host.adapters.terminal_bench import _dedupe_task_results

        assert _dedupe_task_results([]) == []

    def test_dedupe_pass_at_k(self):
        from host.adapters.terminal_bench import (
            TerminalBenchTaskResult,
            _dedupe_task_results,
        )

        tasks = [
            TerminalBenchTaskResult(task_id="a", is_resolved=None),
            TerminalBenchTaskResult(task_id="a", is_resolved=True),
        ]
        assert _dedupe_task_results(tasks)[0].is_resolved is True


def test_parse_tb_results_fallback_to_per_task(tmp_path):
    runner = TerminalBenchRunner(Settings())
    task_dir = tmp_path / "run-2" / "task_a"
    task_dir.mkdir(parents=True)
    (task_dir / "results.json").write_text(
        json.dumps({"task_id": "task_a", "is_resolved": True}), encoding="utf-8"
    )

    result = runner._parse_tb_results(tmp_path, "run-2")

    assert result.n_total == 1
    assert result.n_passed == 1


def test_parse_tb_results_no_results_at_all(tmp_path):
    runner = TerminalBenchRunner(Settings())
    (tmp_path / "empty-run").mkdir()

    result = runner._parse_tb_results(tmp_path, "empty-run")

    assert result.success is False
    assert result.n_total == 0
    assert result.n_passed == 0
    assert result.metric is None
    assert result.value is None
    assert result.summary["tasks"] == []
    assert result.summary["parse_error"] == "no terminal-bench task results found"


def test_verify_local_dataset_found(tmp_path, monkeypatch):
    from host.adapters.terminal_bench import _verify_local_dataset

    cache_root = tmp_path / ".cache" / "terminal-bench" / "myds" / "v1"
    cache_root.mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    _verify_local_dataset(Settings(), "myds==v1")


def test_verify_local_dataset_missing(tmp_path, monkeypatch):
    from host.adapters.terminal_bench import _verify_local_dataset

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    with pytest.raises(FileNotFoundError, match="not found"):
        _verify_local_dataset(Settings(), "nonexistent==v1")
