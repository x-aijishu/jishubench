import json
from pathlib import Path

import pytest
from host.adapters.terminal_bench_2 import TerminalBench2Runner
from host.config import Settings
from host.config.settings import REPO_ROOT, load_settings


def test_load_default_settings_includes_terminal_bench_2():
    settings = load_settings(REPO_ROOT / "configs" / "base.yaml")
    assert settings.terminal_bench_2.agent == "terminus-2"
    assert settings.terminal_bench_2.dataset == "terminal-bench@2.0"


def test_terminal_bench_2_litellm_model():
    settings = Settings(
        target={
            "inference_base_url": "http://edge:8080/v1",
            "model": "Qwen3.5-4B-Q4_K_M",
            "llama_cpp": {
                "model_aliases": {"Qwen3.5-4B-Q4_K_M": "Qwen/Qwen3.5-4B-GGUF:Q4_K_M"},
            },
        }
    )
    assert settings.terminal_bench_2_litellm_model() == "openai/Qwen/Qwen3.5-4B-GGUF:Q4_K_M"


def test_terminal_bench_2_env_for_subprocess():
    settings = Settings(target={"inference_api_key": "secret", "model": "m"})
    env = settings.terminal_bench_2_env_for_subprocess()
    assert env["OPENAI_API_KEY"] == "secret"


def test_parse_harbor_results_rglob(tmp_path):
    runner = TerminalBench2Runner(Settings())
    run_path = tmp_path / "harbor-run"
    trial = run_path / "trials" / "hello-world"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps({"task_id": "hello-world", "success": True}),
        encoding="utf-8",
    )
    result = runner._parse_harbor_results(tmp_path, "harbor-run")
    assert result.n_total == 1
    assert result.n_passed == 1
    assert result.metric == "accuracy"
    assert result.value == 1.0
    assert result.summary["harness"] == "harbor"


def test_parse_harbor_results_rglob_results_json(tmp_path):
    """Harbor 结果文件可能是 results.json 而不是 result.json。"""
    runner = TerminalBench2Runner(Settings())
    run_path = tmp_path / "harbor-run-2"
    trial = run_path / "trials" / "realworld"
    trial.mkdir(parents=True)
    (trial / "results.json").write_text(
        json.dumps({"task_id": "realworld", "is_resolved": True}),
        encoding="utf-8",
    )
    result = runner._parse_harbor_results(tmp_path, "harbor-run-2")
    assert result.n_total == 1
    assert result.n_passed == 1


def test_parse_harbor_results_no_results(tmp_path):
    """没有结果文件时空结果。"""
    runner = TerminalBench2Runner(Settings())
    run_path = tmp_path / "empty-harbor"
    run_path.mkdir()
    result = runner._parse_harbor_results(tmp_path, "empty-harbor")
    assert result.success is False
    assert result.n_total == 0
    assert result.n_passed == 0
    assert result.metric is None
    assert result.value is None
    assert result.summary["tasks"] == []
    assert result.summary["parse_error"] == "no harbor terminal-bench task results found"


def test_parse_harbor_results_skips_summary_dir(tmp_path):
    """跳过 run_path 级别的 results.json（不是 task 目录的）。"""
    runner = TerminalBench2Runner(Settings())
    run_path = tmp_path / "harbor-run-3"
    run_path.mkdir()
    (run_path / "results.json").write_text(json.dumps({"summary": True}), encoding="utf-8")
    trial = run_path / "trials" / "task-a"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps({"task_id": "task-a", "is_resolved": True}),
        encoding="utf-8",
    )
    result = runner._parse_harbor_results(tmp_path, "harbor-run-3")
    assert result.n_total == 1
    assert result.n_passed == 1


def test_parse_harbor_results_harbor_v13_trial_result_json(tmp_path):
    """Harbor 0.13 trial result.json uses task_name and verifier_result.rewards."""
    runner = TerminalBench2Runner(Settings())
    run_path = tmp_path / "harbor-run-v13"
    trial = run_path / "fix-git__yT343xy"
    trial.mkdir(parents=True)
    (run_path / "result.json").write_text(json.dumps({"n_total_trials": 1}), encoding="utf-8")
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "fix-git",
                "task_id": {
                    "git_url": "https://github.com/laude-institute/terminal-bench-2.git",
                    "git_commit_id": "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c",
                    "path": "fix-git",
                },
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    result = runner._parse_harbor_results(tmp_path, "harbor-run-v13")
    assert result.n_total == 1
    assert result.n_passed == 1
    assert result.value == 1.0
    assert result.summary["tasks"][0]["task_id"] == "fix-git"


def test_verify_local_dataset_found(tmp_path, monkeypatch):
    from host.adapters.terminal_bench_2 import _verify_local_dataset

    settings = Settings()
    cache_root = tmp_path / ".cache" / "harbor" / "tasks" / "packages" / "terminal-bench"
    cache_root.mkdir(parents=True)
    (cache_root / "some-task").mkdir()

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    _verify_local_dataset(settings, "terminal-bench@2.0")


def test_verify_local_dataset_missing(tmp_path, monkeypatch):
    from host.adapters.terminal_bench_2 import _verify_local_dataset

    settings = Settings()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    with pytest.raises(FileNotFoundError, match="not found"):
        _verify_local_dataset(settings, "nonexistent@2.0")


def test_download_backend_registered():
    from host.adapters.terminal_bench_2 import TerminalBench2Runner
    from host.cli.commands import _BACKEND_RUNNERS

    assert _BACKEND_RUNNERS.get("terminal-bench-2") is TerminalBench2Runner
    assert callable(TerminalBench2Runner.download_tasks)


def test_build_harbor_command_uses_current_cli_flags():
    runner = TerminalBench2Runner(Settings())
    cmd = runner._build_harbor_command(
        output_dir=Path("outputs/runs/terminal-bench-2/model__ts"),
        run_id="20260716_050510",
        model="openai/Qwen3.6-35B-A3B",
        agent_kwargs={"api_base": "http://127.0.0.1:8080/v1", "temperature": 0.0},
        limit=None,
        task_ids=["fix-git", "overfull-hbox"],
    )
    assert "--jobs-dir" in cmd
    assert "--job-name" in cmd
    assert "--output-dir" not in cmd
    assert "--include-task-name" in cmd
    assert "--task-name" not in cmd
    assert cmd[cmd.index("--jobs-dir") + 1] == "outputs/runs/terminal-bench-2/model__ts"
    assert cmd[cmd.index("--job-name") + 1] == "20260716_050510"
    assert cmd.count("--include-task-name") == 2


def test_cli_run_terminal_bench_2_registered():
    from host.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(["run", "terminal-bench-2", "--limit", "1"])
    assert args.command == "run"
    assert args.benchmarks == ["terminal-bench-2"]
    assert args.limit == 1


def test_cli_preflight_benchmark_choice():
    from host.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(["preflight", "--benchmark", "terminal-bench-2"])
    assert args.benchmark == "terminal-bench-2"
