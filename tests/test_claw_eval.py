from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

import pytest
import yaml
from host.adapters.claw_eval import ClawEvalRunArgs, ClawEvalRunner
from host.config import Settings


def _settings(tmp_path: Path, tasks_dir: Path) -> Settings:
    template = tmp_path / "config_general.yaml"
    template.write_text(
        yaml.safe_dump(
            {
                "model": {"api_key": "${OPENROUTER_API_KEY}", "base_url": "old", "model_id": "old"},
                "judge": {"api_key": "${OPENROUTER_API_KEY}", "base_url": "old"},
                "defaults": {"trace_dir": "traces", "tasks_dir": "tasks"},
            }
        ),
        encoding="utf-8",
    )
    return Settings(
        target={
            "inference_base_url": "http://edge:8080/v1",
            "inference_api_key": "target-key",
            "model": "Qwen",
        },
        judge={
            "openai_api_key": "judge-key",
            "openai_api_base": "https://judge.example/v1",
            "model": "judge-model",
        },
        claw_eval={
            "tasks_dir": tasks_dir,
            "config_template": template,
            "trials": 3,
            "parallel": 1,
            "sandbox": True,
            "sandbox_image": "claw-eval-agent:latest",
            "text_only": False,
        },
    )


def _task(
    tasks_dir: Path,
    name: str,
    *,
    fixture: str | None = None,
    language: str = "zh",
    category: str = "general",
    tags: list[str] | None = None,
) -> None:
    task_dir = tasks_dir / name
    task_dir.mkdir(parents=True)
    payload = {
        "task_id": name,
        "task_name": name,
        "category": category,
        "prompt": {"text": "do it", "language": language},
        "tags": tags or [],
    }
    if fixture:
        payload["environment"] = {"fixtures": [fixture]}
    (task_dir / "task.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_claw_eval_runner_builds_batch_args(tmp_path):
    tasks_dir = tmp_path / "tasks"
    _task(tasks_dir, "T001")
    settings = _settings(tmp_path, tasks_dir)
    runner = ClawEvalRunner(settings)
    options = runner._run_options(ClawEvalRunArgs())

    args = runner._build_batch_args(
        config_path=tmp_path / "run" / "claw_eval_config.yaml",
        tasks_dir=tasks_dir,
        trace_root=tmp_path / "run" / "traces",
        model_name="Qwen",
        options=options,
    )

    assert args.trace_dir == str(tmp_path / "run" / "traces")
    assert args.trials == 3
    assert args.parallel == 1
    assert args.model == "Qwen"
    assert args.base_url == "http://edge:8080/v1"
    assert args.sandbox is True
    assert args.api_key is None


def test_claw_eval_run_writes_config_and_parses_summary(tmp_path, monkeypatch):
    tasks_dir = tmp_path / "tasks"
    _task(tasks_dir, "T001")
    settings = _settings(tmp_path, tasks_dir)

    def fake_run(args):
        assert args.model == "Qwen"
        assert args.api_key is None
        assert args.base_url == "http://edge:8080/v1"
        trace_dir = Path(args.trace_dir) / "Qwen__fake"
        trace_dir.mkdir(parents=True)
        (trace_dir / "batch_summary.json").write_text(
            json.dumps(
                {
                    "tasks": 2,
                    "trials_per_task": 3,
                    "pass_hat_3": 1,
                    "pass_at_3": 2,
                }
            ),
            encoding="utf-8",
        )
        (trace_dir / "batch_results.json").write_text(
            json.dumps(
                [
                    {
                        "task_id": "T001",
                        "trials": [
                            {
                                "passed": True,
                                "task_score": 1.0,
                                "model_input_tokens": 11,
                                "model_output_tokens": 7,
                                "tokens": 18,
                                "wall_time_s": 2.0,
                                "model_time_s": 1.5,
                                "tool_time_s": 0.5,
                            }
                        ],
                    },
                    {"task_id": "T002", "error": "failed", "trials": []},
                ]
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr("host.adapters.claw_eval._run_claw_eval_batch", fake_run)
    run_dir = tmp_path / "run"
    result = ClawEvalRunner(settings).run(
        ClawEvalRunArgs(model_name="Qwen", run_id="run123"),
        output_dir=run_dir,
    )

    config_payload = yaml.safe_load((run_dir / "claw_eval_config.yaml").read_text())
    assert config_payload["model"]["api_key"] == "${JISHU_TARGET_API_KEY}"
    assert config_payload["model"]["base_url"] == "http://edge:8080/v1"
    assert config_payload["judge"]["base_url"] == "https://judge.example/v1"
    assert config_payload["judge"]["model_id"] == "judge-model"
    assert config_payload["user_agent_model"]["base_url"] == "https://judge.example/v1"
    assert config_payload["user_agent_model"]["api_key"] == "${CLAW_USER_API_KEY}"
    assert config_payload["user_agent_model"]["model_id"] == "judge-model"
    assert result.success is True
    assert result.metric == "pass_hat_3"
    assert result.value == pytest.approx(0.5)
    assert result.n_total == 2
    assert result.n_passed == 1
    assert result.summary["failed_task_ids"] == ["T002"]
    assert result.summary["token_totals"]["tokens"] == 18
    assert result.summary["trial_metrics"][0]["model_input_tokens"] == 11
    assert result.raw is not None
    assert len(result.raw["batch_results"]) == 2


def test_claw_eval_language_category_filters_use_subset(tmp_path):
    tasks_dir = tmp_path / "tasks"
    _task(tasks_dir, "T001", language="zh", category="finance")
    _task(tasks_dir, "T002", language="en", category="finance")
    _task(tasks_dir, "T003", language="zh", category="calendar")
    settings = _settings(tmp_path, tasks_dir)
    runner = ClawEvalRunner(settings)
    options = runner._run_options(ClawEvalRunArgs(language="zh", category="finance"))

    subset = runner._prepare_task_subset(
        tasks_dir=tasks_dir,
        run_dir=tmp_path / "run",
        limit=None,
        task_ids=None,
        options=options,
    )

    assert sorted(path.name for path in subset.iterdir()) == ["T001"]


def test_claw_eval_task_subset_links_mock_services(tmp_path):
    claw_dir = tmp_path / "claw-eval"
    tasks_dir = claw_dir / "tasks"
    mock_services_dir = claw_dir / "mock_services" / "web_real"
    mock_services_dir.mkdir(parents=True)
    (mock_services_dir / "server.py").write_text("# mock\n", encoding="utf-8")
    _task(tasks_dir, "T001")
    _task(tasks_dir, "T002")

    run_dir = tmp_path / "run"
    settings = _settings(tmp_path, tasks_dir)
    runner = ClawEvalRunner(settings)
    options = runner._run_options(ClawEvalRunArgs())

    subset = runner._prepare_task_subset(
        tasks_dir=tasks_dir,
        run_dir=run_dir,
        limit=1,
        task_ids=None,
        options=options,
    )

    assert subset == run_dir / "task_subset"
    mock_services_link = run_dir / "mock_services"
    assert mock_services_link.is_symlink()
    assert mock_services_link.resolve() == (claw_dir / "mock_services").resolve()
    assert (run_dir / "mock_services" / "web_real" / "server.py").is_file()


def test_claw_eval_run_surfaces_batch_errors(tmp_path, monkeypatch):
    tasks_dir = tmp_path / "tasks"
    _task(tasks_dir, "T001")
    settings = _settings(tmp_path, tasks_dir)

    def fail_batch(args):
        raise RuntimeError("batch exploded")

    monkeypatch.setattr("host.adapters.claw_eval._run_claw_eval_batch", fail_batch)

    with pytest.raises(RuntimeError, match="batch exploded"):
        ClawEvalRunner(settings).run(
            ClawEvalRunArgs(model_name="Qwen", run_id="run123"),
            output_dir=tmp_path / "run",
        )


def test_claw_eval_download_check_detects_missing_fixtures(tmp_path):
    tasks_dir = tmp_path / "tasks"
    _task(tasks_dir, "T001", fixture="fixtures/missing.json")
    settings = _settings(tmp_path, tasks_dir)
    report = ClawEvalRunner.download_tasks(["claw-eval"], settings, check_only=True)

    assert report.succeeded == []
    assert "missing fixture" in report.failed["claw-eval"]


def test_claw_eval_download_extracts_fixtures(tmp_path, monkeypatch):
    claw_dir = tmp_path / "claw-eval"
    tasks_dir = claw_dir / "tasks"
    (claw_dir / "src" / "claw_eval").mkdir(parents=True)
    _task(tasks_dir, "T001", fixture="fixtures/data.json")
    data_dir = claw_dir / "data"
    data_dir.mkdir()
    for name in (
        "general-00000-of-00001.parquet",
        "multi_turn-00000-of-00001.parquet",
        "multimodal-00000-of-00001.parquet",
    ):
        (data_dir / name).write_text("cached", encoding="utf-8")
    archive = data_dir / "fixtures.tar.gz"
    source_file = tmp_path / "data.json"
    source_file.write_text("{}", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source_file, arcname="tasks/T001/fixtures/data.json")

    monkeypatch.setattr("host.adapters.claw_eval.CLAW_EVAL_DIR", claw_dir)
    monkeypatch.setattr("host.config.submodules.CLAW_EVAL_DIR", claw_dir)
    settings = _settings(tmp_path, tasks_dir)

    report = ClawEvalRunner.download_tasks(["claw-eval"], settings, check_only=False)

    assert report.failed == {}
    assert (tasks_dir / "T001" / "fixtures" / "data.json").is_file()


def test_claw_eval_download_retries_when_data_dir_is_partial(tmp_path, monkeypatch):
    claw_dir = tmp_path / "claw-eval"
    data_dir = claw_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "general-00000-of-00001.parquet").write_text("partial", encoding="utf-8")
    monkeypatch.setattr("host.adapters.claw_eval.CLAW_EVAL_DIR", claw_dir)

    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        argparse.Namespace(snapshot_download=fake_snapshot_download),
    )

    from host.adapters.claw_eval import _download_claw_eval_data

    _download_claw_eval_data(force=False)

    assert captured["repo_id"] == "claw-eval/Claw-Eval"
    assert captured["force_download"] is True


def test_cli_routes_claw_eval(monkeypatch):
    from host.cli import commands
    from host.cli.commands import _cmd_run

    captured = {}

    monkeypatch.setitem(commands._BACKEND_SUBMODULE_CHECKS, "claw-eval", lambda: None)

    def fake_run_with_controller(**kwargs):
        captured.update(kwargs)
        args = kwargs["build_args"]("model", "runid")
        captured["args"] = args
        return 0

    monkeypatch.setattr("host.cli.commands._run_with_controller", fake_run_with_controller)
    args = argparse.Namespace(
        config=Path("configs/base.yaml"),
        model="model",
        benchmarks=["claw-eval"],
        limit=1,
        task_id=["T001"],
        no_judge=True,
        claw_filter="T0",
        claw_tag="smoke",
        claw_range="1-3",
        claw_language="zh",
        claw_category="finance",
    )

    assert _cmd_run(args) == 0
    assert captured["benchmark"] == "claw-eval"
    assert captured["engine"] == "claw_eval"
    assert captured["args"].task_ids == ["T001"]
    assert captured["args"].no_judge is True
    assert captured["args"].tag == "smoke"
    assert captured["args"].language == "zh"


def test_parser_accepts_claw_eval_preflight_and_download_backend():
    from host.cli.parser import build_parser

    parser = build_parser()
    preflight = parser.parse_args(["preflight", "--benchmark", "claw-eval"])
    run = parser.parse_args(
        [
            "run",
            "claw-eval",
            "--no-judge",
            "--claw-language",
            "zh",
            "--claw-category",
            "finance",
        ]
    )
    download = parser.parse_args(["download", "claw-eval", "--check"])

    assert preflight.benchmark == "claw-eval"
    assert run.no_judge is True
    assert run.claw_language == "zh"
    assert run.claw_category == "finance"
    assert download.backend is None
    assert download.benchmarks == ["claw-eval"]


def test_submodule_status_includes_claw_eval(tmp_path, monkeypatch):
    from host.config.submodules import submodule_status

    monkeypatch.setattr("host.config.submodules.LMMS_EVAL_DIR", tmp_path / "lmms-eval")
    monkeypatch.setattr("host.config.submodules.TAU2_BENCH_DIR", tmp_path / "tau2-bench")
    claw_dir = tmp_path / "claw-eval"
    (claw_dir / "src" / "claw_eval").mkdir(parents=True)
    monkeypatch.setattr("host.config.submodules.CLAW_EVAL_DIR", claw_dir)

    assert submodule_status()["claw-eval"] is True


def test_cmd_download_dispatches_claw_eval(monkeypatch, tmp_path):
    from host.adapters import DownloadReport
    from host.cli import commands
    from host.cli.commands import _cmd_download

    captured = {}

    def fake_download(items, settings, *, check_only, force):
        captured.update({"items": items, "check_only": check_only, "force": force})
        return DownloadReport(backend="claw-eval", succeeded=list(items))

    monkeypatch.setitem(commands._BACKEND_SUBMODULE_CHECKS, "claw-eval", lambda: None)
    monkeypatch.setattr("host.cli.commands.load_settings", lambda config: Settings())
    monkeypatch.setattr("host.cli.commands.apply_download_cache_overrides", lambda s, **kw: s)
    monkeypatch.setattr(commands._BACKEND_RUNNERS["claw-eval"], "download_tasks", fake_download)
    args = argparse.Namespace(
        config=tmp_path / "config.yaml",
        backend=None,
        benchmarks=["claw-eval"],
        hub_cache=None,
        datasets_cache=None,
        check=True,
        force=True,
    )

    assert _cmd_download(args) == 0
    assert captured == {"items": ["claw-eval"], "check_only": True, "force": True}


# ---------------------------------------------------------------------------
# user_agent_model channel — dedicated config vs judge fallback
# ---------------------------------------------------------------------------


def test_claw_eval_user_agent_model_uses_dedicated_config(tmp_path, monkeypatch):
    """When claw_eval.user_agent_model_* is set, run config uses those values."""
    tasks_dir = tmp_path / "tasks"
    _task(tasks_dir, "C01")
    settings = _settings(tmp_path, tasks_dir)
    settings = settings.model_copy(
        update={
            "claw_eval": settings.claw_eval.model_copy(
                update={
                    "user_agent_model_url": "https://openrouter.ai/api/v1",
                    "user_agent_model": "google/gemini-3-flash-preview",
                }
            )
        }
    )
    monkeypatch.setenv("CLAW_USER_API_KEY", "sk-claw-ua")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    runner = ClawEvalRunner(settings)
    options = runner._run_options(ClawEvalRunArgs())
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = runner._write_run_config(
        run_dir=run_dir,
        tasks_dir=tasks_dir,
        trace_root=run_dir / "traces",
        model_name="Qwen",
        options=options,
    )
    payload = yaml.safe_load(config_path.read_text())
    assert payload["user_agent_model"]["api_key"] == "${CLAW_USER_API_KEY}"
    assert payload["user_agent_model"]["base_url"] == "https://openrouter.ai/api/v1"
    assert payload["user_agent_model"]["model_id"] == "google/gemini-3-flash-preview"
    # judge should still use judge credentials, not user-agent config
    assert payload["judge"]["base_url"] == "https://judge.example/v1"
    assert payload["judge"]["model_id"] == "judge-model"


def test_claw_eval_user_agent_model_falls_back_to_judge(tmp_path, monkeypatch):
    """When no dedicated user-agent config/env is set, run config falls back to judge."""
    tasks_dir = tmp_path / "tasks"
    _task(tasks_dir, "C01")
    settings = _settings(tmp_path, tasks_dir)
    monkeypatch.delenv("CLAW_USER_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    runner = ClawEvalRunner(settings)
    options = runner._run_options(ClawEvalRunArgs())
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = runner._write_run_config(
        run_dir=run_dir,
        tasks_dir=tasks_dir,
        trace_root=run_dir / "traces",
        model_name="Qwen",
        options=options,
    )
    payload = yaml.safe_load(config_path.read_text())
    assert payload["user_agent_model"]["api_key"] == "${CLAW_USER_API_KEY}"
    assert payload["user_agent_model"]["base_url"] == "https://judge.example/v1"
    assert payload["user_agent_model"]["model_id"] == "judge-model"
