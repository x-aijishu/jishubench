"""Tests for the lmms-eval adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from host.config import Settings

# ===========================================================================
# lmms_eval — _resolve_tasks
# ===========================================================================


def test_lmms_preflight_checks_submodule_and_package(monkeypatch):
    from host.adapters.lmms_eval import LmmsEvalRunner

    monkeypatch.setattr("host.adapters.lmms_eval.ensure_lmms_eval", lambda: None)
    monkeypatch.setattr("host.adapters.lmms_eval.util.find_spec", lambda name: object())

    assert LmmsEvalRunner(Settings()).preflight() is True


class TestLmmsResolveTasks:
    def test_resolve_tasks_with_wildcards_skips_missing_check(self):
        """通配符任务名跳过 missing 检查。"""
        from host.adapters.lmms_eval import LmmsEvalRunner

        class FakeTaskManager:
            all_tasks = ["mmmu_val", "realworldqa"]

            def match_tasks(self, tasks):
                return [t for t in tasks if t in self.all_tasks or "*" in t]

        # 通配符不会报错
        result = LmmsEvalRunner._resolve_tasks(FakeTaskManager(), ["mmmu*", "realworldqa"])
        assert "mmmu*" in result
        assert "realworldqa" in result

    def test_resolve_tasks_all_missing_raises(self):
        from host.adapters.lmms_eval import LmmsEvalRunner

        class FakeTaskManager:
            all_tasks = ["mmmu_val"]

            def match_tasks(self, tasks):
                return []

        with pytest.raises(ValueError, match="missing_task"):
            LmmsEvalRunner._resolve_tasks(FakeTaskManager(), ["missing_task"])

    def test_resolve_tasks_partial_missing_raises(self):
        from host.adapters.lmms_eval import LmmsEvalRunner

        class FakeTaskManager:
            all_tasks = ["mmmu_val"]

            def match_tasks(self, tasks):
                return ["mmmu_val"]

        with pytest.raises(ValueError, match="unknown_task"):
            LmmsEvalRunner._resolve_tasks(FakeTaskManager(), ["mmmu_val", "unknown_task"])

    def test_resolve_tasks_shows_available_in_error(self):
        from host.adapters.lmms_eval import LmmsEvalRunner

        class FakeTaskManager:
            all_tasks = [f"task_{i}" for i in range(30)]

            def match_tasks(self, tasks):
                return []

        with pytest.raises(ValueError) as excinfo:
            LmmsEvalRunner._resolve_tasks(FakeTaskManager(), ["ghost"])
        # 只显示前 20 个
        assert "task_19" in str(excinfo.value)


# ===========================================================================
# lmms_eval — run / metric extraction
# ===========================================================================


class TestLmmsRunnerMetrics:
    def _patch_lmms_mocks(self, monkeypatch, fake_simple_evaluate, fake_results=None):
        """设置 lmms_eval runner 的通用 mock 环境。"""

        class _FakeTaskManager:
            all_tasks = ["task_a", "task_b"]

            def match_tasks(self, tasks):
                return list(tasks)

            def __init__(self, **kwargs):
                pass

        class _FakeTracker:
            def __init__(self, output_path=None):
                self.general_config_tracker = SimpleNamespace(model_source=object())

            def save_results_aggregated(self, *a, **kw):
                pass

            def save_results_samples(self, *a, **kw):
                pass

        monkeypatch.setattr("lmms_eval.evaluator.simple_evaluate", fake_simple_evaluate)
        monkeypatch.setattr("lmms_eval.tasks.TaskManager", _FakeTaskManager)
        monkeypatch.setattr("lmms_eval.loggers.EvaluationTracker", _FakeTracker)
        monkeypatch.setattr(
            "lmms_eval.models.get_model",
            lambda *a, **kw: lambda **kwargs: object(),
        )
        monkeypatch.setattr("lmms_eval.utils.get_datetime_str", lambda: "20250101_000000")
        monkeypatch.setattr("host.adapters.lmms_eval.ensure_lmms_eval", lambda: None)
        monkeypatch.setattr(
            "host.adapters.lmms_eval.lmms_eval_env",
            lambda s: __import__("contextlib").nullcontext(),
        )

    def test_metric_extraction_handles_zero(self, monkeypatch):
        """验证 exact_match=0.0 不被 or 吞噬"""
        from host.adapters.lmms_eval import LmmsEvalRunArgs, LmmsEvalRunner

        fake_results = {
            "results": {
                "task_a": {"exact_match": 0.0, "acc": 0.9},
                "task_b": {"exact_match": 1.0},
            },
            "configs": {},
            "version": "test",
        }

        runner = LmmsEvalRunner(Settings())

        def fake_simple_evaluate(**kwargs):
            return fake_results

        self._patch_lmms_mocks(monkeypatch, fake_simple_evaluate, fake_results)

        result = runner.run(LmmsEvalRunArgs(tasks=["task_a", "task_b"], model_name="m"))

        assert result.metric is None
        assert result.value == {
            "task_a": {"metric": "exact_match", "value": 0.0},
            "task_b": {"metric": "exact_match", "value": 1.0},
        }
        assert result.n_total == 0
        assert result.n_passed == 0

    def test_metric_extraction_handles_lmms_eval_filter_suffix(self, monkeypatch):
        """lmms-eval 聚合结果使用 metric,filter 形式的 key。"""
        from host.adapters.lmms_eval import LmmsEvalRunArgs, LmmsEvalRunner

        fake_results = {
            "results": {
                "realworldqa": {
                    "alias": "realworldqa",
                    "exact_match,none": 0.7725490196078432,
                    "exact_match_stderr,none": 0.015165622805097612,
                    "samples": 765,
                }
            },
            "n-samples": {"realworldqa": {"original": 765, "effective": 765}},
            "configs": {
                "realworldqa": {"metric_list": [{"metric": "exact_match", "aggregation": "mean"}]}
            },
            "version": "test",
        }

        def fake_simple_evaluate(**kwargs):
            return fake_results

        self._patch_lmms_mocks(monkeypatch, fake_simple_evaluate)
        runner = LmmsEvalRunner(Settings())
        result = runner.run(LmmsEvalRunArgs(tasks=["realworldqa"], model_name="m"))

        assert result.metric == "exact_match"
        assert result.value == pytest.approx(0.7725490196078432)
        assert result.n_total == 765
        assert result.n_passed == 591

    def test_metric_extraction_prefers_task_config_metric(self, monkeypatch):
        """优先使用 task config 中声明的主 metric。"""
        from host.adapters.lmms_eval import LmmsEvalRunArgs, LmmsEvalRunner

        fake_results = {
            "results": {
                "task_a": {
                    "exact_match,none": 0.2,
                    "acc_score,none": 0.8,
                }
            },
            "configs": {
                "task_a": {"metric_list": [{"metric": "acc_score", "aggregation": "mean"}]}
            },
            "version": "test",
        }

        def fake_simple_evaluate(**kwargs):
            return fake_results

        self._patch_lmms_mocks(monkeypatch, fake_simple_evaluate)
        runner = LmmsEvalRunner(Settings())
        result = runner.run(LmmsEvalRunArgs(tasks=["task_a"], model_name="m"))

        assert result.metric == "acc_score"
        assert result.value == 0.8

    def test_metric_extraction_prefers_headline_average_for_mmstar(self, monkeypatch):
        """多 metric 任务应优先使用 average 等汇总指标，而非第一个子类。"""
        from host.adapters.lmms_eval import LmmsEvalRunArgs, LmmsEvalRunner

        fake_results = {
            "results": {
                "mmstar": {
                    "alias": "mmstar",
                    "coarse perception,none": 0.726489860326511,
                    "fine-grained perception,none": 0.5454986489805944,
                    "average,none": 0.6253840151899839,
                }
            },
            "n-samples": {"mmstar": {"original": 1500, "effective": 1500}},
            "configs": {
                "mmstar": {
                    "metric_list": [
                        {"metric": "coarse perception"},
                        {"metric": "fine-grained perception"},
                        {"metric": "average"},
                    ]
                }
            },
            "version": "test",
        }

        def fake_simple_evaluate(**kwargs):
            return fake_results

        self._patch_lmms_mocks(monkeypatch, fake_simple_evaluate)
        runner = LmmsEvalRunner(Settings())
        result = runner.run(LmmsEvalRunArgs(tasks=["mmstar"], model_name="m"))

        assert result.metric == "average"
        assert result.value == pytest.approx(0.6253840151899839)
        assert result.n_total == 1500
        assert result.n_passed == 938

    def test_metric_extraction_ignores_stderr_keys(self, monkeypatch):
        """stderr 类 key 不应被当作主 metric。"""
        from host.adapters.lmms_eval import LmmsEvalRunArgs, LmmsEvalRunner

        fake_results = {
            "results": {
                "task_a": {
                    "exact_match_stderr,none": 0.01,
                }
            },
            "configs": {},
            "version": "test",
        }

        def fake_simple_evaluate(**kwargs):
            return fake_results

        self._patch_lmms_mocks(monkeypatch, fake_simple_evaluate)
        runner = LmmsEvalRunner(Settings())
        result = runner.run(LmmsEvalRunArgs(tasks=["task_a"], model_name="m"))

        assert result.metric is None
        assert result.value is None

    def test_metric_extraction_weighted_across_tasks(self, monkeypatch):
        """多样本任务按样本数加权汇总 n_passed。"""
        from host.adapters.lmms_eval import LmmsEvalRunArgs, LmmsEvalRunner

        fake_results = {
            "results": {
                "task_a": {"exact_match,none": 0.0, "samples": 100},
                "task_b": {"exact_match,none": 1.0, "samples": 100},
            },
            "configs": {},
            "version": "test",
        }

        def fake_simple_evaluate(**kwargs):
            return fake_results

        self._patch_lmms_mocks(monkeypatch, fake_simple_evaluate)
        runner = LmmsEvalRunner(Settings())
        result = runner.run(LmmsEvalRunArgs(tasks=["task_a", "task_b"], model_name="m"))

        assert result.metric is None
        assert result.value == {
            "task_a": {"metric": "exact_match", "value": 0.0},
            "task_b": {"metric": "exact_match", "value": 1.0},
        }
        assert result.n_total == 200
        assert result.n_passed == 100

    def test_metric_extraction_heterogeneous_multi_task(self, monkeypatch):
        """多 task 时每个分数自带 metric，不另建平行索引。"""
        from host.adapters.lmms_eval import LmmsEvalRunArgs, LmmsEvalRunner

        fake_results = {
            "results": {
                "task_a": {"exact_match,none": 0.5, "samples": 10},
                "task_b": {"average,none": 0.8, "samples": 10},
            },
            "configs": {
                "task_a": {"metric_list": [{"metric": "exact_match"}]},
                "task_b": {"metric_list": [{"metric": "average"}]},
            },
            "version": "test",
        }

        def fake_simple_evaluate(**kwargs):
            return fake_results

        self._patch_lmms_mocks(monkeypatch, fake_simple_evaluate)
        runner = LmmsEvalRunner(Settings())
        result = runner.run(LmmsEvalRunArgs(tasks=["task_a", "task_b"], model_name="m"))

        assert result.metric is None
        assert result.value == {
            "task_a": {"metric": "exact_match", "value": 0.5},
            "task_b": {"metric": "average", "value": 0.8},
        }

    def test_lmms_runner_run_returns_failure_on_none_results(self, monkeypatch):
        """simple_evaluate 返回 None 时应返回 success=False 的 RunResult。"""
        from host.adapters.lmms_eval import LmmsEvalRunArgs, LmmsEvalRunner

        def fake_simple_evaluate(**kwargs):
            return None

        self._patch_lmms_mocks(monkeypatch, fake_simple_evaluate)

        runner = LmmsEvalRunner(Settings())
        result = runner.run(LmmsEvalRunArgs(tasks=["task_a"], model_name="m"))
        assert result.success is False
        assert result.metric is None
        assert result.value is None
        assert result.n_total == 0

    def test_lmms_runner_run_empty_results_dict(self, monkeypatch):
        """空的 results dict 应返回 success=True 但 value=None。"""
        from host.adapters.lmms_eval import LmmsEvalRunArgs, LmmsEvalRunner

        fake_results = {"results": {}, "configs": {}, "version": "test"}

        def fake_simple_evaluate(**kwargs):
            return fake_results

        self._patch_lmms_mocks(monkeypatch, fake_simple_evaluate, fake_results)

        runner = LmmsEvalRunner(Settings())
        result = runner.run(LmmsEvalRunArgs(tasks=["task_a"], model_name="m"))
        assert result.success is True
        assert result.metric is None
        assert result.value is None
        assert result.n_total == 0

    def test_lmms_runner_passes_cli_args_output_path(self, monkeypatch, tmp_path):
        """GPT-judge tasks need cli_args.output_path during metric aggregation."""
        from host.adapters.lmms_eval import LmmsEvalRunArgs, LmmsEvalRunner

        fake_results = {"results": {}, "configs": {}, "version": "test"}
        captured: dict = {}

        def fake_simple_evaluate(**kwargs):
            captured.update(kwargs)
            return fake_results

        self._patch_lmms_mocks(monkeypatch, fake_simple_evaluate, fake_results)

        settings = Settings(eval={"work_dir": tmp_path})
        runner = LmmsEvalRunner(settings)
        runner.run(
            LmmsEvalRunArgs(tasks=["task_a"], model_name="m", run_id="test-run"),
            output_dir=tmp_path / "run",
        )

        cli_args = captured.get("cli_args")
        assert cli_args is not None
        assert cli_args.output_path == str((tmp_path / "run").resolve())
        assert cli_args.process_with_media is False

    def test_lmms_runner_enables_process_with_media_when_logging_samples(
        self, monkeypatch, tmp_path
    ):
        from host.adapters.lmms_eval import LmmsEvalRunArgs, LmmsEvalRunner

        fake_results = {
            "results": {"task_a": {"acc,none": 1.0}},
            "configs": {},
            "version": "test",
            "samples": {"task_a": [{"doc_id": 0}]},
        }
        captured: dict = {}

        def fake_simple_evaluate(**kwargs):
            captured.update(kwargs)
            return fake_results

        self._patch_lmms_mocks(monkeypatch, fake_simple_evaluate, fake_results)

        settings = Settings(eval={"work_dir": tmp_path})
        runner = LmmsEvalRunner(settings)
        runner.run(
            LmmsEvalRunArgs(
                tasks=["task_a"],
                model_name="m",
                save_samples=True,
                run_id="test-run",
            ),
            output_dir=tmp_path / "run",
        )

        cli_args = captured.get("cli_args")
        assert cli_args is not None
        assert cli_args.process_with_media is True
        assert captured.get("log_samples") is True

    def test_lmms_runner_keeps_target_key_out_of_persisted_model_args(self, monkeypatch, tmp_path):
        from host.adapters.lmms_eval import LmmsEvalRunArgs, LmmsEvalRunner

        target_key = "target-secret-sentinel"
        judge_key = "judge-secret-sentinel"
        captured: dict = {}

        class _FakeModel:
            def __init__(self, **kwargs):
                captured["model_kwargs"] = kwargs

        def fake_simple_evaluate(**kwargs):
            captured["evaluate_kwargs"] = kwargs
            return {
                "results": {},
                "configs": {},
                "version": "test",
                "config": {"model_args": kwargs["model_args"]},
            }

        self._patch_lmms_mocks(monkeypatch, fake_simple_evaluate)
        monkeypatch.setattr("lmms_eval.models.get_model", lambda *a, **kw: _FakeModel)

        settings = Settings(
            target={"inference_api_key": target_key},
            judge={"openai_api_key": judge_key},
            eval={"work_dir": tmp_path},
        )
        result = LmmsEvalRunner(settings).run(LmmsEvalRunArgs(tasks=["task_a"], model_name="m"))

        assert captured["model_kwargs"]["api_key"] == target_key
        assert captured["model_kwargs"]["api_key"] != judge_key
        assert not isinstance(captured["evaluate_kwargs"]["model"], str)
        persisted_args = captured["evaluate_kwargs"]["model_args"]
        assert "api_key" not in persisted_args
        assert target_key not in json.dumps(result.raw)

    def test_lmms_runner_redacts_known_secrets_before_saving(self, monkeypatch, tmp_path):
        from host.adapters.lmms_eval import LmmsEvalRunArgs, LmmsEvalRunner

        target_key = "target-secret-sentinel"
        judge_key = "judge-secret-sentinel"
        saved: dict = {}

        class _CapturingTracker:
            def __init__(self, output_path=None):
                self.general_config_tracker = SimpleNamespace(model_source=object())

            def save_results_aggregated(self, *a, **kwargs):
                saved["results"] = kwargs["results"]
                saved["samples"] = kwargs["samples"]
                saved["model_source"] = self.general_config_tracker.model_source

            def save_results_samples(self, *, task_name, samples):
                saved.setdefault("task_samples", {})[task_name] = samples

        def fake_simple_evaluate(**kwargs):
            return {
                "results": {"task_a": {"acc,none": 1.0, "note": target_key}},
                "configs": {},
                "version": "test",
                "config": {"model_args": f"api_key={target_key}"},
                "samples": {"task_a": [{"judge_debug": judge_key}]},
            }

        self._patch_lmms_mocks(monkeypatch, fake_simple_evaluate)
        monkeypatch.setattr("lmms_eval.loggers.EvaluationTracker", _CapturingTracker)

        settings = Settings(
            target={"inference_api_key": target_key},
            judge={"openai_api_key": judge_key},
            eval={"work_dir": tmp_path},
        )
        result = LmmsEvalRunner(settings).run(
            LmmsEvalRunArgs(tasks=["task_a"], model_name="m", save_samples=True)
        )

        persisted = json.dumps({"saved": saved, "raw": result.raw})
        assert target_key not in persisted
        assert judge_key not in persisted
        assert "[REDACTED]" in persisted
        assert saved["model_source"] == "async_openai"


# ===========================================================================
# lmms_eval — download_tasks
# ===========================================================================


class TestLmmsDownloadTasks:
    def test_download_backend_resolves_tasks(self, monkeypatch):
        import sys
        import types
        from contextlib import nullcontext

        from host.adapters import lmms_eval as lmms_mod
        from host.adapters.lmms_eval import LmmsEvalRunner

        load_calls = []

        class FakeTaskManager:
            all_tasks = ["mmmu_val", "realworldqa"]

            def match_tasks(self, tasks):
                return tasks

            def _get_config(self, name):
                if name == "realworldqa":
                    return {"task": name, "dataset_kwargs": {"token": True}}
                return {"task": name}

            def load_task_or_group(self, args):
                load_calls.append(args)
                return {args[0]["task"]: object()}

        monkeypatch.setattr(lmms_mod, "ensure_lmms_eval", lambda: None)
        monkeypatch.setattr(lmms_mod, "lmms_eval_env", lambda settings: nullcontext())

        fake_tasks = types.ModuleType("lmms_eval.tasks")
        fake_tasks.TaskManager = lambda **kw: FakeTaskManager()
        monkeypatch.setitem(sys.modules, "lmms_eval.tasks", fake_tasks)

        report = LmmsEvalRunner.download_tasks(
            ["mmmu_val", "realworldqa"],
            settings=Settings(),
        )
        assert not report.failed
        assert report.succeeded == ["mmmu_val", "realworldqa"]
        assert len(load_calls) == 2

    def test_download_backend_check_only_merges_kwargs(self, monkeypatch):
        import sys
        import types
        from contextlib import nullcontext

        from host.adapters import lmms_eval as lmms_mod
        from host.adapters.lmms_eval import LmmsEvalRunner

        load_calls = []

        class FakeTaskManager:
            all_tasks = ["realworldqa"]

            def match_tasks(self, tasks):
                return tasks

            def _get_config(self, name):
                return {"task": name, "dataset_kwargs": {"token": True}}

            def load_task_or_group(self, args):
                load_calls.append(args)
                return {args[0]["task"]: object()}

        monkeypatch.setattr(lmms_mod, "ensure_lmms_eval", lambda: None)
        monkeypatch.setattr(lmms_mod, "lmms_eval_env", lambda settings: nullcontext())

        fake_tasks = types.ModuleType("lmms_eval.tasks")
        fake_tasks.TaskManager = lambda **kw: FakeTaskManager()
        monkeypatch.setitem(sys.modules, "lmms_eval.tasks", fake_tasks)

        LmmsEvalRunner.download_tasks(
            ["realworldqa"],
            settings=Settings(),
            check_only=True,
        )
        args = load_calls[0][0]
        assert args["dataset_kwargs"]["token"] is True
        assert args["dataset_kwargs"]["local_files_only"] is True

    def test_download_report_empty_items(self):
        """空 items 列表返回 vacuous success。"""
        from host.adapters.lmms_eval import LmmsEvalRunner

        report = LmmsEvalRunner.download_tasks([], settings=Settings())
        assert not report.failed
        assert report.succeeded == []


# ===========================================================================
# lmms_eval — 辅助函数
# ===========================================================================


class TestLmmsHelpers:
    def test_merge_dataset_kwargs_base_none(self):
        from host.adapters.lmms_eval import _merge_dataset_kwargs

        merged = _merge_dataset_kwargs(None, {"local_files_only": True})
        assert merged == {"local_files_only": True}

    def test_merge_dataset_kwargs_overrides_win(self):
        from host.adapters.lmms_eval import _merge_dataset_kwargs

        merged = _merge_dataset_kwargs(
            {"token": True, "local_files_only": False},
            {"local_files_only": True},
        )
        assert merged["local_files_only"] is True
        assert merged["token"] is True

    def test_build_load_args_no_overrides(self):
        from host.adapters.lmms_eval import _build_load_args

        class FakeTM:
            def _get_config(self, name):
                return {"task": name}

        args = _build_load_args(FakeTM(), "mmmu_val", check_only=False, force=False)
        assert args == {"task": "mmmu_val"}

    def test_build_load_args_with_force(self):
        from host.adapters.lmms_eval import _build_load_args

        class FakeTM:
            def _get_config(self, name):
                return {"task": name, "dataset_kwargs": {}}

        args = _build_load_args(FakeTM(), "mmmu_val", check_only=False, force=True)
        assert "dataset_kwargs" in args
        assert args["dataset_kwargs"]["force_download"] is True

    def test_apply_download_cache_overrides_both_none(self):
        from host.adapters.lmms_eval import apply_download_cache_overrides

        settings = Settings()
        result = apply_download_cache_overrides(settings)
        assert result is settings  # no copy

    def test_apply_download_cache_overrides_with_paths(self, tmp_path):
        from host.adapters.lmms_eval import apply_download_cache_overrides

        settings = Settings(lmms_eval={"hub_cache": None, "datasets_cache": None})
        hub = tmp_path / "hub"
        datasets = tmp_path / "datasets"
        updated = apply_download_cache_overrides(settings, hub_cache=hub, datasets_cache=datasets)
        assert updated.lmms_eval.hub_cache == hub
        assert updated.lmms_eval.datasets_cache == datasets
        # original unchanged
        assert settings.lmms_eval.hub_cache is None

    def test_ensure_cache_dirs_creates_dirs(self, tmp_path):
        from host.adapters.lmms_eval import _ensure_cache_dirs

        settings = Settings(
            lmms_eval={"hub_cache": tmp_path / "hub", "datasets_cache": tmp_path / "datasets"}
        )
        hub_resolved, datasets_resolved = _ensure_cache_dirs(settings)
        assert (tmp_path / "hub").is_dir()
        assert (tmp_path / "datasets").is_dir()
        assert hub_resolved is not None
        assert datasets_resolved is not None

    def test_ensure_cache_dirs_none_paths(self):
        from host.adapters.lmms_eval import _ensure_cache_dirs

        settings = Settings(lmms_eval={"hub_cache": None, "datasets_cache": None})
        hub_resolved, datasets_resolved = _ensure_cache_dirs(settings)
        assert hub_resolved is None
        assert datasets_resolved is None
