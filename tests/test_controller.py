"""Tests for HostController lifecycle and run management."""

from __future__ import annotations

import pytest
from host.config import Settings
from host.controller import HostController


@pytest.fixture
def controller(monkeypatch):
    """Mock 掉 _llama 和 monitor 依赖的 HostController。"""
    from host.client.llama_cpp_client import ModelInfo, ServerProps

    class _FakeControl:
        def list_models(self, *, reload=False):
            return [ModelInfo(id="Qwen3.5-4B-Q4_K_M", status="loaded")]

        def get_model_status(self, model_id):
            return "loaded"

        def load_model(self, model_id):
            from host.client.llama_cpp_client import LoadModelResult

            return LoadModelResult(success=True, message="")

        def unload_model(self, model_id):
            return True

        def wait_until_loaded(self, model_id, *, timeout_s=300.0, poll_interval_s=2.0):
            return True

        def get_props(self, model_id=None):
            return ServerProps(total_slots=4, modalities=["text"])

        def health(self):
            return True

    class _FakeMonitor:
        def __init__(self, _settings):
            self._healthy = True
            self._device_info: dict | None = None
            self.health_calls = 0

        @property
        def device_info(self):
            return self._device_info

        def health(self):
            self.health_calls += 1
            if not self._healthy:
                raise ConnectionError("monitor down")
            self._device_info = {"gpu_model": "RTX 4090"}
            return {"status": "ok", "device_info": self._device_info}

        def create_trace(self, record):
            return {"trace_id": record.trace_id}

        def start_trace(self, trace_id):
            return {}

        def stop_trace(self, trace_id):
            return {}

        def mark(self, trace_id, stage, **extra):
            return {}

        def get_trace(self, trace_id):
            return {"metrics": {}}

    monkeypatch.setattr("host.controller.EvalMonitorClient", _FakeMonitor)

    settings = Settings(
        target={
            "inference_base_url": "http://edge:8080/v1",
            "model": "Qwen3.5-4B-Q4_K_M",
            "llama_cpp": {"mode": "router"},
        },
        eval={"enable_traces": True},
    )
    ctrl = HostController(settings)
    ctrl._llama.control = _FakeControl()  # type: ignore[assignment]
    ctrl._llama.health_check = lambda: True  # type: ignore[method-assign]
    return ctrl


# ===========================================================================
# preflight
# ===========================================================================


class TestPreflight:
    def test_preflight_passes(self, controller):
        status = controller.preflight()
        assert status["ready"] is True
        assert status["target_reachable"] is True
        assert controller.monitor.health_calls == 1

    def test_preflight_model_name_override_works(self, controller):
        """传入不同 model_name 时应使用它而非默认值。"""
        status = controller.preflight(model_name="custom-model")
        assert status["target_model"] == "custom-model"

    def test_preflight_target_unreachable(self, controller):
        controller._llama.health_check = lambda: False  # type: ignore[method-assign]
        status = controller.preflight()
        assert status["ready"] is False
        assert status["target_reachable"] is False

    def test_preflight_cfg_flags_control_readiness(self):
        """check_health / check_model_catalog flag 控制 readiness。"""
        from host.client.llama_cpp_client import LlamaCppClient, ModelInfo

        settings = Settings(
            target={
                "inference_base_url": "http://edge:8080/v1",
                "model": "Qwen3.5-4B-Q4_K_M",
                "llama_cpp": {
                    "mode": "router",
                    "preflight": {"check_health": True, "check_model_catalog": True},
                },
            },
        )
        ctrl = HostController(settings)
        ctrl._llama = LlamaCppClient(settings)

        class _FakeControl:
            def health(self):
                return False

            def list_models(self):
                return [ModelInfo(id="Qwen3.5-4B-Q4_K_M", status="loaded")]

            def get_props(self, model_id=None):
                from host.client.llama_cpp_client import ServerProps

                return ServerProps(modalities=["text"])

        ctrl._llama.control = _FakeControl()  # type: ignore
        ctrl._llama.health_check = lambda: True  # type: ignore[method-assign]

        status = ctrl.preflight()
        assert status["ready"] is False  # health check failed
        assert "warnings" in status

    def test_preflight_with_adapter(self, controller):
        """Adapter-specific readiness is delegated through the runner contract."""

        class _FakeRunner:
            def __init__(self, settings):
                pass

            def preflight(self):
                return True

        status = controller.preflight(adapter_cls=_FakeRunner)
        assert status.get("adapter_ready") is True
        assert status["ready"] is True

    def test_preflight_monitor_unreachable_nonblocking(self, controller):
        """monitor 不可达不应阻止 readiness。"""
        controller.monitor._healthy = False  # type: ignore
        status = controller.preflight()
        assert status["ready"] is True  # monitor 失败不阻塞
        assert any("monitor" in w for w in status.get("warnings", []))


# ===========================================================================
# prepare_for_run
# ===========================================================================


class TestPrepareForRun:
    def test_prepare_for_run_aborts_when_model_missing(self, controller, monkeypatch):
        def _raise_missing(*args, **kwargs):
            raise RuntimeError("model 'missing-model' not on Target")

        controller._llama.require_model_in_catalog = _raise_missing  # type: ignore
        with pytest.raises(RuntimeError):
            controller.prepare_for_run("missing-model")

    def test_prepare_for_run_verifies_loaded_model(self, controller):
        snapshot = controller.prepare_for_run("Qwen3.5-4B-Q4_K_M", run_id="test")
        assert snapshot["resolved_model_id"] == "Qwen3.5-4B-Q4_K_M"
        assert snapshot["model_status"] == "loaded"

    def test_prepare_for_run_trace_start_failure_does_not_block(self, controller, monkeypatch):
        """trace 异常被吞噬，不阻止 prepare_for_run。"""
        import io

        from loguru import logger

        def _fail_start_trace(*args, **kwargs):
            raise RuntimeError("monitor unavailable")

        controller.monitor.start_trace = _fail_start_trace  # type: ignore

        buf = io.StringIO()
        sink_id = logger.add(buf, format="{level} {message}", level="WARNING")

        snapshot = controller.prepare_for_run("Qwen3.5-4B-Q4_K_M", run_id="test", enable_trace=True)
        assert snapshot["resolved_model_id"] == "Qwen3.5-4B-Q4_K_M"
        logger.remove(sink_id)
        assert "failed to start run trace" in buf.getvalue()

    def test_prepare_for_run_unload_on_switch(self):
        """配置 unload_on_model_switch 时切换模型应卸载旧的。"""
        from host.client.llama_cpp_client import ModelInfo

        unloaded = []
        model_statuses = {"old-model": "loaded", "new-model": "unloaded"}

        class _FakeControl:
            def list_models(self, *, reload=False):
                return [ModelInfo(id=mid, status=status) for mid, status in model_statuses.items()]

            def get_model_status(self, model_id):
                return model_statuses.get(model_id, "unloaded")

            def load_model(self, model_id):
                from host.client.llama_cpp_client import LoadModelResult

                model_statuses[model_id] = "loaded"
                return LoadModelResult(success=True, message="")

            def unload_model(self, model_id):
                unloaded.append(model_id)
                model_statuses[model_id] = "unloaded"
                return True

            def wait_until_loaded(self, model_id, *, timeout_s=300.0, poll_interval_s=2.0):
                return True

            def get_props(self, model_id=None):
                from host.client.llama_cpp_client import ServerProps

                return ServerProps(modalities=["text"])

            def health(self):
                return True

        settings = Settings(
            target={
                "inference_base_url": "http://edge:8080/v1",
                "model": "new-model",
                "llama_cpp": {
                    "mode": "router",
                    "lifecycle": {"unload_on_model_switch": True, "ensure_loaded": True},
                },
            },
        )
        ctrl = HostController(settings)
        ctrl._llama.control = _FakeControl()  # type: ignore

        ctrl.prepare_for_run("new-model")
        assert "old-model" in unloaded


# ===========================================================================
# cleanup_after_run
# ===========================================================================


class TestCleanupAfterRun:
    def test_cleanup_after_run_stops_trace_and_returns_snapshot(self, controller, monkeypatch):
        """cleanup 正确停止 trace、返回 snapshot。"""
        # 先 prepare 启动 trace
        controller.prepare_for_run("Qwen3.5-4B-Q4_K_M", run_id="test-run", enable_trace=True)

        snapshot = controller.cleanup_after_run("Qwen3.5-4B-Q4_K_M", enable_trace=True)
        assert "trace_metrics" in snapshot
        assert snapshot["healthy"] is True

    def test_cleanup_after_run_trace_failure_swallowed(self, controller):
        """cleanup 时 trace 异常也被吞噬。"""

        def _fail_stop(*args, **kwargs):
            raise RuntimeError("stop failed")

        controller.monitor.stop_trace = _fail_stop  # type: ignore
        controller.prepare_for_run("Qwen3.5-4B-Q4_K_M", run_id="test-run", enable_trace=True)

        # cleanup 不应抛异常
        snapshot = controller.cleanup_after_run("Qwen3.5-4B-Q4_K_M", enable_trace=True)
        assert "trace_metrics" in snapshot

    def test_cleanup_after_run_optionally_unloads(self, controller, monkeypatch):
        """unload_after_run=True 时在 cleanup 中卸载模型。"""
        unloaded = []

        original_unload = controller._llama.unload_model

        def _track_unload(model_name):
            unloaded.append(model_name)
            return True

        controller._llama.unload_model = _track_unload  # type: ignore

        # 临时修改配置
        controller._llama_cfg.lifecycle.unload_after_run = True  # type: ignore

        controller.cleanup_after_run("Qwen3.5-4B-Q4_K_M")
        assert "Qwen3.5-4B-Q4_K_M" in unloaded

        # 恢复
        controller._llama.unload_model = original_unload

    def test_cleanup_after_run_without_trace(self, controller):
        """没有启动 trace 时 cleanup 不应出错。"""
        snapshot = controller.cleanup_after_run("Qwen3.5-4B-Q4_K_M", enable_trace=False)
        assert snapshot["trace_metrics"] == {}


# ===========================================================================
# archive_run_manifest
# ===========================================================================


class TestArchiveRunManifest:
    def test_archive_run_manifest_writes_correct_json(self, controller, tmp_path):
        manifest_path = controller.archive_run_manifest(
            run_id="run_test123",
            engine="lmms_eval",
            benchmark="mmmu_val",
            work_dir=tmp_path,
            success=True,
            extra={"key": "value"},
        )
        assert manifest_path.is_file()
        assert manifest_path.name == "manifest.json"
        import json

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert payload["run_id"] == "run_test123"
        assert payload["engine"] == "lmms_eval"
        assert payload["benchmark"] == "mmmu_val"
        assert payload["success"] is True
        assert payload["extra"]["key"] == "value"

    def test_archive_run_manifest_serializes_callables(self, controller, tmp_path):
        def sample_fn() -> None:
            return None

        manifest_path = controller.archive_run_manifest(
            run_id="run_callable",
            engine="lmms_eval",
            benchmark="mmmu_val",
            work_dir=tmp_path,
            success=True,
            extra={
                "lmms_eval": {
                    "configs": {
                        "mmmu_val": {
                            "process_results": sample_fn,
                            "task": "mmmu_val",
                        }
                    }
                }
            },
        )
        import json

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert payload["extra"]["lmms_eval"]["configs"]["mmmu_val"]["task"] == "mmmu_val"
        assert payload["extra"]["lmms_eval"]["configs"]["mmmu_val"]["process_results"] == (
            "TestArchiveRunManifest.test_archive_run_manifest_serializes_callables."
            "<locals>.sample_fn"
        )

    def test_archive_run_manifest_creates_parent_dir(self, controller, tmp_path):
        deep_dir = tmp_path / "does" / "not" / "exist"
        assert not deep_dir.exists()
        controller.archive_run_manifest(
            run_id="deep_test",
            engine="tb",
            benchmark="terminal-bench",
            work_dir=deep_dir,
            success=False,
        )
        assert deep_dir.is_dir()

    def test_archive_run_manifest_uses_target_model_override(self, controller, tmp_path):
        import json

        manifest_path = controller.archive_run_manifest(
            run_id="run_override",
            engine="lmms_eval",
            benchmark="mmmu_val",
            work_dir=tmp_path,
            success=True,
            target_model="Qwen3.6-35B-A3B-UD-Q4_K_M",
        )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert payload["settings"]["target_model"] == "Qwen3.6-35B-A3B-UD-Q4_K_M"


# ===========================================================================
# HostController.run
# ===========================================================================


class TestHostControllerRun:
    def test_run_cleanup_on_adapter_failure(self, controller, monkeypatch):
        """adapter 抛出异常后也必须执行 cleanup。"""
        cleaned_up = []

        def _track_cleanup(model_name, *, enable_trace=None):
            cleaned_up.append(model_name)
            return {"trace_metrics": {}}

        controller.cleanup_after_run = _track_cleanup  # type: ignore

        class _FakeAdapter:
            def __init__(self, settings):
                pass

            def run(self, args, *, output_dir=None):
                raise RuntimeError("adapter crashed")

        with pytest.raises(RuntimeError, match="adapter"):
            controller.run(
                model="Qwen3.5-4B-Q4_K_M",
                benchmark="mmmu_val",
                engine="lmms_eval",
                adapter_cls=_FakeAdapter,
                build_args=lambda m, run_id: {"run_id": run_id},
            )

        assert "Qwen3.5-4B-Q4_K_M" in cleaned_up

    def test_run_success(self, controller, monkeypatch, tmp_path):
        """HostController.run 成功路径。"""

        class _FakeAdapter:
            def __init__(self, settings):
                self.settings = settings

            def run(self, args, *, output_dir=None):
                from host.adapters import RunResult

                return RunResult(
                    success=True,
                    run_id=args["run_id"],
                    model="Qwen3.5-4B-Q4_K_M",
                    metric="exact_match",
                    value=0.85,
                    n_total=10,
                    n_passed=8,
                    summary={"key": "value"},
                )

        # 覆盖 work_dir 到 tmp_path
        controller.settings.eval.work_dir = tmp_path

        result = controller.run(
            model="Qwen3.5-4B-Q4_K_M",
            benchmark="mmmu_val",
            engine="lmms_eval",
            adapter_cls=_FakeAdapter,
            build_args=lambda m, run_id: {"run_id": run_id},
        )
        assert result.success is True
        assert result.metric == "exact_match"
        assert result.value == 0.85
        summary_path = next(tmp_path.rglob("summary.json"))
        import json

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["metric"] == "exact_match"
        assert summary["value"] == 0.85
        assert "accuracy" not in summary

    def test_run_rejects_adapter_run_id_mismatch(self, controller, tmp_path):
        """Adapter results must use the controller-owned run_id."""

        class _FakeAdapter:
            def __init__(self, settings):
                self.settings = settings

            def run(self, args, *, output_dir=None):
                from host.adapters import RunResult

                return RunResult(
                    success=True,
                    run_id="adapter-owned-run",
                    model="Qwen3.5-4B-Q4_K_M",
                    metric="exact_match",
                    value=0.85,
                    n_total=10,
                    n_passed=8,
                    summary={},
                )

        controller.settings.eval.work_dir = tmp_path

        with pytest.raises(RuntimeError, match="mismatched run_id"):
            controller.run(
                model="Qwen3.5-4B-Q4_K_M",
                benchmark="mmmu_val",
                engine="lmms_eval",
                adapter_cls=_FakeAdapter,
                build_args=lambda m, run_id: {"run_id": run_id},
            )

    def test_run_reuses_controller_device_info(self, controller, monkeypatch, tmp_path):
        """Preflight 填充的 device_info 应写入 manifest（复用同一 controller）。"""
        import json

        controller.preflight()

        class _FakeAdapter:
            def __init__(self, settings):
                self.settings = settings

            def run(self, args, *, output_dir=None):
                from host.adapters import RunResult

                return RunResult(
                    success=True,
                    run_id=args["run_id"],
                    model="Qwen3.5-4B-Q4_K_M",
                    metric="exact_match",
                    value=0.85,
                    n_total=10,
                    n_passed=8,
                    summary={},
                )

        controller.settings.eval.work_dir = tmp_path
        controller.run(
            model="Qwen3.5-4B-Q4_K_M",
            benchmark="mmmu_val",
            engine="lmms_eval",
            adapter_cls=_FakeAdapter,
            build_args=lambda m, run_id: {"run_id": run_id},
        )

        manifests = list(tmp_path.rglob("manifest.json"))
        assert len(manifests) == 1
        payload = json.loads(manifests[0].read_text(encoding="utf-8"))
        assert payload["device_info"]["gpu_model"] == "RTX 4090"
