"""Host Controller: preflight checks, model lifecycle, and run orchestration.

Ownership:
- LlamaCppClient singleton (one per process)
- EvalMonitorClient singleton (one per process)
- preflight / prepare_for_run / cleanup_after_run
- run manifest archival
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from loguru import logger
from pydantic import TypeAdapter

from host.adapters import Runner, RunResult
from host.client.eval_monitor_client import EvalMonitorClient, TraceRecord
from host.client.llama_cpp_client import LlamaCppClient
from host.config import Settings, load_settings, submodule_status
from host.config.logging_setup import setup_run_logging


class HostController:
    """Manages preflight checks, model lifecycle, and run manifest archival.

    - check_ready()
    - prepare_for_run()
    - cleanup_after_run()
    - archive_run_manifest()
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.monitor = EvalMonitorClient(self.settings)
        self._llama = LlamaCppClient(self.settings)
        self._llama_cfg = self.settings.target.llama_cpp
        self._run_trace_id: str | None = None

    def preflight(
        self,
        model_name: str | None = None,
        *,
        adapter_cls: type[Runner] | None = None,
    ) -> dict[str, Any]:
        """Run all configured preflight checks and return a readiness dict."""
        model_name = model_name or self.settings.target.model
        warnings: list[str] = []

        # data-plane reachability (always checked)
        target_reachable = self._llama.health_check()
        if not target_reachable:
            warnings.append(
                f"target inference endpoint unreachable at "
                f"{self.settings.target.inference_base_url}"
            )

        llama_cpp: dict[str, Any] = {}
        if self._llama_cfg:
            llama_cpp = self._llama.readiness_snapshot(model_name)

        # monitor reachability (optional — failure doesn't block eval)
        monitor_health: dict[str, Any] | None = None
        try:
            monitor_health = self.monitor.health()
        except Exception as exc:
            warnings.append(f"monitor unreachable (non-blocking): {exc}")

        # Determine overall readiness
        ready = target_reachable
        if self._llama_cfg:
            if self._llama_cfg.preflight.check_health and not llama_cpp.get("healthy"):
                warnings.append("llama.cpp /health check failed")
                ready = False
            if self._llama_cfg.preflight.check_model_catalog and not llama_cpp.get("model_found"):
                warnings.append(f"model {model_name!r} not present in Target catalog")
                ready = False
            if (
                llama_cpp.get("model_found")
                and llama_cpp.get("model_status") == "loaded"
                and not llama_cpp.get("props_ok")
            ):
                warnings.append("llama.cpp /props check failed")

        adapter_ready: bool | None = None
        if adapter_cls is not None:
            adapter_ready = adapter_cls(self.settings).preflight()
            ready = ready and adapter_ready

        submodules = submodule_status()
        # submodules
        for name, ok in submodules.items():
            if ok:
                logger.info("submodule {}: OK", name)
            else:
                logger.error(
                    "submodule {}: NOT READY  hint: git submodule update --init --recursive",
                    name,
                )

        # target connectivity
        if target_reachable:
            logger.info("target reachable  model={}", model_name)
        else:
            logger.error(
                "target NOT reachable  model={}  url={}",
                model_name,
                self.settings.target.inference_base_url,
            )

        # llama.cpp detail (only present when a llama-cpp config is active)
        if llama_cpp:
            if llama_cpp.get("healthy"):
                logger.info("llama.cpp: healthy")
            else:
                logger.error("llama.cpp: /health failed")
            if llama_cpp.get("model_found"):
                modalities = ", ".join(llama_cpp.get("modalities") or []) or "none"
                slot_part = (
                    f"  slots={llama_cpp['total_slots']}" if "total_slots" in llama_cpp else ""
                )
                if llama_cpp.get("model_status") == "loaded":
                    logger.info(
                        "model found  status={}  modalities=[{}]{}",
                        "loaded",
                        modalities,
                        slot_part,
                    )
                else:
                    logger.warning(
                        "model found  status={}, please load model by command: "
                        + "jishubench target load --model {}",
                        llama_cpp.get("model_status", "?"),
                        model_name,
                    )
            else:
                logger.error("model NOT found in Target catalog: {}", model_name)
            if llama_cpp.get("props_ok"):
                logger.info("llama.cpp: props OK")

        # monitor (non-blocking)
        if monitor_health is not None:
            caps = monitor_health.get("capabilities", {})
            avail = [key for key, available in caps.items() if available]
            logger.info(
                "monitor: {}  profiler={}  hz={}  capabilities=[{}]",
                "reachable",
                monitor_health.get("profiler", "?"),
                monitor_health.get("sample_hz", "?"),
                ", ".join(avail) if avail else "none",
            )

            device_info = monitor_health.get("device_info") or {}
            device_log = ", ".join(
                f"{key}={value}" for key, value in device_info.items() if value is not None
            )
            if device_log:
                logger.info("target device: {}", device_log)

            for error in monitor_health.get("errors") or []:
                logger.warning("monitor error: {}", error)
        else:
            logger.warning("monitor: NOT reachable (non-blocking)")

        # verdict
        if ready:
            logger.info("preflight PASSED — ready to run")
        else:
            logger.error("preflight FAILED — not ready to run")

        result: dict[str, Any] = {
            "submodules": submodules,
            "target_reachable": target_reachable,
            "target_model": model_name,
            "monitor_reachable": monitor_health is not None,
            "ready": ready,
        }
        if llama_cpp:
            result["llama_cpp"] = llama_cpp
        if warnings:
            result["warnings"] = warnings
        if adapter_ready is not None:
            result["adapter_ready"] = adapter_ready

        return result

    def run(
        self,
        *,
        model: str,
        benchmark: str,
        engine: str,
        adapter_cls: type[Runner],
        build_args: Callable[[str, str], Any],
    ) -> RunResult:
        """Run the adapter lifecycle and archive its outputs."""
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_dir_name = f"{model}__{run_id}"
        run_dir = self.settings.run_dir(benchmark, model_dir_name)

        # Per-run log file
        setup_run_logging(run_dir)

        # 1. provision
        server_snapshot = self.prepare_for_run(
            model,
            benchmark=benchmark,
            run_id=run_id,
        )

        # 2. run adapter with output_dir
        result: RunResult | None = None
        cleanup_snapshot: dict[str, Any] = {}
        adapter_error: Exception | None = None
        try:
            runner = adapter_cls(self.settings)
            result = runner.run(build_args(model, run_id), output_dir=run_dir)
        except Exception as exc:
            adapter_error = exc
            logger.error("adapter run failed: {}", exc)
        finally:
            # 3. cleanup (即使 adapter 抛出异常也要执行)
            try:
                cleanup_snapshot = self.cleanup_after_run(model)
            except Exception as exc:
                logger.warning("cleanup_after_run failed: {}", exc)

        if adapter_error is not None:
            error_extra: dict[str, Any] = {
                "server_snapshot": server_snapshot | cleanup_snapshot,
                "error": str(adapter_error),
            }
            self.archive_run_manifest(
                run_id=run_id,
                engine=engine,
                benchmark=benchmark,
                work_dir=run_dir,
                success=False,
                extra=error_extra,
                target_model=server_snapshot.get("resolved_model_id") or model,
            )
            raise RuntimeError(f"adapter {adapter_cls.__name__} failed") from adapter_error

        if result is None:
            raise RuntimeError(f"adapter {adapter_cls.__name__} returned no result")

        if result.run_id != run_id:
            raise RuntimeError(
                f"adapter {adapter_cls.__name__} returned mismatched run_id "
                f"{result.run_id!r}; expected {run_id!r}"
            )

        # 4. archive manifest
        extra: dict[str, Any] = {
            "server_snapshot": server_snapshot | cleanup_snapshot,
        }
        extra[engine] = result.summary
        self.archive_run_manifest(
            run_id=run_id,
            engine=engine,
            benchmark=benchmark,
            work_dir=run_dir,
            success=result.success,
            extra=extra,
            target_model=server_snapshot.get("resolved_model_id") or model,
        )

        # 5. write summary.json inside the run directory
        summary: dict[str, Any] = {
            "run_id": run_id,
            "success": result.success,
            "metric": result.metric,
            "value": result.value,
            "engine": engine,
            "model": result.model,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 6. print result to stdout
        output: dict[str, Any] = {
            "run_id": run_id,
            "success": result.success,
            "engine": engine,
            "metric": result.metric,
            "value": result.value,
        }
        if isinstance(result.value, dict):
            for task_name, entry in result.value.items():
                if isinstance(entry, dict):
                    task_metric = entry.get("metric", "?")
                    score = float(entry.get("value", 0.0))
                else:
                    task_metric = result.metric or "?"
                    score = float(entry)
                logger.info("  {:<30s} {}={:>6.2f}%", task_name, task_metric, score * 100)
        elif result.metric is not None and result.value is not None:
            logger.info("  {}={:.2f}%", result.metric, float(result.value) * 100)
        print(
            json.dumps(output, indent=2, ensure_ascii=False),
        )
        return result

    def prepare_for_run(
        self,
        model_name: str | None = None,
        *,
        benchmark: str = "",
        run_id: str = "",
        enable_trace: bool | None = None,
    ) -> dict[str, Any]:
        """Ensure model is loaded and optionally start a run-level monitor trace.

        Returns a server_snapshot dict to be stored in the run manifest.

        Raises RuntimeError when the model is missing on Target or load fails.
        """
        model_name = model_name or self.settings.target.model
        if not self._llama_cfg:
            resolved_id = model_name
        else:
            resolved_id = self._llama.resolve_model_id(model_name)
            self._llama.require_model_in_catalog(model_name)

        # Handle model switch: unload previous model if configured
        if self._llama_cfg and self._llama_cfg.lifecycle.unload_on_model_switch:
            try:
                loaded = [
                    m
                    for m in self._llama.control.list_models()
                    if m.status == "loaded" and m.id != resolved_id
                ]
            except Exception as exc:
                logger.warning("unable to list models for unload-on-switch: {}", exc)
                loaded = []
            for m in loaded:
                try:
                    self._llama.control.unload_model(m.id)
                except Exception as exc:
                    logger.warning("unload-on-switch failed for {}: {}", m.id, exc)

        # Ensure target model is loaded and verified
        if self._llama_cfg and self._llama_cfg.lifecycle.ensure_loaded:
            logger.info("ensuring model loaded: {}", resolved_id)
            loaded_id = self._llama.ensure_model_loaded(model_name)
            if loaded_id != resolved_id:
                raise RuntimeError(
                    f"failed to load model {model_name!r} "
                    f"(router id {resolved_id!r}): "
                    f"router loaded unexpected model {loaded_id!r}"
                    f" (status={self._llama.control.get_model_status(resolved_id)})"
                )

        # Take server snapshot (props + slots)
        snapshot = self._llama.readiness_snapshot(model_name)
        snapshot["resolved_model_id"] = resolved_id
        if (
            snapshot.get("model_status") != "loaded"
            and self._llama_cfg
            and self._llama_cfg.lifecycle.ensure_loaded
        ):
            _status = snapshot.get("model_status")
            _msg = (
                f"failed to load model {model_name!r} "
                f"(router id {resolved_id!r}): "
                f"model is not loaded after prepare_for_run"
            )
            if _status:
                _msg = f"{_msg} (status={_status})"
            raise RuntimeError(_msg)

        # Start run-level trace if configured
        should_trace = (
            enable_trace if enable_trace is not None else self.settings.eval.enable_traces
        )
        if should_trace:
            trace_id = f"run_{run_id}"
            self._run_trace_id = trace_id
            record = TraceRecord(
                trace_id=trace_id,
                benchmark=benchmark,
                sample_id="",
                model=model_name,
                phase="run",
            )
            try:
                self.monitor.create_trace(record)
                self.monitor.start_trace(trace_id)
                self.monitor.mark(trace_id, "run_start")
                logger.info("started run trace {}", trace_id)
            except Exception as exc:
                logger.warning("failed to start run trace {}: {}", trace_id, exc)
                self._run_trace_id = None

        return snapshot

    def cleanup_after_run(
        self,
        model_name: str | None = None,
        *,
        enable_trace: bool | None = None,
    ) -> dict[str, Any]:
        """Stop trace (if active) and optionally unload model.

        Returns final server snapshot for the run manifest.
        """
        cfg = self._llama_cfg
        model_name = model_name or self.settings.target.model

        # Stop run-level trace
        should_trace = (
            enable_trace if enable_trace is not None else self.settings.eval.enable_traces
        )
        trace_metrics: dict[str, Any] = {}
        if should_trace and self._run_trace_id:
            try:
                self.monitor.mark(self._run_trace_id, "run_end")
                self.monitor.stop_trace(self._run_trace_id)
                trace_metrics = self.monitor.get_trace(self._run_trace_id)
                logger.info("stopped run trace {}", self._run_trace_id)
            except Exception as exc:
                logger.warning("failed to stop run trace {}: {}", self._run_trace_id, exc)
            self._run_trace_id = None

        # Optionally unload model
        if cfg and cfg.lifecycle.unload_after_run:
            logger.info("unloading model after run: {}", model_name)
            self._llama.unload_model(model_name)

        snapshot = self._llama.readiness_snapshot(model_name)
        snapshot["trace_metrics"] = trace_metrics
        return snapshot

    def archive_run_manifest(
        self,
        run_id: str,
        engine: str,
        benchmark: str,
        work_dir: Path,
        *,
        success: bool,
        extra: dict[str, Any] | None = None,
        target_model: str | None = None,
    ) -> Path:
        manifest = {
            "run_id": run_id,
            "engine": engine,
            "benchmark": benchmark,
            "success": success,
            "settings": {
                "target_model": target_model or self.settings.target.model,
                "target_url": self.settings.target.inference_base_url,
            },
            "device_info": self.monitor.device_info or {},
            "extra": extra or {},
        }
        work_dir.mkdir(parents=True, exist_ok=True)
        path = work_dir / "manifest.json"
        manifest_adapter: TypeAdapter[Any] = TypeAdapter(Any)
        jsonable_manifest = manifest_adapter.dump_python(
            manifest,
            mode="json",
            fallback=lambda value: getattr(value, "__qualname__", repr(value))
            if callable(value)
            else str(value),
        )
        path.write_text(
            json.dumps(jsonable_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("run manifest saved to {} (success={})", path, success)
        return path
