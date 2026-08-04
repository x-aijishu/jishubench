"""Terminal-Bench v0.1.1 runner — external agent harness on Host, inference on Target.

This adapter drives the legacy ``tb`` harness (Terminal-Bench v0.1.1) via its
Python API. Terminal-Bench 2.0 (Harbor) is a separate, fully-decoupled adapter
in ``terminal_bench_2.py`` — see ``docs/configuration/terminal-bench-2.md``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from loguru import logger

from host.adapters import DownloadReport, RunResult
from host.config import Settings, load_settings, terminal_bench_env
from host.config.settings import TerminalBenchConfig

from .utils import check_docker_available, check_harness_cli


@dataclass
class TerminalBenchRunArgs:
    model_name: str | None = None
    limit: int | None = None
    task_ids: Sequence[str] | None = None
    run_id: str | None = None


@dataclass
class TerminalBenchTaskResult:
    task_id: str
    is_resolved: bool | None
    failure_mode: str | None = None


class TerminalBenchRunner:
    """Host-side Terminal-Bench v0.1.1 harness pointed at Target OpenAI-compatible inference."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

    def preflight(self) -> bool:
        cfg = self.settings.terminal_bench
        host_arch = platform.machine().lower()

        logger.info("Terminal-Bench preflight  agent={}  arch={}", cfg.agent, host_arch)

        n_errors = 0

        if check_docker_available():
            logger.info("  docker: OK")
        else:
            logger.error("  docker: not reachable (run `docker info`)")
            n_errors += 1

        cli = check_harness_cli("tb")
        if cli:
            logger.info("  tb CLI: {}", cli)
        else:
            logger.error("  tb CLI: not found (uv sync --extra terminal)")
            n_errors += 1

        if check_terminal_bench_importable():
            logger.info("  terminal_bench package: importable")
        else:
            logger.warning("  terminal_bench package: not importable in this environment")

        if host_arch in {"aarch64", "arm64"}:
            logger.error(
                "  ARM64: unsupported for now; TB task Docker images are linux/amd64 only now"
            )
            n_errors += 1

        if n_errors:
            logger.error("Terminal-Bench preflight: not ready ({} error(s))", n_errors)
        else:
            logger.info("Terminal-Bench preflight: ready")

        return n_errors == 0

    def run(self, args: TerminalBenchRunArgs, *, output_dir: Path | None = None) -> RunResult:
        if output_dir is not None and args.run_id is None:
            raise ValueError("run_id is required when output_dir is externally managed")

        host_arch = platform.machine().lower()
        if host_arch in {"aarch64", "arm64"}:
            raise RuntimeError(
                f"Terminal-Bench requires an x86-64 host; detected architecture: {host_arch!r}. "
                "TB task Docker images are linux/amd64 only and are not supported under "
                "QEMU emulation on ARM64 hosts."
            )

        cfg = self.settings.terminal_bench
        runs_dir = output_dir or self.settings.terminal_bench_runs_dir()
        run_id = args.run_id or _default_run_id()
        model_litellm = self.settings.terminal_bench_litellm_model(args.model_name)
        agent_kwargs = self.settings.merged_terminal_bench_agent_kwargs(args.model_name)

        logger.info(
            "Terminal-Bench agent={} dataset={}",
            cfg.agent,
            cfg.dataset,
        )
        logger.info("  model: {}  api_base: {}", model_litellm, agent_kwargs.get("api_base"))
        logger.info("  runs_dir: {}  run_id: {}", runs_dir, run_id)
        if args.limit is not None:
            logger.info("  limit: {}", args.limit)

        result = self._run_tb_python(
            runs_dir=runs_dir,
            run_id=run_id,
            model_litellm=model_litellm,
            agent_kwargs=agent_kwargs,
            args=args,
        )

        result.summary["model"] = self.settings.resolve_terminal_bench_model_id(args.model_name)
        result.summary["agent"] = cfg.agent
        result.summary["dataset"] = cfg.dataset
        result.summary["harness"] = "tb"
        result.model = self.settings.resolve_terminal_bench_model_id(args.model_name)
        self._update_cache(result)
        return result

    def _run_tb_python(
        self,
        *,
        runs_dir: Path,
        run_id: str,
        model_litellm: str,
        agent_kwargs: dict[str, object],
        args: TerminalBenchRunArgs,
    ) -> RunResult:
        """Run Terminal-Bench using the Python API directly (no subprocess)."""
        from terminal_bench.agents.agent_name import AgentName
        from terminal_bench.harness.harness import Harness

        cfg = self.settings.terminal_bench
        dataset_name, dataset_version = _parse_dataset_spec(cfg.dataset)

        harness_agent_kwargs = dict(agent_kwargs)

        try:
            agent_name_enum: AgentName | None = AgentName(cfg.agent)
            agent_import_path: str | None = None
        except ValueError:
            agent_name_enum = None
            agent_import_path = cfg.agent

        logger.debug("  dataset: {}=={}", dataset_name, dataset_version)

        with terminal_bench_env(self.settings):
            harness = Harness(
                output_path=runs_dir,
                run_id=run_id,
                agent_name=agent_name_enum,
                agent_import_path=agent_import_path,
                model_name=model_litellm,
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                task_ids=list(args.task_ids) if args.task_ids else None,
                n_tasks=args.limit,
                n_concurrent_trials=cfg.n_concurrent,
                n_attempts=cfg.n_attempts,
                agent_kwargs=harness_agent_kwargs,
                log_level=20,  # logging.INFO
                **self._harness_timeout_kwargs(cfg),
            )
            harness.run()

        return self._parse_tb_results(runs_dir, run_id)

    @staticmethod
    def _harness_timeout_kwargs(cfg: TerminalBenchConfig) -> dict[str, float | None]:
        return {
            "global_timeout_multiplier": cfg.global_timeout_multiplier,
            "global_agent_timeout_sec": cfg.global_agent_timeout_sec,
            "global_test_timeout_sec": cfg.global_test_timeout_sec,
        }

    def _parse_tb_results(self, runs_dir: Path, run_id: str) -> RunResult:
        run_path = runs_dir / run_id
        results_path = run_path / "results.json"
        raw_tasks: list[TerminalBenchTaskResult] = []

        if results_path.is_file():
            payload = json.loads(results_path.read_text(encoding="utf-8"))
            for item in payload.get("results", []):
                raw_tasks.append(
                    TerminalBenchTaskResult(
                        task_id=item.get("task_id", "unknown"),
                        is_resolved=item.get("is_resolved"),
                        failure_mode=_stringify(item.get("failure_mode")),
                    )
                )
        else:
            raw_tasks = self._collect_per_task_results(run_path)

        tasks = _dedupe_task_results(raw_tasks)
        n_resolved = sum(1 for t in tasks if t.is_resolved)
        n_total = len(tasks)
        success = n_total > 0
        value = n_resolved / n_total if n_total else None
        parse_error = None if success else "no terminal-bench task results found"

        resolved = [t.task_id for t in tasks if t.is_resolved]
        unresolved = [t.task_id for t in tasks if not t.is_resolved]
        return RunResult(
            success=success,
            run_id=run_id,
            model="",
            metric="accuracy" if value is not None else None,
            value=value,
            n_total=n_total,
            n_passed=n_resolved,
            summary={
                "harness": "tb",
                "agent": "",
                "dataset": "",
                "n_resolved": n_resolved,
                "resolved_ids": resolved,
                "unresolved_ids": unresolved,
                "tasks": [
                    {
                        "task_id": t.task_id,
                        "is_resolved": t.is_resolved,
                        "failure_mode": t.failure_mode,
                    }
                    for t in tasks
                ],
                "output_dir": str(run_path),
                "results_path": str(results_path) if results_path.is_file() else None,
                "parse_error": parse_error,
            },
            raw=None,
        )

    @staticmethod
    def _collect_per_task_results(run_path: Path) -> list[TerminalBenchTaskResult]:
        tasks: list[TerminalBenchTaskResult] = []
        if not run_path.is_dir():
            return tasks
        for child in sorted(run_path.iterdir()):
            if not child.is_dir():
                continue
            per_task = child / "results.json"
            if not per_task.is_file():
                continue
            try:
                payload = json.loads(per_task.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            tasks.append(
                TerminalBenchTaskResult(
                    task_id=payload.get("task_id", child.name),
                    is_resolved=payload.get("is_resolved"),
                    failure_mode=_stringify(payload.get("failure_mode")),
                )
            )
        return tasks

    def _update_cache(self, result: RunResult) -> None:
        cache_path = self.settings.terminal_bench_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tasks = result.summary.get("tasks", [])
        with cache_path.open("a", encoding="utf-8") as fh:
            for task in tasks:
                if task.get("is_resolved") is None:
                    continue
                record = {
                    "run_id": result.run_id,
                    "task_id": task["task_id"],
                    "is_resolved": task["is_resolved"],
                    "harness": result.summary.get("harness", ""),
                    "dataset": result.summary.get("dataset", ""),
                    "model": result.model,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def list_completed_task_ids(self, run_id: str) -> set[str]:
        runs_dir = self.settings.terminal_bench_runs_dir()
        run_path = runs_dir / run_id
        completed: set[str] = set()
        for task in self._collect_per_task_results(run_path):
            if task.is_resolved is not None:
                completed.add(task.task_id)
        return completed

    @classmethod
    def download_tasks(
        cls,
        items: list[str],
        settings: Settings,
        *,
        check_only: bool = False,
        force: bool = False,
    ) -> DownloadReport:
        """Download or verify datasets for Terminal-Bench v0.1.1."""
        report = DownloadReport(backend="terminal-bench")

        for item in items:
            try:
                if not check_docker_available():
                    raise RuntimeError("Docker daemon is not reachable")

                cli = check_harness_cli("tb")
                if cli is None:
                    raise RuntimeError("tb CLI not found on PATH")

                dataset = item if "==" in item or "@" in item else settings.terminal_bench.dataset

                if check_only:
                    _verify_local_dataset(settings, dataset)
                else:
                    _download_dataset(settings, dataset, force=force)

                report.succeeded.append(item)
                verb = "verified" if check_only else "downloaded"
                print(f"{verb}: {item}")
            except Exception as exc:
                verb = "verify" if check_only else "download"
                report.failed[item] = str(exc)
                print(f"failed to {verb} {item}: {exc}", file=sys.stderr)

        return report


def check_terminal_bench_importable() -> bool:
    try:
        import terminal_bench  # noqa: F401

        return True
    except ImportError:
        return False


def _dedupe_task_results(
    raw: list[TerminalBenchTaskResult],
) -> list[TerminalBenchTaskResult]:
    """Collapse multiple attempts of the same task into one entry.

    A task counts as resolved if any attempt resolved it (pass@k). Order of
    first appearance is preserved.
    """
    merged: dict[str, TerminalBenchTaskResult] = {}
    for task in raw:
        existing = merged.get(task.task_id)
        if existing is None:
            merged[task.task_id] = task
            continue
        resolved = bool(existing.is_resolved) or bool(task.is_resolved)
        merged[task.task_id] = TerminalBenchTaskResult(
            task_id=task.task_id,
            is_resolved=resolved,
            failure_mode=None if resolved else (existing.failure_mode or task.failure_mode),
        )
    return list(merged.values())


def _default_run_id() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S")


def _parse_dataset_spec(dataset: str) -> tuple[str, str | None]:
    """Parse 'name==version' or 'name@version' or bare 'name' into (name, version)."""
    if "==" in dataset:
        name, version = dataset.split("==", 1)
        return name.strip(), version.strip()
    if "@" in dataset:
        name, version = dataset.split("@", 1)
        return name.strip(), version.strip()
    return dataset.strip(), None


def _stringify(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# Dataset download helpers
# ---------------------------------------------------------------------------


def _download_dataset(settings: Settings, dataset: str, *, force: bool) -> None:
    cmd = ["tb", "datasets", "download", "--dataset", dataset]
    if force:
        cmd.append("--overwrite")
    with terminal_bench_env(settings):
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        raise RuntimeError(f"CLI exit {proc.returncode}: {tail}")


def _verify_local_dataset(settings: Settings, dataset: str) -> None:
    cache_root = Path.home() / ".cache" / "terminal-bench"
    if "==" in dataset:
        name, version = dataset.split("==", 1)
        candidate = cache_root / name.strip() / version.strip()
        if candidate.is_dir():
            return
    elif dataset and cache_root.is_dir():
        if any(cache_root.iterdir()):
            return
    raise FileNotFoundError(
        f"dataset cache not found under {cache_root} for {dataset!r}; run download without --check"
    )
