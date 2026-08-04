"""Terminal-Bench 2.0 (Harbor) runner — external agent harness on Host, inference on Target.

Harbor is the official harness for Terminal-Bench 2.0 (TB 2.0+), a ground-up rewrite
of the legacy ``tb`` harness (v0.1.1). TB 2.0 is fully decoupled from TB v0.1.1:

- Different task format: ``task.toml`` + ``instruction.md`` + ``environment/`` (vs
  ``task.yaml`` + ``solution.sh`` + ``run-tests.sh``).
- Different reward scheme: tasks emit ``logs/verifier/reward.txt`` (vs a TB-internal
  parser mapping test stdout to a binary reward).
- Different CLI: ``harbor run --dataset terminal-bench@2.0`` (vs ``tb`` Python API).
- Multi-file solutions (``/solution/``) and modular test deps (``/tests/``).

This adapter is self-contained and mirrors the structure of the other adapters
(``tau_bench.py``, ``claw_eval.py``): own ``RunArgs``/``Runner``/``preflight``/
``download_tasks``, own config section (``terminal_bench_2``), own registered
benchmark name (``terminal-bench-2``). It does not share code with
``terminal_bench.py`` (TB v0.1.1).

Architecture:
- Agent (被测模型): Target llama-server via OpenAI-compatible endpoint, reached by
  Harbor's litellm layer using ``openai/<model>`` and ``api_base=target.inference_base_url``.
- Harness: ``harbor`` CLI subprocess on the Host, Docker containers per task.
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
from host.config import Settings, load_settings, terminal_bench_2_env

from .utils import check_docker_available, check_harness_cli


@dataclass
class TerminalBench2RunArgs:
    model_name: str | None = None
    limit: int | None = None
    task_ids: Sequence[str] | None = None
    run_id: str | None = None


@dataclass
class TerminalBench2TaskResult:
    task_id: str
    is_resolved: bool | None
    failure_mode: str | None = None


class TerminalBench2Runner:
    """Host-side Terminal-Bench 2.0 (Harbor) harness pointed at Target inference."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

    def preflight(self) -> bool:
        cfg = self.settings.terminal_bench_2
        host_arch = platform.machine().lower()

        logger.info(
            "Terminal-Bench 2.0 (Harbor) preflight  agent={}  arch={}",
            cfg.agent,
            host_arch,
        )

        n_errors = 0

        if check_docker_available():
            logger.info("  docker: OK")
        else:
            logger.error("  docker: not reachable (run `docker info`)")
            n_errors += 1

        cli = check_harness_cli("harbor")
        if cli:
            logger.info("  harbor CLI: {}", cli)
        else:
            logger.error(
                "  harbor CLI: not found (uv sync --extra harbor or uv tool install harbor)"
            )
            n_errors += 1

        if check_harbor_importable():
            logger.info("  harbor package: importable")
        else:
            logger.warning("  harbor package: not importable in this environment")

        if host_arch in {"aarch64", "arm64"}:
            logger.error(
                "  ARM64: unsupported for now; TB 2.0 task Docker images are linux/amd64 only"
            )
            n_errors += 1

        if n_errors:
            logger.error("Terminal-Bench 2.0 preflight: not ready ({} error(s))", n_errors)
        else:
            logger.info("Terminal-Bench 2.0 preflight: ready")

        return n_errors == 0

    def run(self, args: TerminalBench2RunArgs, *, output_dir: Path | None = None) -> RunResult:
        if output_dir is not None and args.run_id is None:
            raise ValueError("run_id is required when output_dir is externally managed")

        host_arch = platform.machine().lower()
        if host_arch in {"aarch64", "arm64"}:
            raise RuntimeError(
                f"Terminal-Bench 2.0 requires an x86-64 host; detected architecture: "
                f"{host_arch!r}. TB 2.0 task Docker images are linux/amd64 only and "
                "are not supported under QEMU emulation on ARM64 hosts."
            )

        cfg = self.settings.terminal_bench_2
        runs_dir = output_dir or self.settings.terminal_bench_2_runs_dir()
        run_id = args.run_id or _default_run_id()
        model_litellm = self.settings.terminal_bench_2_litellm_model(args.model_name)
        agent_kwargs = self.settings.merged_terminal_bench_2_agent_kwargs(args.model_name)

        logger.info(
            "Terminal-Bench 2.0 (Harbor) agent={} dataset={}",
            cfg.agent,
            cfg.dataset,
        )
        logger.info("  model: {}  api_base: {}", model_litellm, agent_kwargs.get("api_base"))
        logger.info("  runs_dir: {}  run_id: {}", runs_dir, run_id)
        if args.limit is not None:
            logger.info("  limit: {}", args.limit)

        cmd = self._build_harbor_command(
            output_dir=runs_dir,
            run_id=run_id,
            model=model_litellm,
            agent_kwargs=agent_kwargs,
            limit=args.limit,
            task_ids=args.task_ids,
        )
        with terminal_bench_2_env(self.settings):
            proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Harbor harness exited with code {proc.returncode} (cmd: {' '.join(cmd)})"
            )

        result = self._parse_harbor_results(runs_dir, run_id)
        result.summary["model"] = self.settings.resolve_terminal_bench_2_model_id(args.model_name)
        result.summary["agent"] = cfg.agent
        result.summary["dataset"] = cfg.dataset
        result.summary["harness"] = "harbor"
        result.model = self.settings.resolve_terminal_bench_2_model_id(args.model_name)
        self._update_cache(result)
        return result

    def _build_harbor_command(
        self,
        *,
        output_dir: Path,
        run_id: str,
        model: str,
        agent_kwargs: dict[str, object],
        limit: int | None,
        task_ids: Sequence[str] | None,
    ) -> list[str]:
        cfg = self.settings.terminal_bench_2
        cmd: list[str] = [
            "harbor",
            "run",
            "--dataset",
            cfg.dataset,
            "--agent",
            cfg.agent,
            "--model",
            model,
            "--jobs-dir",
            str(output_dir),
            "--job-name",
            run_id,
            "--n-concurrent",
            str(cfg.n_concurrent),
            "--n-attempts",
            str(cfg.n_attempts),
        ]
        if cfg.global_timeout_multiplier != 1.0:
            cmd.extend(["--timeout-multiplier", str(cfg.global_timeout_multiplier)])
        if limit is not None:
            cmd.extend(["--n-tasks", str(limit)])
        if task_ids:
            for task_id in task_ids:
                cmd.extend(["--include-task-name", task_id])
        cmd.extend(self._agent_kwarg_flags(agent_kwargs))
        return cmd

    @staticmethod
    def _agent_kwarg_flags(agent_kwargs: dict[str, object]) -> list[str]:
        flags: list[str] = []
        for key, value in agent_kwargs.items():
            if key == "model_name":
                continue
            if isinstance(value, dict):
                encoded = json.dumps(value)
            else:
                encoded = str(value)
            flags.extend(["--agent-kwarg", f"{key}={encoded}"])
        return flags

    def _parse_harbor_results(self, runs_dir: Path, run_id: str) -> RunResult:
        run_path = runs_dir / run_id
        tasks: list[TerminalBench2TaskResult] = []

        for candidate in sorted(run_path.rglob("result.json")):
            if candidate.parent == run_path:
                continue
            task = _parse_harbor_trial_payload(candidate)
            if task is not None:
                tasks.append(task)

        if not tasks:
            for candidate in sorted(run_path.rglob("results.json")):
                if candidate.parent == run_path:
                    continue
                task = _parse_harbor_trial_payload(candidate)
                if task is not None:
                    tasks.append(task)

        tasks = _dedupe_task_results(tasks)
        n_resolved = sum(1 for t in tasks if t.is_resolved)
        n_total = len(tasks)
        success = n_total > 0
        value = n_resolved / n_total if n_total else None
        parse_error = None if success else "no harbor terminal-bench task results found"
        resolved = [t.task_id for t in tasks if t.is_resolved]
        unresolved = [t.task_id for t in tasks if not t.is_resolved]
        summary_path = run_path / "results.json"
        return RunResult(
            success=success,
            run_id=run_id,
            model="",
            metric="accuracy" if value is not None else None,
            value=value,
            n_total=n_total,
            n_passed=n_resolved,
            summary={
                "harness": "harbor",
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
                "results_path": str(summary_path) if summary_path.is_file() else None,
                "parse_error": parse_error,
            },
            raw=None,
        )

    def _update_cache(self, result: RunResult) -> None:
        cache_path = self.settings.terminal_bench_2_cache_path()
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

    @classmethod
    def download_tasks(
        cls,
        items: list[str],
        settings: Settings,
        *,
        check_only: bool = False,
        force: bool = False,
    ) -> DownloadReport:
        """Download or verify datasets for Terminal-Bench 2.0 (Harbor)."""
        report = DownloadReport(backend="terminal-bench-2")

        for item in items:
            try:
                if not check_docker_available():
                    raise RuntimeError("Docker daemon is not reachable")

                cli = check_harness_cli("harbor")
                if cli is None:
                    raise RuntimeError("harbor CLI not found on PATH")

                dataset = item if "==" in item or "@" in item else settings.terminal_bench_2.dataset

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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def check_harbor_importable() -> bool:
    try:
        import harbor  # noqa: F401

        return True
    except ImportError:
        return False


def _parse_harbor_trial_payload(path: Path) -> TerminalBench2TaskResult | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return TerminalBench2TaskResult(
        task_id=_extract_harbor_task_id(payload, path.parent.name),
        is_resolved=_extract_harbor_is_resolved(payload),
        failure_mode=_extract_harbor_failure_mode(payload),
    )


def _extract_harbor_task_id(payload: dict[str, object], trial_dir_name: str) -> str:
    task_name = payload.get("task_name")
    if isinstance(task_name, str) and task_name:
        return task_name

    task_id = payload.get("task_id")
    if isinstance(task_id, dict):
        path = task_id.get("path")
        if isinstance(path, str) and path:
            return path
    if isinstance(task_id, str) and task_id:
        return task_id

    if "__" in trial_dir_name:
        return trial_dir_name.split("__", 1)[0]
    return trial_dir_name


def _extract_harbor_is_resolved(payload: dict[str, object]) -> bool | None:
    resolved = payload.get("is_resolved")
    if resolved is not None:
        return bool(resolved)

    success = payload.get("success")
    if success is not None:
        return bool(success)

    verifier = payload.get("verifier_result")
    if isinstance(verifier, dict):
        rewards = verifier.get("rewards")
        if isinstance(rewards, dict):
            reward = rewards.get("reward")
            if reward is not None:
                try:
                    return float(reward) >= 1.0
                except (TypeError, ValueError):
                    return None

    if payload.get("exception_info"):
        return False
    return None


def _extract_harbor_failure_mode(payload: dict[str, object]) -> str | None:
    failure_mode = _stringify(payload.get("failure_mode"))
    if failure_mode:
        return failure_mode
    exception_info = payload.get("exception_info")
    if exception_info is None:
        return None
    if isinstance(exception_info, dict):
        for key in ("type", "message", "exception_type"):
            value = exception_info.get(key)
            if isinstance(value, str) and value:
                return value
    return _stringify(exception_info)


def _dedupe_task_results(
    raw: list[TerminalBench2TaskResult],
) -> list[TerminalBench2TaskResult]:
    """Collapse multiple attempts of the same task into one entry.

    A task counts as resolved if any attempt resolved it (pass@k). Order of
    first appearance is preserved.
    """
    merged: dict[str, TerminalBench2TaskResult] = {}
    for task in raw:
        existing = merged.get(task.task_id)
        if existing is None:
            merged[task.task_id] = task
            continue
        resolved = bool(existing.is_resolved) or bool(task.is_resolved)
        merged[task.task_id] = TerminalBench2TaskResult(
            task_id=task.task_id,
            is_resolved=resolved,
            failure_mode=None if resolved else (existing.failure_mode or task.failure_mode),
        )
    return list(merged.values())


def _default_run_id() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S")


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
    cmd = ["harbor", "datasets", "download", dataset, "--cache"]
    if force:
        cmd.append("--overwrite")
    with terminal_bench_2_env(settings):
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        raise RuntimeError(f"CLI exit {proc.returncode}: {tail}")


def _verify_local_dataset(settings: Settings, dataset: str) -> None:
    cache_root = Path.home() / ".cache" / "harbor" / "tasks" / "packages"
    if "==" in dataset or "@" in dataset:
        name, _version = dataset.replace("==", "@").split("@", 1)
        candidate = cache_root / name.strip()
    else:
        candidate = cache_root / dataset.strip()
    if candidate.is_dir() and any(candidate.iterdir()):
        return
    raise FileNotFoundError(
        f"dataset cache not found under {cache_root} for {dataset!r}; run download without --check"
    )
