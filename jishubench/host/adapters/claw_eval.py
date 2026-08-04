"""Claw-Eval runner — external Host sandbox harness pointed at Target inference."""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from importlib import import_module, util
from pathlib import Path
from typing import Any, Sequence, TextIO, cast

import yaml
from loguru import logger

from host.adapters import DownloadReport, RunResult
from host.adapters.utils import check_docker_available
from host.config import REPO_ROOT, Settings, claw_eval_env, load_settings
from host.config.submodules import CLAW_EVAL_DIR, ensure_claw_eval


@dataclass
class ClawEvalRunArgs:
    model_name: str | None = None
    limit: int | None = None
    task_ids: Sequence[str] | None = None
    run_id: str | None = None
    no_judge: bool | None = None
    filter: str | None = None
    tag: str | None = None
    range: str | None = None
    language: str | None = None
    category: str | None = None


@dataclass(frozen=True)
class _ClawEvalRunOptions:
    config_template: Path
    no_judge: bool
    text_only: bool
    filter: str | None
    tag: str | None
    range: str | None
    language: str | None
    category: str | None


class ClawEvalRunner:
    """Run Claw-Eval batch while keeping artifacts inside jishubench run dirs."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

    def preflight(self) -> bool:
        cfg = self.settings.claw_eval
        n_errors = 0

        try:
            ensure_claw_eval()
            logger.info("  claw-eval submodule: OK")
        except FileNotFoundError as exc:
            logger.error("  claw-eval submodule: {}", exc)
            n_errors += 1

        if util.find_spec("claw_eval") is not None:
            logger.info("  claw_eval package: importable")
        else:
            logger.error("  claw_eval package: not importable  hint: uv sync --extra claw-eval")
            n_errors += 1

        fixture_error = self._check_local_assets()
        if fixture_error is None:
            logger.info("  claw-eval tasks/fixtures: OK")
        else:
            logger.error(
                "  claw-eval tasks/fixtures: {}  hint: uv run jishubench download "
                "--backend claw-eval claw-eval",
                fixture_error,
            )
            n_errors += 1

        if cfg.sandbox:
            if check_docker_available():
                logger.info("  docker: OK")
            else:
                logger.error("  docker: not reachable (run `docker info`)")
                n_errors += 1
            if _docker_image_exists(cfg.sandbox_image):
                logger.info("  sandbox image: {}", cfg.sandbox_image)
            else:
                logger.error(
                    "  sandbox image: missing {}  hint: uv run claw-eval build-image "
                    "--config submodules/claw-eval/config_general.yaml",
                    cfg.sandbox_image,
                )
                n_errors += 1

        judge_key_ok = bool(
            os.environ.get("OPENROUTER_API_KEY") or self.settings.judge.openai_api_key
        )
        if cfg.no_judge or judge_key_ok:
            logger.info("  judge API key: {}", "disabled" if cfg.no_judge else "configured")
        else:
            logger.error(
                "  judge API key: not configured  hint: set OPENROUTER_API_KEY or judge key"
            )
            n_errors += 1

        ua_key_dedicated = bool(
            os.environ.get("CLAW_USER_API_KEY", "").strip()
            or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        )
        if ua_key_dedicated:
            logger.info("  user-agent API key: configured (dedicated)")
        elif self.settings.resolve_claw_eval_user_agent_api_key():
            logger.info("  user-agent API key: judge fallback")
        else:
            logger.warning(
                "  user-agent API key: not set  "
                "hint: set CLAW_USER_API_KEY (multi-turn C* tasks need a user-simulator LLM)"
            )

        if n_errors:
            logger.error("Claw-Eval preflight: not ready ({} error(s))", n_errors)
        else:
            logger.info("Claw-Eval preflight: ready")
        return n_errors == 0

    def run(self, args: ClawEvalRunArgs, *, output_dir: Path | None = None) -> RunResult:
        if output_dir is not None and args.run_id is None:
            raise ValueError("run_id is required when output_dir is externally managed")

        run_id = args.run_id or _default_run_id()
        run_dir = output_dir or self.settings.run_dir("claw-eval", run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        trace_root = run_dir / "traces"
        trace_root.mkdir(parents=True, exist_ok=True)

        model_name = self.settings.resolve_claw_eval_model_id(args.model_name)
        options = self._run_options(args)
        tasks_dir = self._resolve_tasks_dir()
        effective_tasks_dir = self._prepare_task_subset(
            tasks_dir=tasks_dir,
            run_dir=run_dir,
            limit=args.limit,
            task_ids=args.task_ids,
            options=options,
        )
        config_path = self._write_run_config(
            run_dir=run_dir,
            tasks_dir=effective_tasks_dir,
            trace_root=trace_root,
            model_name=model_name,
            options=options,
        )
        batch_args = self._build_batch_args(
            config_path=config_path,
            tasks_dir=effective_tasks_dir,
            trace_root=trace_root,
            model_name=model_name,
            options=options,
            subset_applied=effective_tasks_dir != tasks_dir,
        )

        stdout_path = run_dir / "claw_eval_stdout.log"
        stderr_path = run_dir / "claw_eval_stderr.log"

        logger.info("Claw-Eval run_id={} model={} output_dir={}", run_id, model_name, run_dir)
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_fh,
            stderr_path.open("w", encoding="utf-8") as stderr_fh,
        ):
            with claw_eval_env(self.settings):
                with (
                    redirect_stdout(cast(TextIO, _Tee(sys.stdout, stdout_fh))),
                    redirect_stderr(cast(TextIO, _Tee(sys.stderr, stderr_fh))),
                ):
                    _run_claw_eval_batch(batch_args)

        summary_path = _find_latest_artifact(trace_root, "batch_summary.json")
        results_path = _find_latest_artifact(trace_root, "batch_results.json")
        result = self._parse_results(
            run_id=run_id,
            model_name=model_name,
            trace_root=trace_root,
            summary_path=summary_path,
            results_path=results_path,
        )
        result.summary["stdout_log"] = str(stdout_path)
        result.summary["stderr_log"] = str(stderr_path)
        result.summary["config_path"] = str(config_path)
        return result

    def _run_options(self, args: ClawEvalRunArgs) -> _ClawEvalRunOptions:
        cfg = self.settings.claw_eval
        return _ClawEvalRunOptions(
            config_template=_resolve_config_template(cfg.config_template),
            no_judge=cfg.no_judge if args.no_judge is None else args.no_judge,
            text_only=cfg.text_only,
            filter=args.filter if args.filter is not None else cfg.filter,
            tag=args.tag if args.tag is not None else cfg.tag,
            range=args.range if args.range is not None else cfg.range,
            language=args.language if args.language is not None else cfg.language,
            category=args.category if args.category is not None else cfg.category,
        )

    def _build_batch_args(
        self,
        *,
        config_path: Path,
        tasks_dir: Path,
        trace_root: Path,
        model_name: str,
        options: _ClawEvalRunOptions,
        subset_applied: bool = False,
    ) -> Namespace:
        cfg = self.settings.claw_eval
        return Namespace(
            tasks_dir=str(tasks_dir),
            filter=None if subset_applied else options.filter,
            tag=None if subset_applied else options.tag,
            text_only=False if subset_applied else options.text_only,
            range=None if subset_applied else options.range,
            parallel=cfg.parallel,
            model=model_name,
            api_key=None,
            base_url=self.settings.target.inference_base_url.rstrip("/"),
            config=str(config_path),
            trials=cfg.trials,
            trace_dir=str(trace_root),
            judge_model=self.settings.judge.model.strip() or None,
            no_judge=options.no_judge,
            proxy=None,
            port_base_offset=0,
            sandbox=cfg.sandbox,
            sandbox_image=cfg.sandbox_image,
            sandbox_tools=cfg.sandbox_tools,
            rerun_errors=str(_resolve_path(cfg.rerun_errors)) if cfg.rerun_errors else None,
            continue_dir=str(trace_root) if cfg.continue_existing else None,
        )

    def _write_run_config(
        self,
        *,
        run_dir: Path,
        tasks_dir: Path,
        trace_root: Path,
        model_name: str,
        options: _ClawEvalRunOptions,
    ) -> Path:
        cfg = self.settings.claw_eval
        template_path = options.config_template
        payload: dict[str, Any] = {}
        if template_path.exists():
            payload = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}

        payload.setdefault("model", {})
        payload["model"].update(
            {
                "api_key": f"${{{cfg.api_key_env}}}",
                "base_url": self.settings.target.inference_base_url.rstrip("/"),
                "model_id": model_name,
            }
        )
        payload.setdefault("defaults", {})
        payload["defaults"].update(
            {
                "trace_dir": str(trace_root),
                "tasks_dir": str(tasks_dir),
            }
        )
        payload.setdefault("judge", {})
        if options.no_judge:
            payload["judge"]["enabled"] = False
        judge_base = self.settings.judge.openai_api_base.rstrip("/")
        judge_model = self.settings.judge.model.strip() or None
        if self.settings.judge.openai_api_key:
            payload["judge"]["api_key"] = "${OPENROUTER_API_KEY}"
        if judge_base:
            payload["judge"]["base_url"] = judge_base
        if judge_model:
            payload["judge"]["model_id"] = judge_model
        payload.setdefault("user_agent_model", {})
        ua_key = self.settings.resolve_claw_eval_user_agent_api_key()
        ua_base_cfg = self.settings.claw_eval.user_agent_model_url.strip()
        ua_model_cfg = self.settings.claw_eval.user_agent_model.strip()
        ua_base = ua_base_cfg or judge_base
        ua_model = ua_model_cfg or judge_model
        ua_key_dedicated = bool(
            os.environ.get("CLAW_USER_API_KEY", "").strip()
            or os.environ.get("DEEPSEEK_API_KEY", "").strip()
        )
        if ua_key:
            payload["user_agent_model"]["api_key"] = "${CLAW_USER_API_KEY}"
        if ua_base:
            payload["user_agent_model"]["base_url"] = ua_base
        if ua_model:
            payload["user_agent_model"]["model_id"] = ua_model
        if not ua_key_dedicated and not ua_base_cfg and not ua_model_cfg:
            logger.info(
                "  user_agent_model: using judge fallback  "
                "hint: set CLAW_USER_API_KEY or claw_eval.user_agent_model_url / "
                "claw_eval.user_agent_model to override"
            )
        payload.setdefault("sandbox", {})
        payload["sandbox"].update({"enabled": cfg.sandbox, "image": cfg.sandbox_image})

        config_path = run_dir / "claw_eval_config.yaml"
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return config_path

    def _parse_results(
        self,
        *,
        run_id: str,
        model_name: str,
        trace_root: Path,
        summary_path: Path | None,
        results_path: Path | None,
    ) -> RunResult:
        summary_payload: dict[str, Any] = {}
        batch_results: list[dict[str, Any]] | None = None
        if summary_path is not None:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if results_path is not None:
            batch_results = json.loads(results_path.read_text(encoding="utf-8"))
        result_summary = _summarize_batch_results(batch_results or [])

        trials = int(summary_payload.get("trials_per_task") or self.settings.claw_eval.trials)
        n_total = int(summary_payload.get("tasks") or 0)
        n_passed = int(summary_payload.get(f"pass_hat_{trials}") or 0)
        value = n_passed / n_total if n_total else None
        success = n_total > 0
        parse_error = None if success else "no claw-eval batch summary found"
        metric = f"pass_hat_{trials}" if value is not None else None

        return RunResult(
            success=success,
            run_id=run_id,
            model=model_name,
            metric=metric,
            value=value,
            n_total=n_total,
            n_passed=n_passed,
            summary={
                "tasks": n_total,
                "trials": trials,
                "pass_hat": n_passed,
                "pass_at": summary_payload.get(f"pass_at_{trials}"),
                "avg_score": summary_payload.get("avg_score"),
                "errored": summary_payload.get("errored"),
                "trace_dir": str(trace_root),
                "batch_summary_path": str(summary_path) if summary_path else None,
                "batch_results_path": str(results_path) if results_path else None,
                "batch_summary": summary_payload,
                "failed_task_ids": result_summary["failed_task_ids"],
                "trial_metrics": result_summary["trial_metrics"],
                "token_totals": result_summary["token_totals"],
                "parse_error": parse_error,
            },
            raw={"batch_results": batch_results} if batch_results is not None else None,
        )

    def _resolve_tasks_dir(self) -> Path:
        tasks_dir = _resolve_path(self.settings.claw_eval.tasks_dir)
        if not tasks_dir.is_dir():
            raise FileNotFoundError(f"Claw-Eval tasks_dir not found: {tasks_dir}")
        return tasks_dir

    def _prepare_task_subset(
        self,
        *,
        tasks_dir: Path,
        run_dir: Path,
        limit: int | None,
        task_ids: Sequence[str] | None,
        options: _ClawEvalRunOptions,
    ) -> Path:
        needs_subset = (
            limit is not None or bool(task_ids) or bool(options.language or options.category)
        )
        if not needs_subset:
            return tasks_dir

        selected = self._select_task_dirs(
            tasks_dir=tasks_dir,
            task_ids=task_ids,
            limit=limit,
            options=options,
        )
        if not selected:
            raise ValueError("no Claw-Eval tasks selected")

        subset_dir = run_dir / "task_subset"
        subset_dir.mkdir(parents=True, exist_ok=True)
        for task_dir in selected:
            link = subset_dir / task_dir.name
            if link.exists() or link.is_symlink():
                continue
            link.symlink_to(task_dir, target_is_directory=True)

        # claw-eval starts mock services with cwd=tasks_dir.parent; a run-local
        # task_subset shifts that parent to run_dir, so mirror mock_services there.
        mock_services_src = tasks_dir.parent / "mock_services"
        if mock_services_src.is_dir():
            mock_services_link = run_dir / "mock_services"
            if not mock_services_link.exists() and not mock_services_link.is_symlink():
                mock_services_link.symlink_to(mock_services_src, target_is_directory=True)

        return subset_dir

    def _select_task_dirs(
        self,
        *,
        tasks_dir: Path,
        task_ids: Sequence[str] | None,
        limit: int | None,
        options: _ClawEvalRunOptions,
    ) -> list[Path]:
        dirs = sorted(d for d in tasks_dir.iterdir() if d.is_dir() and (d / "task.yaml").exists())
        requested = set(task_ids or [])
        if requested:
            found = {d.name for d in dirs}
            missing = sorted(requested - found)
            if missing:
                raise ValueError(f"unknown Claw-Eval task id(s): {', '.join(missing)}")
            dirs = [d for d in dirs if d.name in requested]

        if options.filter:
            filt = options.filter.lower()
            dirs = [d for d in dirs if filt in str(d).lower()]
        if options.text_only:
            dirs = _filter_text_only(tasks_dir, dirs)
        if options.tag:
            dirs = [d for d in dirs if options.tag in _task_metadata(d / "task.yaml")["tags"]]
        if options.language:
            dirs = [
                d for d in dirs if _task_metadata(d / "task.yaml")["language"] == options.language
            ]
        if options.category:
            dirs = [
                d for d in dirs if _task_metadata(d / "task.yaml")["category"] == options.category
            ]
        if options.range:
            dirs = _filter_range(dirs, options.range)
        if limit is not None:
            dirs = dirs[:limit]
        return dirs

    def _check_local_assets(self) -> str | None:
        try:
            tasks_dir = self._resolve_tasks_dir()
        except FileNotFoundError as exc:
            return str(exc)
        if not any(tasks_dir.glob("*/task.yaml")):
            return f"no task.yaml files under {tasks_dir}"
        split_path = CLAW_EVAL_DIR / "splits" / "text_only.yaml"
        if self.settings.claw_eval.text_only and not split_path.is_file():
            return f"missing split manifest {split_path}"
        fixtures_error = _missing_fixture_error(tasks_dir)
        if fixtures_error:
            return fixtures_error
        return None

    @classmethod
    def download_tasks(
        cls,
        items: list[str],
        settings: Settings,
        *,
        check_only: bool = False,
        force: bool = False,
    ) -> DownloadReport:
        report = DownloadReport(backend="claw-eval")
        runner = cls(settings)
        for item in items:
            try:
                if item != "claw-eval":
                    raise ValueError("Claw-Eval download item must be 'claw-eval'")
                ensure_claw_eval()
                if check_only:
                    error = runner._check_local_assets()
                    if error:
                        raise FileNotFoundError(error)
                else:
                    _download_claw_eval_data(force=force)
                    _extract_fixtures(force=force)
                    error = runner._check_local_assets()
                    if error:
                        raise FileNotFoundError(error)
                report.succeeded.append(item)
                print(f"{'verified' if check_only else 'downloaded'}: {item}")
            except Exception as exc:
                report.failed[item] = str(exc)
                verb = "verify" if check_only else "download"
                print(f"failed to {verb} {item}: {exc}", file=sys.stderr)
        return report


class _Tee(io.TextIOBase):
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _run_claw_eval_batch(args: Namespace) -> None:
    cmd_batch = import_module("claw_eval.cli").cmd_batch

    try:
        cmd_batch(args)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code != 0:
            raise RuntimeError(f"claw-eval batch exited with code {code}") from exc


def _docker_image_exists(image: str) -> bool:
    if not image:
        return False
    try:
        docker = import_module("docker")
        client = docker.from_env()
        client.images.get(image)
        return True
    except Exception:
        return False


def _resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return REPO_ROOT / expanded


def _resolve_config_template(config_template: Path | None) -> Path:
    if config_template is not None:
        return _resolve_path(config_template)
    return CLAW_EVAL_DIR / "config_general.yaml"


def _default_run_id() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S")


def _find_latest_artifact(root: Path, name: str) -> Path | None:
    candidates = sorted(root.rglob(name), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _filter_text_only(tasks_dir: Path, task_dirs: list[Path]) -> list[Path]:
    split_path = tasks_dir.parent / "splits" / "text_only.yaml"
    if not split_path.is_file():
        split_path = CLAW_EVAL_DIR / "splits" / "text_only.yaml"
    if not split_path.is_file():
        return task_dirs
    payload = yaml.safe_load(split_path.read_text(encoding="utf-8")) or {}
    prefixes = tuple(payload.get("prefixes") or [])
    excluded = {entry.get("id") for entry in payload.get("excluded", []) if isinstance(entry, dict)}
    return [d for d in task_dirs if d.name.startswith(prefixes) and d.name not in excluded]


def _filter_range(task_dirs: list[Path], range_value: str) -> list[Path]:
    import re

    match = re.match(r"(\d+)-(\d+)$", range_value)
    if not match:
        raise ValueError(f"invalid Claw-Eval range {range_value!r}; expected L-R")
    lo, hi = int(match.group(1)), int(match.group(2))

    def in_range(path: Path) -> bool:
        task_match = re.match(r"T(\d+)", path.name)
        return task_match is not None and lo <= int(task_match.group(1)) <= hi

    return [d for d in task_dirs if in_range(d)]


def _task_metadata(task_yaml: Path) -> dict[str, Any]:
    payload = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
    prompt = payload.get("prompt") or {}
    tags = payload.get("tags") or []
    return {
        "category": str(payload.get("category") or ""),
        "language": str(prompt.get("language") or ""),
        "tags": [str(tag) for tag in tags],
    }


def _summarize_batch_results(batch_results: list[dict[str, Any]]) -> dict[str, Any]:
    failed_task_ids: list[str] = []
    trial_metrics: list[dict[str, Any]] = []
    token_totals = {
        "tokens": 0,
        "model_input_tokens": 0,
        "model_output_tokens": 0,
        "wall_time_s": 0.0,
        "model_time_s": 0.0,
        "tool_time_s": 0.0,
    }

    for result in batch_results:
        task_id = str(result.get("task_id") or "unknown")
        trials = result.get("trials") or []
        has_error = bool(result.get("error"))
        passed_all_trials = bool(trials) and all(bool(trial.get("passed")) for trial in trials)
        if has_error or not passed_all_trials:
            failed_task_ids.append(task_id)

        for index, trial in enumerate(trials):
            input_tokens = int(trial.get("model_input_tokens", trial.get("input_tokens", 0)) or 0)
            output_tokens = int(
                trial.get("model_output_tokens", trial.get("output_tokens", 0)) or 0
            )
            tokens = int(trial.get("tokens", input_tokens + output_tokens) or 0)
            wall_time_s = float(trial.get("wall_time_s", 0.0) or 0.0)
            model_time_s = float(trial.get("model_time_s", 0.0) or 0.0)
            tool_time_s = float(trial.get("tool_time_s", 0.0) or 0.0)
            token_totals["tokens"] += tokens
            token_totals["model_input_tokens"] += input_tokens
            token_totals["model_output_tokens"] += output_tokens
            token_totals["wall_time_s"] += wall_time_s
            token_totals["model_time_s"] += model_time_s
            token_totals["tool_time_s"] += tool_time_s
            trial_metrics.append(
                {
                    "task_id": task_id,
                    "trial": index,
                    "passed": trial.get("passed"),
                    "task_score": trial.get("task_score"),
                    "tokens": tokens,
                    "model_input_tokens": input_tokens,
                    "model_output_tokens": output_tokens,
                    "wall_time_s": wall_time_s,
                    "model_time_s": model_time_s,
                    "tool_time_s": tool_time_s,
                    "error": trial.get("error"),
                }
            )

    return {
        "failed_task_ids": failed_task_ids,
        "trial_metrics": trial_metrics,
        "token_totals": token_totals,
    }


def _missing_fixture_error(tasks_dir: Path) -> str | None:
    for task_yaml in sorted(tasks_dir.glob("*/task.yaml")):
        payload = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
        fixture_paths: list[str] = []
        environment = payload.get("environment") or {}
        fixture_paths.extend(str(p) for p in environment.get("fixtures") or [])
        fixture_paths.extend(str(p) for p in payload.get("sandbox_files") or [])
        fixture_paths.extend(str(p) for p in payload.get("sandbox_grader_files") or [])
        fixture_paths.extend(str(p) for p in payload.get("local_grader_files") or [])
        for rel_path in fixture_paths:
            if rel_path.startswith("tasks/"):
                candidate = CLAW_EVAL_DIR / rel_path
            else:
                candidate = task_yaml.parent / rel_path
            if not candidate.exists():
                return f"missing fixture for {task_yaml.parent.name}: {rel_path}"
    return None


_CLAW_EVAL_DATA_FILES = (
    "general-00000-of-00001.parquet",
    "multi_turn-00000-of-00001.parquet",
    "multimodal-00000-of-00001.parquet",
    "fixtures.tar.gz",
)


def _claw_eval_data_complete(data_dir: Path) -> bool:
    for name in _CLAW_EVAL_DATA_FILES:
        path = data_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            return False

    try:
        with tarfile.open(data_dir / "fixtures.tar.gz", "r:gz") as tar:
            tar.getmembers()
    except (tarfile.TarError, OSError):
        return False
    return True


def _download_claw_eval_data(*, force: bool) -> None:
    data_dir = CLAW_EVAL_DIR / "data"
    complete = _claw_eval_data_complete(data_dir)
    if complete and not force:
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download Claw-Eval data; "
            "install the lmms-eval/dev dependencies with uv sync"
        ) from exc

    snapshot_download(
        repo_id="claw-eval/Claw-Eval",
        repo_type="dataset",
        local_dir=CLAW_EVAL_DIR,
        allow_patterns="data/*",
        force_download=force or data_dir.exists(),
    )


def _extract_fixtures(*, force: bool) -> None:
    archive = CLAW_EVAL_DIR / "data" / "fixtures.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(f"fixtures archive not found: {archive}")
    target_dir = CLAW_EVAL_DIR / "tasks"
    with tempfile.TemporaryDirectory(prefix="jishubench_claw_fixtures_") as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(archive, "r:gz") as tar:
            _safe_extract(tar, tmp_path)
        for src in tmp_path.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(tmp_path)
            if rel.parts and rel.parts[0] == "tasks":
                rel = Path(*rel.parts[1:])
            dest = target_dir / rel
            if dest.exists() and not force:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)


def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise RuntimeError(f"unsafe tar member path: {member.name}")
    tar.extractall(destination, filter="data")
