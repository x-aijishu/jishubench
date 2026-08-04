"""lmms-eval runner — in-process simple_evaluate() pointed at Target async_openai."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, Sequence

from loguru import logger

from host.adapters import DownloadReport, RunResult
from host.config import Settings, ensure_lmms_eval, lmms_eval_env, load_settings


@dataclass
class LmmsEvalRunArgs:
    tasks: Sequence[str]
    model_name: str | None = None
    limit: int | None = None
    save_samples: bool | None = None
    run_id: str | None = None


_METADATA_KEYS = frozenset({"alias", "samples"})
_PREFERRED_ACCURACY_METRICS = ("exact_match", "acc", "acc_score")
_METRIC_NOISE_SUFFIXES = ("_stderr", "_stderr_clt", "_stderr_clustered")
# lmms-eval tasks often list sub-category metrics before the headline aggregate
# (e.g. mmstar puts "average" last; snsbench puts "average" first).
_HEADLINE_METRIC_NAMES = frozenset(
    {
        "average",
        "overall",
        "overall_accuracy",
        "overall_score",
        "score_overall",
    }
)
_REDACTED = "[REDACTED]"


def _redact_known_secrets(value: Any, secrets: Sequence[str]) -> Any:
    """Return a copy with configured secret values removed from serializable data."""
    known = tuple(
        sorted(
            {secret for secret in secrets if secret and secret != "EMPTY"},
            key=len,
            reverse=True,
        )
    )
    if not known:
        return value
    if isinstance(value, str):
        for secret in known:
            value = value.replace(secret, _REDACTED)
        return value
    if isinstance(value, dict):
        return {
            _redact_known_secrets(key, known): _redact_known_secrets(item, known)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_known_secrets(item, known) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_known_secrets(item, known) for item in value)
    return value


class LmmsEvalRunner:
    """Host Processor Mode: dataset + scoring on Host, inference via Target OpenAI API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

    def preflight(self) -> bool:
        n_errors = 0

        try:
            ensure_lmms_eval()
            logger.info("  lmms-eval submodule: OK")
        except FileNotFoundError as exc:
            logger.error("  lmms-eval submodule: {}", exc)
            n_errors += 1

        if util.find_spec("lmms_eval") is not None:
            logger.info("  lmms_eval package: OK")
        else:
            logger.error("  lmms_eval package: not discoverable  hint: uv sync --extra lmms-eval")
            n_errors += 1

        return n_errors == 0

    @staticmethod
    def _resolve_tasks(task_manager: Any, tasks: list[str]) -> list[str]:
        task_names = task_manager.match_tasks(tasks)
        missing = [name for name in tasks if name not in task_names and "*" not in name]
        if missing:
            available = ", ".join(sorted(task_manager.all_tasks)[:20])
            raise ValueError(
                f"Tasks not found: {', '.join(missing)}. "
                f"Use `jishubench benchmarks` or check task names. Sample: {available}..."
            )
        return task_names

    def list_benchmarks(
        self,
        *,
        kind: str = "subtasks",
        search: str | None = None,
    ) -> list[str]:
        ensure_lmms_eval()
        with lmms_eval_env(self.settings):
            from lmms_eval.tasks import TaskManager

            tm = TaskManager(verbosity="INFO")
            if kind == "subtasks":
                names = list(tm.all_subtasks)
            elif kind == "groups":
                names = list(tm.all_groups)
            elif kind == "tags":
                names = list(tm.all_tags)
            elif kind == "all":
                names = list(tm.all_tasks)
            else:
                raise ValueError(f"unknown kind: {kind}")

            if search:
                needle = search.casefold()
                names = [name for name in names if needle in name.casefold()]
            return names

    def run(self, args: LmmsEvalRunArgs, *, output_dir: Path | None = None) -> RunResult:
        if output_dir is not None and args.run_id is None:
            raise ValueError("run_id is required when output_dir is externally managed")

        ensure_lmms_eval()
        with lmms_eval_env(self.settings):
            from lmms_eval import models as lmms_models
            from lmms_eval.evaluator import simple_evaluate
            from lmms_eval.loggers import EvaluationTracker
            from lmms_eval.tasks import TaskManager

            # lmms-eval models/__init__.py calls logger.remove() at import time
            # (triggered by evaluator → models chain), wiping our loguru sinks
            # (file + stderr). Re-instate them so lmms-eval's eval_logger
            # output is captured in the log file.
            from host.config.logging_setup import setup_logging

            setup_logging(self.settings, force=True)

            model = args.model_name or self.settings.target.model
            task_manager = TaskManager(verbosity="INFO", model_name=model)
            task_names = LmmsEvalRunner._resolve_tasks(task_manager, list(args.tasks))

            output_path = output_dir or self.settings.ensure_work_dir()
            resolved_output_path = output_path.resolve()
            tracker = EvaluationTracker(output_path=str(resolved_output_path))
            save_samples = (
                args.save_samples
                if args.save_samples is not None
                else self.settings.eval.save_samples
            )
            # lmms-eval postprocess reads cli_args.output_path (e.g. hallusion judge
            # artifacts) and cli_args.process_with_media when iterating docs for scoring
            # / per-sample logs. Match upstream CLI defaults: process_with_media=False
            # unless we are saving samples (vision docs need media attached).
            cli_args = SimpleNamespace(
                output_path=str(resolved_output_path),
                process_with_media=save_samples,
            )

            logger.info("lmms-eval → Target {}", self.settings.target.inference_base_url)
            logger.info("  tasks: {}  model: {}", ", ".join(task_names), model)
            if args.limit is not None:
                logger.info("  limit: {}", args.limit)
            if save_samples:
                logger.info("  save_samples: true")
            persisted_model_args = self.settings.build_lmms_model_args(model)
            model_class = lmms_models.get_model(self.settings.lmms_eval.model_backend)
            runtime_model = model_class(**self.settings.build_lmms_model_kwargs(model))
            results = simple_evaluate(
                model=runtime_model,
                model_args=persisted_model_args,
                tasks=task_names,
                task_manager=task_manager,
                batch_size=self.settings.lmms_eval.batch_size,
                limit=args.limit,
                log_samples=save_samples,
                evaluation_tracker=tracker,
                cli_args=cli_args,
            )
            if results is not None:
                from lmms_eval import utils as lmms_utils

                results = _redact_known_secrets(
                    results,
                    (
                        self.settings.target.resolved_inference_api_key(),
                        self.settings.judge.openai_api_key.strip(),
                    ),
                )
                samples = results.pop("samples", None) if save_samples else None
                datetime_str = lmms_utils.get_datetime_str()
                cfg_tracker = getattr(tracker, "general_config_tracker", None)
                if cfg_tracker is not None:
                    cfg_tracker.model_source = self.settings.lmms_eval.model_backend
                tracker.save_results_aggregated(
                    results=results,
                    samples=samples,
                    datetime_str=datetime_str,
                )
                if save_samples and samples:
                    for task_name in results.get("configs", {}):
                        tracker.save_results_samples(
                            task_name=task_name,
                            samples=samples[task_name],
                        )
                logger.info("results saved under {}", output_path)

                metric, value, n_total, n_passed = _extract_lmms_eval_stats(results)
                return RunResult(
                    success=True,
                    run_id=args.run_id or "",
                    model=model,
                    metric=metric,
                    value=value,
                    n_total=n_total,
                    n_passed=n_passed,
                    summary={
                        "configs": results.get("configs"),
                        "version": results.get("version"),
                    },
                    raw=results,
                )

            return RunResult(
                success=False,
                run_id=args.run_id or "",
                model=model,
                metric=None,
                value=None,
                n_total=0,
                n_passed=0,
                summary={},
                raw=None,
            )

    @classmethod
    def download_tasks(
        cls,
        items: list[str],
        settings: Settings,
        *,
        check_only: bool = False,
        force: bool = False,
    ) -> DownloadReport:
        """Download or verify datasets for lmms-eval tasks."""
        report = DownloadReport(backend="lmms-eval")
        _ensure_cache_dirs(settings)

        with lmms_eval_env(settings):
            from lmms_eval.tasks import TaskManager

            tm = TaskManager(verbosity="INFO", model_name=settings.target.model)

            for item in items:
                try:
                    task_names = cls._resolve_tasks(tm, [item])
                    for name in task_names:
                        load_args = _build_load_args(
                            tm,
                            name,
                            check_only=check_only,
                            force=force,
                        )
                        tm.load_task_or_group([load_args])
                    report.succeeded.append(item)
                    verb = "verified" if check_only else "downloaded"
                    print(f"{verb}: {item}")
                except Exception as exc:
                    verb = "verify" if check_only else "download"
                    report.failed[item] = str(exc)
                    print(f"failed to {verb} {item}: {exc}", file=sys.stderr)

        return report


def _metric_base_name(key: str) -> str:
    return key.split(",", 1)[0]


def _is_scored_metric_key(key: str) -> bool:
    if key in _METADATA_KEYS:
        return False
    base = _metric_base_name(key)
    return not any(noise in base for noise in _METRIC_NOISE_SUFFIXES)


def _get_metric_value(task_result: dict[str, Any], metric_name: str) -> float | None:
    direct = task_result.get(metric_name)
    if isinstance(direct, (int, float)):
        return float(direct)

    none_key = f"{metric_name},none"
    filtered = task_result.get(none_key)
    if isinstance(filtered, (int, float)):
        return float(filtered)

    prefix = f"{metric_name},"
    for key, value in task_result.items():
        if key.startswith(prefix) and isinstance(value, (int, float)):
            return float(value)
    return None


def _is_headline_metric(metric_name: str) -> bool:
    normalized = metric_name.casefold().replace(" ", "_")
    if normalized in _HEADLINE_METRIC_NAMES:
        return True
    return normalized.startswith("overall_") or normalized.endswith("_overall")


def _preferred_metrics_for_task(
    configs: dict[str, Any],
    task_name: str,
) -> list[str]:
    task_config = configs.get(task_name, {})
    metric_list = task_config.get("metric_list") or []
    ordered: list[str] = []
    seen: set[str] = set()
    for item in metric_list:
        if not isinstance(item, dict):
            continue
        name = item.get("metric")
        if isinstance(name, str) and name not in seen:
            seen.add(name)
            ordered.append(name)

    preferred = list(ordered)
    headline = [name for name in ordered if _is_headline_metric(name)]
    if headline:
        primary = headline[0]
        preferred = [primary] + [name for name in ordered if name != primary]

    for fallback in _PREFERRED_ACCURACY_METRICS:
        if fallback not in seen:
            seen.add(fallback)
            preferred.append(fallback)
    return preferred


def _pick_primary_metric(
    task_result: dict[str, Any],
    *,
    preferred: Sequence[str] | None = None,
) -> tuple[str, float] | None:
    for metric_name in preferred or _PREFERRED_ACCURACY_METRICS:
        value = _get_metric_value(task_result, metric_name)
        if value is not None:
            return metric_name, value

    for key, value in task_result.items():
        if _is_scored_metric_key(key) and isinstance(value, (int, float)):
            return _metric_base_name(key), float(value)
    return None


def _task_sample_count(
    task_name: str,
    task_result: dict[str, Any],
    n_samples: dict[str, Any] | None,
) -> int:
    samples = task_result.get("samples")
    if isinstance(samples, int) and samples > 0:
        return samples
    if n_samples and task_name in n_samples:
        task_samples = n_samples[task_name]
        if isinstance(task_samples, dict):
            effective = task_samples.get("effective")
            if isinstance(effective, int) and effective > 0:
                return effective
    return 0


def _extract_lmms_eval_stats(
    results: dict[str, Any],
) -> tuple[str | None, float | dict[str, dict[str, float | str]] | None, int, int]:
    """Extract headline metric/value and aggregate sample counts.

    Returns ``(metric, value, total_samples, n_passed_estimate)``.
    Single task → ``(metric_name, float, ...)``.
    Multi-task → ``(None, {task: {"metric": name, "value": score}}, ...)``.
    ``n_passed_estimate`` is the weighted-sum-of-correct rounded to an int
    (an approximation, since different tasks use different metrics).
    """
    per_task_results = results.get("results", {})
    if not isinstance(per_task_results, dict) or not per_task_results:
        return None, None, 0, 0

    configs = results.get("configs", {})
    n_samples = results.get("n-samples")
    if not isinstance(configs, dict):
        configs = {}
    if not isinstance(n_samples, dict):
        n_samples = None

    per_task: dict[str, dict[str, float | str]] = {}
    weighted_sum = 0.0
    total_samples = 0

    for task_name, task_result in per_task_results.items():
        if not isinstance(task_result, dict):
            continue

        preferred = _preferred_metrics_for_task(configs, task_name)
        picked = _pick_primary_metric(task_result, preferred=preferred)
        if picked is None:
            continue

        metric_name, score = picked
        per_task[task_name] = {"metric": metric_name, "value": score}

        sample_count = _task_sample_count(task_name, task_result, n_samples)
        if sample_count > 0:
            weighted_sum += score * sample_count
            total_samples += sample_count

    if not per_task:
        return None, None, total_samples or len(per_task_results), 0

    n_passed = round(weighted_sum) if total_samples > 0 else 0
    if len(per_task) == 1:
        entry = next(iter(per_task.values()))
        return str(entry["metric"]), float(entry["value"]), total_samples, n_passed
    return None, per_task, total_samples, n_passed


# ---------------------------------------------------------------------------
# Dataset download helpers
# ---------------------------------------------------------------------------


@dataclass
class _LmmsEvalTaskManager(Protocol):
    def _get_config(self, name: str) -> dict: ...


def _ensure_cache_dirs(settings: Settings) -> tuple[str | None, str | None]:
    hub = settings.lmms_eval.hub_cache
    datasets = settings.lmms_eval.datasets_cache
    hub_resolved: str | None = None
    datasets_resolved: str | None = None
    if hub is not None:
        resolved = hub.expanduser()
        resolved.mkdir(parents=True, exist_ok=True)
        hub_resolved = str(resolved.resolve())
    if datasets is not None:
        resolved = datasets.expanduser()
        resolved.mkdir(parents=True, exist_ok=True)
        datasets_resolved = str(resolved.resolve())
    return hub_resolved, datasets_resolved


def _merge_dataset_kwargs(base: dict | None, overrides: dict) -> dict:
    merged = dict(base or {})
    merged.update(overrides)
    return merged


def _build_load_args(
    task_manager: _LmmsEvalTaskManager,
    name: str,
    *,
    check_only: bool,
    force: bool,
) -> dict:
    base = task_manager._get_config(name)
    overrides: dict = {}
    if check_only:
        overrides["local_files_only"] = True
    if force:
        overrides["force_download"] = True
    args: dict = {"task": name}
    if overrides:
        args["dataset_kwargs"] = _merge_dataset_kwargs(
            (base or {}).get("dataset_kwargs"),
            overrides,
        )
    return args


def apply_download_cache_overrides(
    settings: Settings,
    *,
    hub_cache: Path | None = None,
    datasets_cache: Path | None = None,
) -> Settings:
    """Return settings with optional lmms_eval cache path overrides."""
    if hub_cache is None and datasets_cache is None:
        return settings
    lmms = settings.lmms_eval.model_copy(deep=True)
    if hub_cache is not None:
        lmms.hub_cache = hub_cache
    if datasets_cache is not None:
        lmms.datasets_cache = datasets_cache
    return settings.model_copy(update={"lmms_eval": lmms})
