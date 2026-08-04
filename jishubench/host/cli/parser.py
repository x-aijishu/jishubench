"""CLI parser: jishubench <command> argument definitions."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from host.config.profiles import CONFIG_ENV_VAR, resolve_config_path


def _config_path(value: str) -> Path:
    return resolve_config_path(value)


def _expanded_path(value: str) -> Path:
    return Path(value).expanduser()


def _default_config_path() -> Path:
    return resolve_config_path(None)


def _is_config_token(token: str) -> bool:
    return token == "--config" or token.startswith("--config=")


def _normalize_config_arg(tokens: Sequence[str], parser: argparse.ArgumentParser) -> list[str]:
    config_indexes = [index for index, token in enumerate(tokens) if _is_config_token(token)]
    if not config_indexes:
        return list(tokens)
    if len(config_indexes) > 1:
        parser.error("--config may only be specified once")

    index = config_indexes[0]
    token = tokens[index]
    if token == "--config":
        if index == len(tokens) - 1:
            parser.error("argument --config: expected one argument")
        config_args = [token, tokens[index + 1]]
        rest = [*tokens[:index], *tokens[index + 2 :]]
    else:
        config_args = [token]
        rest = [*tokens[:index], *tokens[index + 1 :]]

    if not rest:
        return list(tokens)

    command = rest[0]
    if command in {"preflight", "benchmarks", "run", "download"}:
        return [command, *config_args, *rest[1:]]
    if command == "target" and len(rest) > 1 and not rest[1].startswith("-"):
        return ["target", rest[1], *config_args, *rest[2:]]
    return list(tokens)


class JishuArgumentParser(argparse.ArgumentParser):
    def parse_args(  # type: ignore[override]
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        tokens = sys.argv[1:] if args is None else args
        normalized = _normalize_config_arg(tokens, self)
        return super().parse_args(normalized, namespace)


def _add_config_arg(
    parser: argparse.ArgumentParser,
    *,
    default: object | None = None,
) -> None:
    parser.add_argument(
        "--config",
        type=_config_path,
        default=_default_config_path() if default is None else default,
        help=(
            "Device profile (name like macstudio, path, or omit for "
            f"local/default; env {CONFIG_ENV_VAR})"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = JishuArgumentParser(
        prog="jishubench",
        description="Host-Target edge AI evaluation infrastructure",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    preflight_parser = sub.add_parser(
        "preflight",
        help="Check submodules and Target connectivity",
    )
    _add_config_arg(preflight_parser)
    preflight_parser.add_argument(
        "--benchmark",
        choices=("lmms-eval", "terminal-bench", "terminal-bench-2", "tau-bench", "claw-eval"),
        help="Run benchmark-specific preflight checks",
    )
    preflight_parser.add_argument(
        "--model",
        help="Model name on Target (default: from config)",
    )
    benchmarks = sub.add_parser(
        "benchmarks",
        help="List available lmms-eval benchmarks (default: leaf tasks)",
    )
    _add_config_arg(benchmarks)
    benchmarks.add_argument(
        "--kind",
        choices=("subtasks", "groups", "tags", "all"),
        default="subtasks",
        help="Which benchmark names to list (default: subtasks)",
    )
    benchmarks.add_argument(
        "--search",
        metavar="QUERY",
        help="Filter benchmark names by case-insensitive substring match",
    )

    run = sub.add_parser(
        "run",
        help=(
            "Run benchmark(s) via Target inference "
            "(supports lmms-eval, tau-bench, terminal-bench, terminal-bench-2, claw-eval)"
        ),
    )
    _add_config_arg(run)
    run.add_argument(
        "benchmarks",
        nargs="+",
        metavar="BENCHMARK",
        help="Task names (lmms-eval), or tau-bench / terminal-bench / terminal-bench-2 / claw-eval",
    )
    run.add_argument("--model", help="Model name on Target (default: from config)")
    run.add_argument("--limit", type=int, help="Limit samples or tasks")
    run.add_argument(
        "--task-id",
        action="append",
        dest="task_id",
        metavar="TASK_ID",
        help="Run specific task id(s) only; repeatable (agent benchmarks)",
    )
    run.add_argument(
        "--domain",
        action="append",
        dest="domain",
        metavar="DOMAIN",
        help="Tau-Bench domain(s) to run; repeatable",
    )
    run.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable judge scoring where supported (claw-eval)",
    )
    run.add_argument("--claw-filter", help="Claw-Eval task id/path substring filter")
    run.add_argument("--claw-tag", help="Claw-Eval tag filter")
    run.add_argument("--claw-range", help="Claw-Eval T-task numeric range, e.g. 1-104")
    run.add_argument("--claw-language", help="Claw-Eval prompt language filter, e.g. zh or en")
    run.add_argument("--claw-category", help="Claw-Eval task category filter")
    run.add_argument(
        "--log-samples",
        "--save-samples",
        dest="save_samples",
        action="store_true",
        default=None,
        help="Save per-sample model outputs under the run directory",
    )

    download = sub.add_parser(
        "download",
        help="Pre-download benchmark datasets to configured cache directories",
    )
    _add_config_arg(download)
    download.add_argument(
        "--backend",
        default=None,
        help="Dataset backend override (default: infer from BENCHMARK, otherwise lmms-eval)",
    )
    download.add_argument(
        "benchmarks",
        nargs="+",
        metavar="BENCHMARK",
        help="Benchmark names, or an agent backend object such as claw-eval",
    )
    download.add_argument(
        "--hub-cache",
        type=_expanded_path,
        metavar="PATH",
        help="Override lmms_eval.hub_cache (lmms-eval backend only)",
    )
    download.add_argument(
        "--datasets-cache",
        type=_expanded_path,
        metavar="PATH",
        help="Override lmms_eval.datasets_cache (lmms-eval backend only)",
    )
    download.add_argument(
        "--check",
        action="store_true",
        help="Verify local cache only; do not download from the network",
    )
    download.add_argument(
        "--force",
        action="store_true",
        help="Force re-download where the backend supports it",
    )

    target = sub.add_parser("target", help="Manage Target llama.cpp server")
    target_sub = target.add_subparsers(dest="target_command", required=True)

    t_status = target_sub.add_parser("status", help="Show model / slots / readiness")
    _add_config_arg(t_status)
    t_status.add_argument("--model", help="Model name (default: from config)")

    t_models = target_sub.add_parser(
        "models",
        help="List all models on Target (GET /models)",
    )
    _add_config_arg(t_models)
    t_models.add_argument(
        "--reload",
        action="store_true",
        help="Rescan models-dir / cache before listing (?reload=1)",
    )
    t_models.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON catalog (default: one model name per line)",
    )

    t_load = target_sub.add_parser("load", help="Load model on Target")
    _add_config_arg(t_load)
    t_load.add_argument("--model", help="Model name (default: from config)")

    t_unload = target_sub.add_parser("unload", help="Unload model on Target")
    _add_config_arg(t_unload)
    t_unload.add_argument(
        "--model",
        help="Model name to unload (default: the only loaded model on Target; "
        "fails if 0 or >1 are loaded)",
    )

    config = sub.add_parser("config", help="Select and inspect device profiles")
    config_sub = config.add_subparsers(dest="config_command", required=True)

    config_sub.add_parser("list", help="List available device profiles")
    config_sub.add_parser("show", help="Print the active profile path")

    config_use = config_sub.add_parser(
        "use",
        help="Set personal default (~/.jishubench/profile)",
    )
    config_use.add_argument(
        "profile",
        nargs="?",
        metavar="PROFILE",
        help="Profile name (e.g. macstudio); interactive if omitted",
    )

    return parser
