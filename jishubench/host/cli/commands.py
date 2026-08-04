"""CLI subcommand handlers for jishubench."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from loguru import logger

from host.adapters import Runner
from host.adapters.claw_eval import ClawEvalRunArgs, ClawEvalRunner
from host.adapters.lmms_eval import LmmsEvalRunArgs, LmmsEvalRunner, apply_download_cache_overrides
from host.adapters.tau_bench import TauBenchRunArgs, TauBenchRunner
from host.adapters.terminal_bench import TerminalBenchRunArgs, TerminalBenchRunner
from host.adapters.terminal_bench_2 import TerminalBench2RunArgs, TerminalBench2Runner
from host.client.llama_cpp_client import LlamaCppClient
from host.config import (
    Settings,
    ensure_claw_eval,
    ensure_lmms_eval,
    ensure_tau2_bench,
    load_settings,
    setup_console_logging,
    setup_logging,
)
from host.config.profiles import (
    ACTIVE_PROFILE_FILE,
    activate_profile,
    active_profile_name,
    list_device_profiles,
)
from host.controller import HostController

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKEND_RUNNERS: dict[str, type[Runner]] = {
    "lmms-eval": LmmsEvalRunner,
    "tau-bench": TauBenchRunner,
    "terminal-bench": TerminalBenchRunner,
    "terminal-bench-2": TerminalBench2Runner,
    "claw-eval": ClawEvalRunner,
}

_DOWNLOAD_OBJECT_BACKENDS = {"tau-bench", "terminal-bench", "terminal-bench-2", "claw-eval"}

_BACKEND_SUBMODULE_CHECKS: dict[str, Callable[[], Any]] = {
    "lmms-eval": ensure_lmms_eval,
    "tau-bench": ensure_tau2_bench,
    "claw-eval": ensure_claw_eval,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _effective_model(model: str | None, settings: Settings) -> str:
    """Return explicit --model or config default."""
    if model is not None:
        return model
    return settings.target.model


def _run_with_controller(
    *,
    config: Any,
    model: str | None,
    benchmark: str,
    engine: str,
    adapter_cls: type[Runner],
    build_args: Callable[[str, str], Any],
) -> int:
    """Common run lifecycle: load settings, preflight, run, and return an exit code."""
    settings = load_settings(config)
    setup_logging(settings)
    effective = _effective_model(model, settings)
    if model is None:
        logger.info("Use Default Model: {}", effective)

    controller = HostController(settings)
    status = controller.preflight(
        effective,
        adapter_cls=adapter_cls,
    )
    if not status.get("ready", False):
        logger.error("Preflight failed: {}", json.dumps(status, ensure_ascii=False))
        return 1

    try:
        result = controller.run(
            model=effective,
            benchmark=benchmark,
            engine=engine,
            adapter_cls=adapter_cls,
            build_args=build_args,
        )
        return 0 if result.success else 1
    except RuntimeError as exc:
        logger.exception("run failed: {}", exc)
        return 1


def _ensure_backend_submodule(backend: str) -> int:
    check = _BACKEND_SUBMODULE_CHECKS.get(backend)
    if check is None:
        return 0
    try:
        check()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_benchmarks(args: argparse.Namespace) -> int:
    if (code := _ensure_backend_submodule("lmms-eval")) != 0:
        return code

    settings = load_settings(args.config)
    setup_console_logging(settings)
    try:
        names = LmmsEvalRunner(settings).list_benchmarks(
            kind=args.kind,
            search=args.search,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    for name in names:
        print(name)
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    controller = HostController(settings)
    effective_model = _effective_model(getattr(args, "model", None), settings)
    benchmark = getattr(args, "benchmark", None)
    adapter_cls = _BACKEND_RUNNERS.get(benchmark) if isinstance(benchmark, str) else None
    status = controller.preflight(
        effective_model,
        adapter_cls=adapter_cls,
    )
    return 0 if status.get("ready", False) else 1


def _cmd_run(args: argparse.Namespace) -> int:
    benchmark = args.benchmarks[0]
    no_extra_benchmarks = {"tau-bench", "terminal-bench", "terminal-bench-2", "claw-eval"}
    if benchmark in no_extra_benchmarks and len(args.benchmarks) > 1:
        extras = ", ".join(args.benchmarks[1:])
        print(
            f"{benchmark} accepts no additional benchmark arguments: {extras}",
            file=sys.stderr,
        )
        print("Use --task-id to select specific agent benchmark tasks.", file=sys.stderr)
        return 1

    if benchmark == "tau-bench":
        if (code := _ensure_backend_submodule(benchmark)) != 0:
            return code
        return _run_with_controller(
            config=args.config,
            model=args.model,
            benchmark="tau-bench",
            engine="tau_bench",
            adapter_cls=TauBenchRunner,
            build_args=lambda m, run_id: TauBenchRunArgs(
                model_name=m,
                domains=args.domain or None,
                limit=args.limit,
                task_ids=args.task_id,
                run_id=run_id,
            ),
        )

    elif benchmark == "terminal-bench":
        return _run_with_controller(
            config=args.config,
            model=args.model,
            benchmark="terminal-bench",
            engine="terminal_bench",
            adapter_cls=TerminalBenchRunner,
            build_args=lambda m, run_id: TerminalBenchRunArgs(
                model_name=m,
                limit=args.limit,
                task_ids=args.task_id,
                run_id=run_id,
            ),
        )

    elif benchmark == "terminal-bench-2":
        return _run_with_controller(
            config=args.config,
            model=args.model,
            benchmark="terminal-bench-2",
            engine="terminal_bench_2",
            adapter_cls=TerminalBench2Runner,
            build_args=lambda m, run_id: TerminalBench2RunArgs(
                model_name=m,
                limit=args.limit,
                task_ids=args.task_id,
                run_id=run_id,
            ),
        )

    elif benchmark == "claw-eval":
        if (code := _ensure_backend_submodule(benchmark)) != 0:
            return code
        return _run_with_controller(
            config=args.config,
            model=args.model,
            benchmark="claw-eval",
            engine="claw_eval",
            adapter_cls=ClawEvalRunner,
            build_args=lambda m, run_id: ClawEvalRunArgs(
                model_name=m,
                limit=args.limit,
                task_ids=args.task_id,
                run_id=run_id,
                no_judge=args.no_judge or None,
                filter=args.claw_filter,
                tag=args.claw_tag,
                range=args.claw_range,
                language=args.claw_language,
                category=args.claw_category,
            ),
        )

    # Default: lmms-eval
    return _run_with_controller(
        config=args.config,
        model=args.model,
        benchmark="+".join(args.benchmarks),
        engine="lmms_eval",
        adapter_cls=LmmsEvalRunner,
        build_args=lambda m, run_id: LmmsEvalRunArgs(
            tasks=args.benchmarks,
            model_name=m,
            limit=args.limit,
            save_samples=args.save_samples,
            run_id=run_id,
        ),
    )


def _cmd_download(args: argparse.Namespace) -> int:
    backend, items = _resolve_download_request(args)

    if (code := _ensure_backend_submodule(backend)) != 0:
        return code

    runner_cls = _BACKEND_RUNNERS.get(backend)
    if runner_cls is None:
        available = ", ".join(sorted(_BACKEND_RUNNERS))
        print(
            f"unknown download backend {backend!r}; available: {available}",
            file=sys.stderr,
        )
        return 1

    settings = load_settings(args.config)
    setup_console_logging(settings)
    settings = apply_download_cache_overrides(
        settings,
        hub_cache=args.hub_cache,
        datasets_cache=args.datasets_cache,
    )

    try:
        report = runner_cls.download_tasks(
            items,
            settings=settings,
            check_only=args.check,
            force=args.force,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    if report.succeeded:
        print(
            f"backend={report.backend} succeeded ({len(report.succeeded)}): "
            f"{', '.join(report.succeeded)}"
        )
    if report.failed:
        for name, message in report.failed.items():
            print(f"  {name}: {message}", file=sys.stderr)
        return 1
    return 0


def _resolve_download_request(args: argparse.Namespace) -> tuple[str, list[str]]:
    items = list(args.benchmarks)
    if args.backend is not None:
        return args.backend, items
    if len(items) == 1 and items[0] in _DOWNLOAD_OBJECT_BACKENDS:
        return items[0], items
    return "lmms-eval", items


# ---------------------------------------------------------------------------
# target subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_target_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    controller = HostController(settings)
    model = _effective_model(args.model, settings)
    status = controller.preflight(model)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if status.get("ready", False) else 1


def _cmd_target_load(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    client = LlamaCppClient(settings)
    model = _effective_model(args.model, settings)
    resolved = client.resolve_model_id(model)
    try:
        match = client.require_model_in_catalog(model, rescan=True)
        models = client.list_models()
        pre_mmproj_path = client.resolve_mmproj_path(match, models)
        if pre_mmproj_path:
            print(f"loading {resolved} (mmproj: {pre_mmproj_path}) ...")
        else:
            print(f"loading {resolved} ...")
        loaded_id = client.ensure_model_loaded(model)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    # Re-resolve mmproj from the post-load catalog -- the router may have
    # changed the pairing during load.
    refreshed = client.find_model(loaded_id) or match
    mmproj_path = client.resolve_mmproj_path(refreshed, models)
    if mmproj_path:
        print(f"loaded: {loaded_id} (mmproj: {mmproj_path})")
    else:
        print(f"loaded: {loaded_id}")
    return 0


def _cmd_target_unload(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    client = LlamaCppClient(settings)

    if args.model is not None:
        model = args.model
        resolved = client.resolve_model_id(model)
        print(f"unloading {resolved} ...")
        ok = client.unload_model(model)
        if ok:
            print(f"unloaded: {resolved}")
        else:
            print(f"failed to unload {resolved}", file=sys.stderr)
        return 0 if ok else 1

    # No explicit model: infer from live Target state.
    try:
        models = client.list_models()
    except Exception as exc:
        print(f"failed to list models on Target: {exc}", file=sys.stderr)
        print(
            "specify the model explicitly: jishubench target unload --model NAME",
            file=sys.stderr,
        )
        return 1

    loaded = _loaded_model_names(models)
    if not loaded:
        print("no model is currently loaded on Target", file=sys.stderr)
        print("load one with: jishubench target load --model NAME", file=sys.stderr)
        print("or unload a specific model: jishubench target unload --model NAME", file=sys.stderr)
        return 1
    if len(loaded) > 1:
        print("multiple models are loaded on Target; specify which to unload", file=sys.stderr)
        for mid in loaded:
            print(f"  - {mid}", file=sys.stderr)
        print("jishubench target unload --model NAME", file=sys.stderr)
        return 1

    resolved = loaded[0]
    print(f"unloading {resolved} ...")
    ok = client.unload_model(resolved)
    if ok:
        print(f"unloaded: {resolved}")
    else:
        print(f"failed to unload {resolved}", file=sys.stderr)
    return 0 if ok else 1


def _usable_model_names(models: list) -> list[str]:
    """Inference-loadable model ids (exclude mmproj projector entries)."""
    from host.client.llama_cpp_client import is_mmproj_model_id

    return sorted(m.id for m in models if m.id and not is_mmproj_model_id(m.id))


def _loaded_model_names(models: list) -> list[str]:
    """Inference-loadable model ids currently in the 'loaded' state."""
    from host.client.llama_cpp_client import is_mmproj_model_id

    return sorted(
        m.id
        for m in models
        if m.id and getattr(m, "status", None) == "loaded" and not is_mmproj_model_id(m.id)
    )


def _cmd_target_models(args: argparse.Namespace) -> int:
    """List loadable model names on the Target llama-server router."""
    from dataclasses import asdict

    settings = load_settings(args.config)
    client = LlamaCppClient(settings)
    try:
        models = client.list_models(reload=args.reload)
    except Exception as exc:
        print(f"failed to list models: {exc}", file=sys.stderr)
        return 1

    if args.json:
        names = _usable_model_names(models)
        payload = {
            "base_url": settings.target.inference_root_url(),
            "reload": args.reload,
            "count": len(names),
            "models": [asdict(m) for m in models if m.id in names],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    for name in _usable_model_names(models):
        print(name)
    return 0


def _cmd_target(args: argparse.Namespace) -> int:
    sub_handlers = {
        "status": _cmd_target_status,
        "models": _cmd_target_models,
        "load": _cmd_target_load,
        "unload": _cmd_target_unload,
    }
    return sub_handlers[args.target_command](args)


# ---------------------------------------------------------------------------
# config subcommand handlers
# ---------------------------------------------------------------------------


def _select_profile_interactive() -> str | None:
    profiles = list_device_profiles()
    if not profiles:
        print("No shared profiles found in configs/devices/", file=sys.stderr)
        return None

    current = active_profile_name()
    default_idx = 0
    for i, profile in enumerate(profiles):
        if profile.name == current:
            default_idx = i
            break

    for i, profile in enumerate(profiles):
        marker = " (current)" if profile.name == current else ""
        print(f"  [{i + 1}] {profile.name}{marker}  {profile.summary()}")

    default_choice = default_idx + 1
    try:
        choice = input(f"Select profile [{default_choice}]: ").strip()
    except EOFError:
        print(file=sys.stderr)
        return None

    if not choice:
        return profiles[default_idx].name
    if choice.isdigit():
        index = int(choice) - 1
        if 0 <= index < len(profiles):
            return profiles[index].name
        print(f"Invalid selection: {choice}", file=sys.stderr)
        return None

    return choice


def _cmd_config_list(_args: argparse.Namespace) -> int:
    profiles = list_device_profiles()
    active = active_profile_name()
    for profile in profiles:
        marker = " *" if profile.name == active else ""
        print(f"{profile.name}{marker}\t{profile.summary()}")
    return 0


def _cmd_config_show(_args: argparse.Namespace) -> int:
    from host.config.profiles import resolve_config_path

    try:
        path = resolve_config_path(None)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(path)
    return 0


def _cmd_config_use(args: argparse.Namespace) -> int:
    name = args.profile
    if name is None:
        if not sys.stdin.isatty():
            print(
                "Profile name required in non-interactive mode, e.g. "
                "`jishubench config use macstudio`",
                file=sys.stderr,
            )
            return 1
        name = _select_profile_interactive()
        if name is None:
            return 1

    try:
        path = activate_profile(name)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Active profile: {name} -> {path}")
    print(f"Saved selection to {ACTIVE_PROFILE_FILE}")
    print("This profile is now the default when --config is omitted.")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    sub_handlers = {
        "list": _cmd_config_list,
        "show": _cmd_config_show,
        "use": _cmd_config_use,
    }
    return sub_handlers[args.config_command](args)


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, Any] = {
    "benchmarks": _cmd_benchmarks,
    "config": _cmd_config,
    "download": _cmd_download,
    "preflight": _cmd_preflight,
    "target": _cmd_target,
    "run": _cmd_run,
}
