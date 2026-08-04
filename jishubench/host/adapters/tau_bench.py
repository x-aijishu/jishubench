"""Tau-Bench runner — in-process tau2 simulation on Host, inference on Target + remote user LLM.

Architecture (mirrors terminal_bench.py):
- Agent (被测模型): Target llama-server via OpenAI-compatible client
- User simulator: Remote API (DeepSeek / OpenAI) via separate OpenAI client

The tau2 library calls ``tau2.utils.llm_utils.generate(model, messages, ...)``
internally for both agent and user turns.  We cannot modify tau2 source, so we
monkey-patch that function and route based on model sentinel strings:
  ``llm_agent="agent"``  → Target inference
  ``llm_user="user"``   → Remote user-simulator API

The patch function is process-global, while each run binds its own clients and
models through a context-local route. Already-imported tau2 modules holding the
original ``generate`` reference are updated by ``_patch_tau2_generate()``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterator, Optional, Sequence

from loguru import logger
from openai import OpenAI

from host.adapters import DownloadReport, RunResult
from host.config import Settings, load_settings

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

_AGENT_SENTINEL = "agent"
_USER_SENTINEL = "user"
_TAU2_REMOTE_LLM_DEFAULTS = {
    "gpt-4.1-2025-04-14",
    "claude-opus-4-5",
}


@dataclass
class TauBenchRunArgs:
    model_name: str | None = None
    domains: list[str] | None = None
    limit: int | None = None
    task_ids: Sequence[str] | None = None
    run_id: str | None = None


@dataclass
class TauBenchTaskResult:
    task_id: str
    domain: str
    reward: float | None
    success: bool | None
    error: str | None = None
    trial: int = 0


def _pass_hat_k(rewards: list[float], *, threshold: float = 1.0) -> float:
    """Return 1.0 if any trial achieved reward >= threshold, else 0.0."""
    return 1.0 if any(r >= threshold for r in rewards) else 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TauBenchRunner:
    """Host-side Tau-Bench runner pointed at Target + remote user-sim LLM."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

    def preflight(self) -> bool:
        from host.config.submodules import TAU2_BENCH_DIR, tau2_data_dir

        n_errors = 0

        # 1. Submodule present
        submodule_ok = TAU2_BENCH_DIR.is_dir() and (TAU2_BENCH_DIR / "src" / "tau2").is_dir()
        if submodule_ok:
            logger.info("  tau2-bench submodule: OK")
        else:
            logger.error(
                "  tau2-bench submodule: NOT FOUND at {}  "
                "hint: git submodule update --init submodules/tau2-bench",
                TAU2_BENCH_DIR,
            )
            n_errors += 1

        # 2. tau2 importable
        tau2_ok = _check_tau2_importable()
        if tau2_ok:
            logger.info("  tau2: importable")
        else:
            logger.error("  tau2: not importable  hint: uv sync --extra tau  (or pip install tau2)")
            n_errors += 1

        # 3. Data directory present
        if submodule_ok:
            data_dir = tau2_data_dir()
            data_ok = data_dir.is_dir() and any(data_dir.iterdir())
            if data_ok:
                logger.info("  tau2-bench data: OK")
            else:
                logger.error(
                    "  tau2-bench data: missing or empty at {}  hint: uv run tau2 check-data",
                    data_dir,
                )
                n_errors += 1

        # 4. User API key
        key_ok = bool(self.settings.resolve_tau_user_api_key())
        if key_ok:
            logger.info("  user API key: configured")
        else:
            logger.error(
                "  user API key: not configured  hint: set TAU_USER_API_KEY or DEEPSEEK_API_KEY"
            )
            n_errors += 1

        if n_errors:
            logger.error("Tau-Bench preflight: not ready ({} error(s))", n_errors)
        else:
            logger.info("Tau-Bench preflight: ready")

        return n_errors == 0

    def run(self, args: TauBenchRunArgs, *, output_dir: Path | None = None) -> RunResult:
        if output_dir is not None and args.run_id is None:
            raise ValueError("run_id is required when output_dir is externally managed")

        cfg = self.settings.tau_bench
        domains = args.domains or cfg.domains
        model_name = args.model_name or self.settings.target.model
        run_id = args.run_id or _default_run_id()
        output_dir = output_dir or self.settings.run_dir("tau-bench", run_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        limit = args.limit if args.limit is not None else cfg.num_tasks

        # Install the tau2 monkey patch once and create an isolated route for this run.
        llm_route = _ensure_llm_patch(self.settings, agent_model=model_name)

        from tau2.data_model.simulation import TextRunConfig
        from tau2.runner.batch import run_single_task
        from tau2.runner.helpers import get_tasks

        all_tasks: list[TauBenchTaskResult] = []

        logger.info(
            "Tau-Bench run_id={} domains={} model={}",
            run_id,
            domains,
            model_name,
        )
        logger.info(
            "  agent_sentinel={}  user_model={}  output_dir={}",
            _AGENT_SENTINEL,
            cfg.user_model,
            output_dir,
        )

        with _use_llm_route(llm_route):
            for domain in domains:
                tasks = get_tasks(
                    task_set_name=domain,
                    task_split_name=cfg.split,
                    task_ids=list(args.task_ids) if args.task_ids else None,
                    num_tasks=limit,
                )
                if not tasks:
                    logger.warning("No tasks found for domain={} split={}", domain, cfg.split)
                    continue

                logger.info("  domain={}  n_tasks={}  repeats={}", domain, len(tasks), cfg.repeats)

                run_config = TextRunConfig(
                    domain=domain,
                    llm_agent=_AGENT_SENTINEL,
                    llm_user=_USER_SENTINEL,
                    llm_args_agent={
                        "temperature": cfg.agent_temperature,
                        "max_tokens": cfg.agent_max_tokens,
                    },
                    llm_args_user={
                        "temperature": cfg.user_temperature,
                        "max_tokens": cfg.user_max_tokens,
                    },
                    max_steps=cfg.max_steps,
                )

                # Build work list: (task, trial_index)
                work: list[tuple[Any, int]] = []
                for task in tasks:
                    for trial in range(cfg.repeats):
                        work.append((task, trial))

                def _run_one(task_trial: tuple[Any, int]) -> TauBenchTaskResult:
                    task, trial = task_trial
                    try:
                        with _use_llm_route(llm_route):
                            sim = run_single_task(
                                run_config,
                                task,
                                seed=trial,
                                save_dir=output_dir / domain,
                                verbose_logs=self.settings.tau_bench.verbose_logs,
                            )
                        reward = sim.reward_info.reward if sim.reward_info else None
                        success = reward is not None and reward >= 1.0
                        return TauBenchTaskResult(
                            task_id=task.id,
                            domain=domain,
                            reward=reward,
                            success=success,
                            trial=trial,
                        )
                    except Exception as exc:
                        logger.error("task {} trial={} failed: {}", task.id, trial, exc)
                        return TauBenchTaskResult(
                            task_id=task.id,
                            domain=domain,
                            reward=None,
                            success=False,
                            error=str(exc),
                            trial=trial,
                        )

                results: list[TauBenchTaskResult] = []
                if cfg.n_concurrent <= 1:
                    for item in work:
                        results.append(_run_one(item))
                else:
                    with ThreadPoolExecutor(max_workers=cfg.n_concurrent) as executor:
                        futs = {executor.submit(_run_one, item): item for item in work}
                        for fut in as_completed(futs):
                            results.append(fut.result())

                for r in results:
                    all_tasks.append(r)

        # Aggregate by task (pass@k across repeats)
        task_rewards: dict[str, list[float]] = {}
        for t in all_tasks:
            key = f"{t.domain}:{t.task_id}"
            task_rewards.setdefault(key, [])
            if t.reward is not None:
                task_rewards[key].append(t.reward)

        n_total = len(task_rewards)
        n_success = sum(1 for rews in task_rewards.values() if _pass_hat_k(rews) >= 1.0)
        success = n_total > 0
        value = n_success / n_total if n_total else None
        all_rewards = [r for rews in task_rewards.values() for r in rews]
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        parse_error = None if success else "no tau-bench tasks selected"

        logger.info(
            "Tau-Bench complete: pass_hat={:.2f}% ({}/{})  avg_reward={:.3f}",
            (value or 0.0) * 100,
            n_success,
            n_total,
            avg_reward,
        )

        # Write aggregate results.json inside the run directory
        _write_aggregate_results(
            output_dir=output_dir,
            run_id=run_id,
            model_name=model_name,
            domains=domains,
            pass_hat=value,
            avg_reward=avg_reward,
            tasks=all_tasks,
            n_total=n_total,
            n_success=n_success,
            success=success,
            parse_error=parse_error,
        )

        return RunResult(
            success=success,
            run_id=run_id,
            model=model_name,
            metric="pass_hat" if value is not None else None,
            value=value,
            n_total=n_total,
            n_passed=n_success,
            summary={
                "domains": domains,
                "avg_reward": avg_reward,
                "n_success": n_success,
                "tasks": [
                    {
                        "task_id": t.task_id,
                        "domain": t.domain,
                        "reward": t.reward,
                        "success": t.success,
                        "error": t.error,
                        "trial": t.trial,
                    }
                    for t in all_tasks
                ],
                "output_dir": str(output_dir) if output_dir else None,
                "parse_error": parse_error,
            },
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
        """Download or verify tau-bench data (initialise the git submodule).

        tau2-bench has no HuggingFace datasets; data lives in the submodule.
        """
        from host.config.submodules import TAU2_BENCH_DIR, tau2_data_dir

        report = DownloadReport(backend="tau-bench")
        for item in items:
            try:
                data_dir = tau2_data_dir()

                if check_only:
                    if not (
                        TAU2_BENCH_DIR.is_dir() and data_dir.is_dir() and any(data_dir.iterdir())
                    ):
                        raise FileNotFoundError(
                            f"tau2-bench submodule or data not present at {TAU2_BENCH_DIR}. "
                            "Run: git submodule update --init submodules/tau2-bench"
                        )
                else:
                    proc = subprocess.run(
                        ["git", "submodule", "update", "--init", "submodules/tau2-bench"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if proc.returncode != 0:
                        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
                        raise RuntimeError(f"git submodule update failed: {tail}")

                    if not (data_dir.is_dir() and any(data_dir.iterdir())):
                        raise FileNotFoundError(
                            f"data/ not populated after submodule init at {data_dir}"
                        )

                report.succeeded.append(item)
                verb = "verified" if check_only else "downloaded"
                print(f"{verb}: {item}")
            except Exception as exc:
                verb = "verify" if check_only else "download"
                report.failed[item] = str(exc)
                print(f"failed to {verb} {item}: {exc}", file=sys.stderr)

        return report


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S")


def _write_aggregate_results(
    *,
    output_dir: Path,
    run_id: str,
    model_name: str,
    domains: list[str],
    pass_hat: float | None,
    avg_reward: float,
    tasks: list[TauBenchTaskResult],
    n_total: int,
    n_success: int,
    success: bool,
    parse_error: str | None,
) -> None:
    """Write aggregate results JSON to the run directory."""
    results_path = output_dir / "results.json"
    payload = {
        "run_id": run_id,
        "model": model_name,
        "domains": domains,
        "success": success,
        "pass_hat": pass_hat,
        "avg_reward": avg_reward,
        "n_total": n_total,
        "n_success": n_success,
        "parse_error": parse_error,
        "tasks": [
            {
                "task_id": t.task_id,
                "domain": t.domain,
                "reward": t.reward,
                "success": t.success,
                "error": t.error,
                "trial": t.trial,
            }
            for t in tasks
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# tau2 import check
# ---------------------------------------------------------------------------


def _check_tau2_importable() -> bool:
    try:
        import tau2  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Private: LLM patch (process-level singleton)
# ---------------------------------------------------------------------------

_patch_applied: bool = False
_patch_lock = Lock()


@dataclass(frozen=True)
class _LLMRoute:
    agent_client: OpenAI
    user_client: OpenAI
    agent_model: str
    user_model: str


_llm_route: ContextVar[_LLMRoute | None] = ContextVar("tau_bench_llm_route", default=None)


@contextmanager
def _use_llm_route(route: _LLMRoute) -> Iterator[None]:
    token = _llm_route.set(route)
    try:
        yield
    finally:
        _llm_route.reset(token)


def _ensure_llm_patch(settings: Settings, *, agent_model: str | None = None) -> _LLMRoute:
    """Install the process-wide patch and return an isolated route for this run."""
    global _patch_applied

    route = _LLMRoute(
        *_build_model_clients(
            settings,
            agent_model=agent_model,
        )
    )
    with _patch_lock:
        if not _patch_applied:
            _patch_tau2_generate()
            _patch_applied = True
    logger.info(
        "tau2 LLM patch ready — agent→{} model={}  user→{} model={}",
        settings.target.inference_base_url,
        route.agent_model,
        settings.tau_bench.user_model_url,
        route.user_model,
    )
    return route


def _build_model_clients(
    settings: Settings,
    *,
    agent_model: str | None = None,
) -> tuple[OpenAI, OpenAI, str, str]:
    """Construct OpenAI clients for agent (Target) and user (remote)."""
    agent_client = OpenAI(
        base_url=settings.target.inference_base_url.rstrip("/"),
        api_key=settings.target.resolved_inference_api_key(),
    )
    resolved_agent_model = agent_model or settings.target.model

    user_key = settings.resolve_tau_user_api_key() or "EMPTY"
    user_client = OpenAI(
        base_url=settings.tau_bench.user_model_url.rstrip("/"),
        api_key=user_key,
    )
    user_model = settings.tau_bench.user_model

    return agent_client, user_client, resolved_agent_model, user_model


def _patched_generate(
    model: str,
    messages: list,
    tools: Optional[list] = None,
    tool_choice: Optional[str] = None,
    call_name: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Drop-in replacement for ``tau2.utils.llm_utils.generate``.

    Routes calls to Target (agent) or remote user-simulator (user) based on
    sentinel model strings, bypassing litellm entirely for those sentinels.
    Unknown model strings fall through to the original litellm-backed generate.
    """
    # Late import — tau2 must already be importable when this is called
    from tau2.data_model.message import AssistantMessage, ToolCall
    from tau2.utils.llm_utils import to_litellm_messages

    if model not in (_AGENT_SENTINEL, _USER_SENTINEL, *_TAU2_REMOTE_LLM_DEFAULTS):
        # Not a sentinel — call original litellm generate
        import tau2.utils.llm_utils as _llm

        return _llm._original_generate(
            model, messages, tools=tools, tool_choice=tool_choice, call_name=call_name, **kwargs
        )  # noqa: SLF001

    route = _llm_route.get()
    if route is None:
        raise RuntimeError("tau2 LLM route not bound to the current task")
    client = route.agent_client if model == _AGENT_SENTINEL else route.user_client
    resolved_model = route.agent_model if model == _AGENT_SENTINEL else route.user_model

    oai_messages = to_litellm_messages(messages)

    # Build tools schema — openai_schema is already {"type": "function", "function": {...}}
    tools_schema: list[dict] | None = None
    if tools:
        tools_schema = [t.openai_schema for t in tools]
        if tool_choice is None:
            tool_choice = "auto"

    # Build request kwargs — strip tau2-internal keys
    req_kwargs: dict[str, Any] = {}
    temperature = kwargs.get("temperature")
    max_tokens = kwargs.get("max_tokens")
    if temperature is not None:
        req_kwargs["temperature"] = temperature
    if max_tokens is not None:
        req_kwargs["max_tokens"] = max_tokens

    # DeepSeek: disable thinking mode for user simulator to avoid JSON issues
    if model != _AGENT_SENTINEL:
        req_kwargs.setdefault("extra_body", {})
        req_kwargs["extra_body"]["thinking"] = {"type": "disabled"}

    response = client.chat.completions.create(
        model=resolved_model,
        messages=oai_messages,
        tools=tools_schema or None,
        tool_choice=tool_choice if tools_schema else None,
        **req_kwargs,
    )

    choice = response.choices[0]
    content = choice.message.content
    raw_tool_calls = choice.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=json.loads(tc.function.arguments),
        )
        for tc in raw_tool_calls
    ] or None

    return AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=None,
        usage=None,
    )


def _patch_tau2_generate() -> None:
    """Patch tau2.utils.llm_utils.generate and fan-out to importee namespaces."""
    import tau2.utils.llm_utils as _llm

    # Save original so non-sentinel calls can fall through
    if not hasattr(_llm, "_original_generate"):
        _llm._original_generate = _llm.generate  # type: ignore[attr-defined]
    original_generate = _llm._original_generate  # type: ignore[attr-defined]

    _llm.generate = _patched_generate  # type: ignore[attr-defined]

    # Update modules that imported the original function before the patch was installed.
    for module_name, mod in list(sys.modules.items()):
        if module_name.startswith("tau2.") and getattr(mod, "generate", None) is original_generate:
            mod.generate = _patched_generate  # type: ignore[attr-defined]
