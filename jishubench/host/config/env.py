"""Environment helpers for isolating Judge credentials from Target inference."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from host.config.settings import Settings


@contextmanager
def _temporary_env(overrides: dict[str, str]) -> Iterator[None]:
    if not overrides:
        yield
        return

    previous: dict[str, str | None] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, previous_value in previous.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


@contextmanager
def judge_env(settings: Settings) -> Iterator[None]:
    """Apply judge OpenAI credentials only inside the context."""
    with _temporary_env(settings.judge_env_for_upstream()):
        yield


@contextmanager
def lmms_eval_env(settings: Settings) -> Iterator[None]:
    """Apply lmms-eval runtime env (datasets cache, judge) before imports."""
    with _temporary_env(settings.lmms_eval_env_for_upstream()):
        yield


@contextmanager
def tau_bench_env(settings: Settings) -> Iterator[None]:
    """Apply tau-bench fallback LLM credentials only inside the context."""
    with _temporary_env(settings.tau_bench_env_for_upstream()):
        yield


@contextmanager
def terminal_bench_env(settings: Settings) -> Iterator[None]:
    """Apply Terminal-Bench runtime environment inside the context."""
    with _temporary_env(settings.terminal_bench_env_for_subprocess()):
        yield


@contextmanager
def terminal_bench_2_env(settings: Settings) -> Iterator[None]:
    """Apply Terminal-Bench 2.0 (Harbor) runtime environment inside the context."""
    with _temporary_env(settings.terminal_bench_2_env_for_subprocess()):
        yield


@contextmanager
def claw_eval_env(settings: Settings) -> Iterator[None]:
    """Apply claw-eval Target/judge credentials only inside the context."""
    with _temporary_env(settings.claw_eval_env_for_upstream()):
        yield
