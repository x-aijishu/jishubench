"""Upstream benchmark framework runners.

Public types:
- ``RunResult`` — unified return value for all runner ``run()`` methods.
- ``Runner`` — protocol that every adapter runner must satisfy.
- ``DownloadReport`` — result from dataset download / verify operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# Single-task / whole-run score: top-level metric + float value.
# Multi-task: value is {task_name: {"metric": str, "value": float}} and
# top-level metric is None (name lives with each score).
TaskScoreDict = dict[str, float | str]
MultiTaskValue = dict[str, TaskScoreDict]


@dataclass
class RunResult:
    """Adapter ``run()`` 的统一返回值。"""

    success: bool
    run_id: str
    model: str
    # Upstream headline metric name when ``value`` is a scalar; None for multi-task.
    metric: str | None
    # float=single score; multi-task={task: {"metric": name, "value": score}}.
    value: float | MultiTaskValue | None
    n_total: int
    n_passed: int
    summary: dict[str, Any]  # 每家独有数据（配置、版本等）
    raw: dict[str, Any] | None = None  # 原始返回（存档用）


@dataclass
class DownloadReport:
    """Result from a dataset download operation."""

    backend: str
    succeeded: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


class Runner(Protocol):
    """Protocol that every adapter runner must satisfy."""

    def __init__(self, settings: Any) -> None: ...
    def preflight(self) -> bool: ...
    def run(self, args: Any, *, output_dir: Path | None = None) -> RunResult: ...

    @classmethod
    def download_tasks(
        cls,
        items: list[str],
        settings: Any,
        *,
        check_only: bool = False,
        force: bool = False,
    ) -> DownloadReport: ...
