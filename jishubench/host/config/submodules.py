"""Locate and expose git submodule paths for benchmark backends.

Note: submodules are installed as Python packages via `uv sync --extra <name>`
([tool.uv.sources] in pyproject.toml). These helpers exist only for:

- Checking that the git submodule is cloned (YAML definitions, config files)
- Exposing the submodule root path for reference
"""

from __future__ import annotations

from pathlib import Path

from host.config.settings import REPO_ROOT

SUBMODULES_DIR = REPO_ROOT / "submodules"

LMMS_EVAL_DIR = SUBMODULES_DIR / "lmms-eval"
TAU2_BENCH_DIR = SUBMODULES_DIR / "tau2-bench"
CLAW_EVAL_DIR = SUBMODULES_DIR / "claw-eval"


def _is_initialized(root: Path, marker: Path) -> bool:
    return root.is_dir() and (root / marker).is_dir()


def _ensure_initialized(name: str, root: Path, marker: Path) -> Path:
    if not _is_initialized(root, marker):
        raise FileNotFoundError(
            f"{name} submodule not found at {root}. "
            f"Run: git submodule update --init submodules/{root.name}"
        )
    return root


def ensure_lmms_eval() -> Path:
    return _ensure_initialized("lmms-eval", LMMS_EVAL_DIR, Path("lmms_eval"))


def ensure_tau2_bench() -> Path:
    return _ensure_initialized("tau2-bench", TAU2_BENCH_DIR, Path("src/tau2"))


def ensure_claw_eval() -> Path:
    return _ensure_initialized("claw-eval", CLAW_EVAL_DIR, Path("src/claw_eval"))


def tau2_data_dir() -> Path:
    return TAU2_BENCH_DIR / "data"


def submodule_status() -> dict[str, bool]:
    return {
        "lmms-eval": _is_initialized(LMMS_EVAL_DIR, Path("lmms_eval")),
        "tau2-bench": _is_initialized(TAU2_BENCH_DIR, Path("src/tau2")),
        "claw-eval": _is_initialized(CLAW_EVAL_DIR, Path("src/claw_eval")),
    }
