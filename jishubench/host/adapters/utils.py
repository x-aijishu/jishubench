"""Adapter 共享工具函数。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def check_docker_available() -> bool:
    """Test Docker socket is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def check_harness_cli(harness: str) -> Path | None:
    """Find a harness CLI on PATH."""
    cli = "tb" if harness == "tb" else "harbor"
    found = shutil.which(cli)
    return Path(found) if found else None
