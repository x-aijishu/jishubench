"""Device profile discovery and config path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from host.config.settings import REPO_ROOT, _read_yaml_dict

DEVICES_DIR = REPO_ROOT / "configs" / "devices"
USER_CONFIG_DIR = Path.home() / ".jishubench"
ACTIVE_PROFILE_FILE = USER_CONFIG_DIR / "profile"
FALLBACK_PROFILE = DEVICES_DIR / "macstudio.yaml"
CONFIG_ENV_VAR = "JISHU_BENCH_CONFIG"


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    path: Path

    def summary(self) -> str:
        data = _read_yaml_dict(self.path)
        target = data.get("target") or {}
        url = target.get("inference_base_url", "?")
        model = target.get("model", "?")
        return f"{url}  model={model}"


def list_device_profiles() -> list[DeviceProfile]:
    if not DEVICES_DIR.is_dir():
        return []

    return [DeviceProfile(name=path.stem, path=path) for path in sorted(DEVICES_DIR.glob("*.yaml"))]


def _profile_path_for_name(name: str) -> Path | None:
    stem = name.removesuffix(".yaml")
    candidate = DEVICES_DIR / f"{stem}.yaml"
    if candidate.is_file():
        return candidate
    return None


def selected_profile_name() -> str | None:
    if not ACTIVE_PROFILE_FILE.is_file():
        return None
    name = ACTIVE_PROFILE_FILE.read_text(encoding="utf-8").strip()
    return name or None


def _selected_profile_path() -> Path | None:
    name = selected_profile_name()
    if name is None:
        return None
    path = _profile_path_for_name(name)
    if path is None:
        raise FileNotFoundError(
            f"Selected profile {name!r} in {ACTIVE_PROFILE_FILE} not found under {DEVICES_DIR}"
        )
    return path.resolve()


def resolve_config_path(value: str | Path | None = None) -> Path:
    """Resolve a CLI/config reference to an on-disk device profile."""
    if value is not None and str(value).strip():
        raw = Path(value).expanduser()
        if raw.is_file():
            return raw.resolve()

        lookup = raw.name if raw.suffix == ".yaml" else str(raw)
        named = _profile_path_for_name(lookup)
        if named is not None:
            return named.resolve()

        repo_relative = (REPO_ROOT / raw).resolve()
        if repo_relative.is_file():
            return repo_relative

        raise FileNotFoundError(f"Config not found: {value}")

    env = os.environ.get(CONFIG_ENV_VAR, "").strip()
    if env:
        return resolve_config_path(env)

    selected = _selected_profile_path()
    if selected is not None:
        return selected

    if FALLBACK_PROFILE.is_file():
        return FALLBACK_PROFILE.resolve()

    raise FileNotFoundError(
        "No device profile found. Run `jishubench config use <profile>` or create "
        f"a profile under {DEVICES_DIR}"
    )


def active_profile_name() -> str | None:
    name = selected_profile_name()
    if name is not None:
        if _profile_path_for_name(name) is not None:
            return name
        return None
    try:
        return resolve_config_path(None).stem
    except FileNotFoundError:
        return None


def activate_profile(name: str) -> Path:
    """Record a shared device profile as the personal default."""
    source = _profile_path_for_name(name)
    if source is None:
        known = ", ".join(p.name for p in list_device_profiles())
        raise ValueError(f"Unknown profile {name!r}; available: {known or '(none)'}")

    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_PROFILE_FILE.write_text(f"{source.stem}\n", encoding="utf-8")
    return source.resolve()
