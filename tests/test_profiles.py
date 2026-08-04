"""Tests: device profile discovery and config path resolution."""

from __future__ import annotations

import pytest
from host.config.profiles import (
    CONFIG_ENV_VAR,
    activate_profile,
    list_device_profiles,
    resolve_config_path,
    selected_profile_name,
)


def test_list_device_profiles_includes_repo_profiles():
    names = {profile.name for profile in list_device_profiles()}
    assert "macstudio" in names
    assert "user-local" in names


def test_resolve_config_path_by_profile_name():
    path = resolve_config_path("macstudio")
    assert path.name == "macstudio.yaml"


def test_resolve_config_path_uses_home_profile_pointer(tmp_path, monkeypatch):
    devices = tmp_path / "devices"
    devices.mkdir()
    user_local = devices / "user-local.yaml"
    user_local.write_text("target:\n  model: local\n", encoding="utf-8")
    (devices / "macstudio.yaml").write_text("target:\n  model: m1\n", encoding="utf-8")

    config_dir = tmp_path / "home" / ".jishubench"
    config_dir.mkdir(parents=True)
    profile_file = config_dir / "profile"
    profile_file.write_text("user-local\n", encoding="utf-8")

    monkeypatch.setattr("host.config.profiles.DEVICES_DIR", devices)
    monkeypatch.setattr("host.config.profiles.USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr("host.config.profiles.ACTIVE_PROFILE_FILE", profile_file)
    monkeypatch.setattr("host.config.profiles.FALLBACK_PROFILE", devices / "macstudio.yaml")
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

    assert resolve_config_path(None) == user_local.resolve()
    assert selected_profile_name() == "user-local"


def test_resolve_config_path_env_override(tmp_path, monkeypatch):
    devices = tmp_path / "devices"
    devices.mkdir()
    profile = devices / "user-local.yaml"
    profile.write_text("target:\n  model: env\n", encoding="utf-8")

    config_dir = tmp_path / "home" / ".jishubench"
    monkeypatch.setattr("host.config.profiles.DEVICES_DIR", devices)
    monkeypatch.setattr("host.config.profiles.USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr("host.config.profiles.ACTIVE_PROFILE_FILE", config_dir / "profile")
    monkeypatch.setattr("host.config.profiles.FALLBACK_PROFILE", devices / "missing.yaml")
    monkeypatch.setenv(CONFIG_ENV_VAR, "user-local")

    assert resolve_config_path(None) == profile.resolve()


def test_activate_profile_writes_pointer_only(tmp_path, monkeypatch):
    devices = tmp_path / "devices"
    devices.mkdir()
    macstudio = devices / "macstudio.yaml"
    macstudio.write_text("target:\n  model: m1\n", encoding="utf-8")

    config_dir = tmp_path / "home" / ".jishubench"
    profile_file = config_dir / "profile"

    monkeypatch.setattr("host.config.profiles.DEVICES_DIR", devices)
    monkeypatch.setattr("host.config.profiles.USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr("host.config.profiles.ACTIVE_PROFILE_FILE", profile_file)

    path = activate_profile("macstudio")
    assert path == macstudio.resolve()
    assert profile_file.read_text(encoding="utf-8") == "macstudio\n"
    assert not (config_dir / "local.yaml").exists()


def test_resolve_config_path_missing_raises(tmp_path, monkeypatch):
    devices = tmp_path / "devices"
    devices.mkdir()
    config_dir = tmp_path / "home" / ".jishubench"
    monkeypatch.setattr("host.config.profiles.DEVICES_DIR", devices)
    monkeypatch.setattr("host.config.profiles.USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr("host.config.profiles.ACTIVE_PROFILE_FILE", config_dir / "profile")
    monkeypatch.setattr("host.config.profiles.FALLBACK_PROFILE", devices / "missing.yaml")
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)

    with pytest.raises(FileNotFoundError):
        resolve_config_path("unknown-profile")
