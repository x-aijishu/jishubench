"""Tests: CLI 解析 + 命令分发逻辑（不测 argparse 默认值）。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from host.cli.commands import HANDLERS, _cmd_benchmarks, _cmd_download, _cmd_run, _cmd_target_unload
from host.cli.parser import build_parser
from host.client.llama_cpp_client import ModelInfo

# ---------------------------------------------------------------------------
# 命令注册一致性
# ---------------------------------------------------------------------------


def test_cli_subcommands_registered():
    """所有 parser 子命令在 HANDLERS 中都有对应处理器。"""
    parser = build_parser()
    tested = set()
    # Simple commands without required positional args
    for command in ("preflight", "benchmarks"):
        args = parser.parse_args([command])
        assert args.command == command
        tested.add(command)
    # target requires a subcommand
    cmd_t = parser.parse_args(["target", "status"])
    assert cmd_t.command == "target"
    tested.add("target")
    cmd_cfg = parser.parse_args(["config", "show"])
    assert cmd_cfg.command == "config"
    tested.add("config")
    # run and download require BENCHMARK positional arg
    cmd_run = parser.parse_args(["run", "dummy"])
    assert cmd_run.command == "run"
    tested.add("run")
    cmd_dl = parser.parse_args(["download", "dummy"])
    assert cmd_dl.command == "download"
    tested.add("download")
    assert HANDLERS.keys() == tested


def test_main_handlers_key_match_parser():
    """HANDLERS dict 的 key 必须与 parser 注册的 command 一一对应。"""
    parser = build_parser()
    sub = parser._subparsers
    for action in sub._actions:
        if hasattr(action, "choices") and action.choices:
            parser_commands = set(action.choices.keys())
            assert HANDLERS.keys() == parser_commands
            return
    raise AssertionError("no choices found in parser")


# ---------------------------------------------------------------------------
# download 子命令
# ---------------------------------------------------------------------------


def test_cli_config_use_subcommand_registered():
    parser = build_parser()
    args = parser.parse_args(["config", "use", "macstudio"])
    assert args.command == "config"
    assert args.config_command == "use"
    assert args.profile == "macstudio"


@pytest.mark.parametrize(
    "argv",
    [
        ["preflight", "--config", "user-local"],
        ["benchmarks", "--config", "user-local"],
        ["--config", "user-local", "run", "mmmu_val"],
        ["run", "--config", "user-local", "mmmu_val"],
        ["run", "mmmu_val", "--config", "user-local"],
        ["run", "mmmu_val", "--config", "user-local", "--model", "Qwen"],
        ["run", "mmmu_val", "--config", "user-local", "realworldqa"],
        ["run", "mmmu_val", "--model", "Qwen", "--config", "user-local"],
        ["download", "--config", "user-local", "mmmu_val"],
        ["download", "mmmu_val", "--config", "user-local"],
        ["--config", "user-local", "target", "status"],
        ["target", "--config", "user-local", "status"],
        ["target", "status", "--config", "user-local"],
        ["target", "status", "--config", "user-local", "--model", "Qwen"],
        ["target", "models", "--config", "user-local"],
        ["target", "load", "--config", "user-local"],
        ["target", "unload", "--config", "user-local"],
        ["target", "status", "--config=user-local"],
    ],
)
def test_cli_config_can_be_passed_within_leaf_subcommand(argv):
    parser = build_parser()
    args = parser.parse_args(argv)
    assert args.config.name == "user-local.yaml"


@pytest.mark.parametrize(
    "argv",
    [
        ["--config", "macstudio", "target", "status", "--config", "user-local"],
        ["target", "status", "--config", "user-local", "extra"],
        ["target", "status", "--config"],
        ["config", "list", "--config", "user-local"],
        ["config", "use", "--config", "user-local", "macstudio"],
    ],
)
def test_cli_config_is_only_accepted_on_commands_that_use_settings(argv):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_cli_download_subcommand_registered():
    """download 子命令的 flag 解析。"""
    parser = build_parser()
    args = parser.parse_args(["download", "mmmu_val", "realworldqa"])
    assert args.command == "download"
    assert args.benchmarks == ["mmmu_val", "realworldqa"]
    assert args.check is False
    assert args.force is False

    args_explicit = parser.parse_args(["download", "mmmu_val", "--backend", "lmms-eval", "--check"])
    assert args_explicit.backend == "lmms-eval"
    assert args_explicit.check is True


@pytest.mark.parametrize("backend", ["tau-bench", "terminal-bench", "claw-eval"])
def test_cli_download_agent_backend_objects_parse(backend):
    parser = build_parser()
    args = parser.parse_args(["download", backend, "--check"])

    assert args.command == "download"
    assert args.backend is None
    assert args.benchmarks == [backend]
    assert args.check is True


def test_cli_download_verbose_rejected():
    """download 子命令不再暴露 verbose flag；失败默认打印 traceback。"""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["download", "mmmu_val", "-v"])


def test_cli_download_cache_paths_parse_as_directories(tmp_path):
    """download cache flags are filesystem paths, not device profile references."""
    parser = build_parser()
    hub_cache = tmp_path / "hub-cache"
    datasets_cache = tmp_path / "datasets-cache"

    args = parser.parse_args(
        [
            "download",
            "--hub-cache",
            str(hub_cache),
            "--datasets-cache",
            str(datasets_cache),
            "mmmu_val",
        ]
    )

    assert args.hub_cache == hub_cache
    assert args.datasets_cache == datasets_cache


# ---------------------------------------------------------------------------
# run 子命令 flag 解析
# ---------------------------------------------------------------------------


def test_run_flags_after_benchmark_parse_directly():
    """放在 benchmark 后面的 --limit 10 应被 argparse 正常解析。"""
    parser = build_parser()
    args = parser.parse_args(["run", "realworldqa", "--limit", "10"])
    assert args.limit == 10
    assert args.benchmarks == ["realworldqa"]


def test_run_flags_no_optional_args():
    """没有额外 flags 时维持原样。"""
    parser = build_parser()
    args = parser.parse_args(["run", "realworldqa"])
    assert args.limit is None
    assert args.benchmarks == ["realworldqa"]


def test_run_verbose_rejected():
    """run 子命令不再暴露未使用的 verbose flag。"""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "realworldqa", "-v"])


def test_run_unknown_lmms_eval_extra_args_rejected():
    """lmms-eval 的泛用 extra args 已删除，未知参数应直接失败。"""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "realworldqa", "--unknown-flag"])


@pytest.mark.parametrize("backend", ["tau-bench", "terminal-bench"])
def test_run_agent_backends_reject_extra_benchmarks(backend, capsys):
    args = argparse.Namespace(benchmarks=[backend, "mmmu_val"])

    assert _cmd_run(args) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert f"{backend} accepts no additional benchmark arguments: mmmu_val" in err
    assert "Use --task-id to select specific agent benchmark tasks." in err


# ---------------------------------------------------------------------------
# _cmd_download 错误处理
# ---------------------------------------------------------------------------


def test_ensure_lmms_eval_rejects_empty_submodule_directory(tmp_path, monkeypatch):
    from host.config.submodules import ensure_lmms_eval, submodule_status

    empty_submodule = tmp_path / "lmms-eval"
    empty_submodule.mkdir()
    monkeypatch.setattr("host.config.submodules.LMMS_EVAL_DIR", empty_submodule)

    assert submodule_status()["lmms-eval"] is False
    with pytest.raises(FileNotFoundError, match="submodules/lmms-eval"):
        ensure_lmms_eval()


def test_cmd_download_unknown_backend(monkeypatch):
    monkeypatch.setattr(
        "host.cli.commands.load_settings", lambda config: __import__("host.config").Settings()
    )

    args = argparse.Namespace(
        config=Path("configs/base.yaml"),
        backend="unknown",
        benchmarks=["mmmu_val"],
        hub_cache=None,
        datasets_cache=None,
        check=False,
        force=False,
    )
    assert _cmd_download(args) == 1


@pytest.mark.parametrize("backend", ["tau-bench", "terminal-bench"])
def test_cmd_download_infers_agent_backend_objects(monkeypatch, tmp_path, backend):
    from host.adapters import DownloadReport
    from host.cli import commands

    captured = {}

    def fake_download(items, settings, *, check_only, force):
        captured.update({"items": items, "check_only": check_only, "force": force})
        return DownloadReport(backend=backend, succeeded=list(items))

    monkeypatch.setattr("host.cli.commands.load_settings", lambda config: object())
    monkeypatch.setattr("host.cli.commands.setup_console_logging", lambda settings: None)
    monkeypatch.setattr("host.cli.commands.apply_download_cache_overrides", lambda s, **kw: s)
    monkeypatch.setattr(commands._BACKEND_RUNNERS[backend], "download_tasks", fake_download)
    args = argparse.Namespace(
        config=tmp_path / "config.yaml",
        backend=None,
        benchmarks=[backend],
        hub_cache=None,
        datasets_cache=None,
        check=True,
        force=False,
    )

    assert _cmd_download(args) == 0
    assert captured == {"items": [backend], "check_only": True, "force": False}


# ---------------------------------------------------------------------------
# _cmd_benchmarks 错误处理
# ---------------------------------------------------------------------------


def test_cmd_benchmarks_fails_without_submodule(monkeypatch, capsys):
    from host.cli import commands

    def missing_lmms_eval():
        raise FileNotFoundError(
            "lmms-eval submodule not found. Run: git submodule update --init submodules/lmms-eval"
        )

    monkeypatch.setitem(commands._BACKEND_SUBMODULE_CHECKS, "lmms-eval", missing_lmms_eval)

    args = argparse.Namespace(
        config=Path("configs/base.yaml"),
        kind="subtasks",
        search=None,
    )
    assert _cmd_benchmarks(args) == 1
    assert "git submodule update --init submodules/lmms-eval" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# target unload 子命令 — 行为测试
# ---------------------------------------------------------------------------


def _make_unload_args(*, model: str | None) -> argparse.Namespace:
    return argparse.Namespace(config=Path("configs/base.yaml"), model=model)


def _install_fake_client(monkeypatch, *, models, unload_result=True):
    """Replace load_settings + LlamaCppClient with a fake client."""
    settings = object()  # opaque to the handler

    class _FakeClient:
        def __init__(self, _settings):
            self._models = list(models)
            self.unloaded: list[str] = []

        def list_models(self, *, reload: bool = False):
            return list(self._models)

        def resolve_model_id(self, name):
            return name

        def unload_model(self, name):
            self.unloaded.append(name)
            return unload_result

    fake = _FakeClient(settings)

    monkeypatch.setattr("host.cli.commands.load_settings", lambda config: settings)
    monkeypatch.setattr("host.cli.commands.LlamaCppClient", lambda _settings: fake)
    return fake


def test_unload_explicit_model(monkeypatch, capsys):
    """Explicit --model bypasses live-state inference."""
    models = [ModelInfo(id="a", status="loaded")]
    fake = _install_fake_client(monkeypatch, models=models)

    rc = _cmd_target_unload(_make_unload_args(model="explicit-model"))
    out, err = capsys.readouterr()

    assert rc == 0
    assert "unloading explicit-model ..." in out
    assert "unloaded: explicit-model" in out
    assert fake.unloaded == ["explicit-model"]


def test_unload_infers_single_loaded(monkeypatch, capsys):
    """No --model + exactly one loaded model -> unload it."""
    models = [
        ModelInfo(id="Qwen3.6-35B", status="loaded"),
        ModelInfo(id="Qwen3.6-27B", status="unloaded"),
    ]
    fake = _install_fake_client(monkeypatch, models=models)

    rc = _cmd_target_unload(_make_unload_args(model=None))
    out, err = capsys.readouterr()

    assert rc == 0
    assert "unloading Qwen3.6-35B ..." in out
    assert "unloaded: Qwen3.6-35B" in out
    assert fake.unloaded == ["Qwen3.6-35B"]


def test_unload_no_loaded_model(monkeypatch, capsys):
    """No --model + zero loaded models -> error + suggestion."""
    models = [ModelInfo(id="a", status="unloaded"), ModelInfo(id="b", status="sleeping")]
    fake = _install_fake_client(monkeypatch, models=models)

    rc = _cmd_target_unload(_make_unload_args(model=None))
    out, err = capsys.readouterr()

    assert rc == 1
    assert "no model is currently loaded" in err
    assert "target load --model" in err
    assert fake.unloaded == []


def test_unload_multiple_loaded_is_ambiguous(monkeypatch, capsys):
    """No --model + multiple loaded models -> error listing them."""
    models = [
        ModelInfo(id="Qwen3.6-35B", status="loaded"),
        ModelInfo(id="Qwen3.6-27B", status="loaded"),
    ]
    fake = _install_fake_client(monkeypatch, models=models)

    rc = _cmd_target_unload(_make_unload_args(model=None))
    out, err = capsys.readouterr()

    assert rc == 1
    assert "multiple models are loaded" in err
    assert "Qwen3.6-27B" in err  # sorted first
    assert "Qwen3.6-35B" in err
    assert "target unload --model" in err
    assert fake.unloaded == []


def test_unload_list_models_failure(monkeypatch, capsys):
    """If listing models raises, fall back to an explicit suggestion."""
    settings = object()

    class _BrokenClient:
        def __init__(self, _settings):
            pass

        def list_models(self, *, reload: bool = False):
            raise RuntimeError("router unreachable")

        def resolve_model_id(self, name):
            return name

        def unload_model(self, name):
            raise AssertionError("should not be called")

    monkeypatch.setattr("host.cli.commands.load_settings", lambda config: settings)
    monkeypatch.setattr(
        "host.cli.commands.LlamaCppClient",
        lambda _settings: _BrokenClient(settings),
    )

    rc = _cmd_target_unload(_make_unload_args(model=None))
    out, err = capsys.readouterr()

    assert rc == 1
    assert "failed to list models" in err
    assert "router unreachable" in err
    assert "target unload --model" in err


def test_unload_ignores_mmproj_entries(monkeypatch, capsys):
    """mmproj projector entries must not be treated as loadable models."""
    models = [
        ModelInfo(id="Qwen3.6-35B", status="loaded"),
        ModelInfo(id="Qwen3.6-35B-mmproj-F16.gguf", status="loaded"),
    ]
    fake = _install_fake_client(monkeypatch, models=models)

    rc = _cmd_target_unload(_make_unload_args(model=None))
    out, err = capsys.readouterr()

    assert rc == 0
    assert fake.unloaded == ["Qwen3.6-35B"]


def test_unload_failure_returns_nonzero(monkeypatch, capsys):
    """Inferred model but router unload fails -> rc=1, stderr message."""
    models = [ModelInfo(id="Qwen3.6-35B", status="loaded")]
    fake = _install_fake_client(monkeypatch, models=models, unload_result=False)

    rc = _cmd_target_unload(_make_unload_args(model=None))
    out, err = capsys.readouterr()

    assert rc == 1
    assert "failed to unload Qwen3.6-35B" in err
    assert fake.unloaded == ["Qwen3.6-35B"]
