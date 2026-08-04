"""Tests: logging_setup — _CONFIGURED 守卫、文件写入。"""

from __future__ import annotations

import logging
from datetime import datetime

from host.config import Settings
from host.config.logging_setup import log_filename_for_run, setup_console_logging, setup_logging


def _is_configured() -> bool:
    import host.config.logging_setup as ls

    return ls._CONFIGURED


def _reset_configured():
    import host.config.logging_setup as ls

    ls._CONFIGURED = False


def test_logging_setup_writes_file(tmp_path):
    """验证 setup_logging 正确写入文件。"""
    _reset_configured()
    log_dir = tmp_path / "logs"
    stamp = datetime(2025, 6, 9, 14, 30, 22)
    settings = Settings(logging={"save_dir": log_dir, "level": "DEBUG"})
    setup_logging(settings, log_at=stamp)
    logging.getLogger("jishubench.test").info("hello from file")
    log_file = log_dir / log_filename_for_run(stamp)
    assert log_file.is_file()
    assert "hello from file" in log_file.read_text(encoding="utf-8")


def test_logging_setup_guard_blocks_duplicate_calls(tmp_path):
    """_CONFIGURED 守卫阻止第二次调用。"""
    _reset_configured()
    log_dir = tmp_path / "logs1"
    settings = Settings(logging={"save_dir": log_dir, "level": "INFO"})
    setup_logging(settings)
    assert _is_configured()

    # 第二次调用不同设置，应被守卫忽略
    log_dir2 = tmp_path / "logs2"
    settings2 = Settings(logging={"save_dir": log_dir2, "level": "DEBUG"})
    setup_logging(settings2)

    assert not log_dir2.exists()


def test_console_logging_writes_no_file(tmp_path):
    """非 run 子命令只配置 stderr，不写日志文件。"""
    _reset_configured()
    log_dir = tmp_path / "logs"
    settings = Settings(logging={"save_dir": log_dir, "level": "INFO"})
    setup_console_logging(settings)
    logging.getLogger("jishubench.test").info("console only")
    assert not log_dir.exists()


def test_console_logging_suppresses_noisy_debug(capsys):
    """非 run 子命令屏蔽第三方 DEBUG 日志。"""
    _reset_configured()
    settings = Settings(logging={"level": "INFO"})
    setup_console_logging(settings)
    logging.getLogger("lmms_eval.tasks").debug("hidden debug")
    logging.getLogger("lmms_eval.tasks").warning("visible warning")
    captured = capsys.readouterr()
    assert "hidden debug" not in captured.err
    assert "visible warning" in captured.err
