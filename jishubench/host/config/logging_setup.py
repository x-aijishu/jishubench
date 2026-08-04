"""Application logging for the Host CLI."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from host.config.settings import Settings

_CONFIGURED = False
_RUN_LOG_PATH: Path | None = None  # set by setup_run_logging(), read by setup_logging()

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}"
)

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "openai",
    "LiteLLM",
    "numexpr",
    "lmms_eval",
    "datasets",
    "transformers",
    "huggingface_hub",
    "filelock",
    "torch",
)


def _is_noisy_record(record: logging.LogRecord) -> bool:
    name = record.name
    return any(name == noisy or name.startswith(f"{noisy}.") for noisy in _NOISY_LOGGERS)


class _InterceptHandler(logging.Handler):
    """Redirect stdlib logging records into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        if _is_noisy_record(record) and record.levelno < logging.WARNING:
            return
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno  # type: ignore[assignment]
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def log_filename_for_run(at: datetime | None = None) -> str:
    """Return a run-scoped log file name: ``YYYYMMDD-HHMMSS.log``."""
    stamp = (at or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}.log"


def resolve_log_path(save_dir: Path | None, *, at: datetime | None = None) -> Path | None:
    if save_dir is None:
        return None
    return save_dir / log_filename_for_run(at)


def setup_console_logging(settings: Settings) -> None:
    """Configure loguru stderr only for non-run subcommands (no log file)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = settings.logging
    level = cfg.level.upper()

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=_CONSOLE_FORMAT,
        colorize=True,
    )

    suppress_noisy_loggers()
    _CONFIGURED = True


def setup_logging(
    settings: Settings,
    *,
    log_at: datetime | None = None,
    force: bool = False,
) -> None:
    """Configure loguru for ``jishubench run`` (stderr console + optional file sink).

    When *force* is true the ``_CONFIGURED`` guard is bypassed — this is needed
    after lmms-eval's import-time ``logger.remove()`` wipes all sinks.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    cfg = settings.logging
    level = cfg.level.upper()

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=_CONSOLE_FORMAT,
        colorize=True,
    )

    log_path = resolve_log_path(cfg.resolved_save_dir(), at=log_at)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} {level} [{name}] {message}",
            rotation=cfg.max_bytes,
            retention=cfg.backup_count,
            encoding="utf-8",
        )

    # Per-run log file (if set via setup_run_logging)
    if _RUN_LOG_PATH is not None:
        logger.add(
            str(_RUN_LOG_PATH),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} {level} [{name}] {message}",
            rotation=cfg.max_bytes,
            retention=cfg.backup_count,
            encoding="utf-8",
        )

    suppress_noisy_loggers()
    _CONFIGURED = True


def setup_run_logging(run_dir: Path, *, level: str = "INFO") -> Path:
    """Add a per-run loguru file sink at ``<run_dir>/logs/run.log``.

    The path is recorded globally so that subsequent ``setup_logging()`` calls
    (e.g. after lmms-eval import wipes sinks) also re-add this sink.
    Returns the log file path.
    """
    global _RUN_LOG_PATH
    log_path = run_dir / "logs" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level=level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} {level} [{name}] {message}",
        encoding="utf-8",
    )
    _RUN_LOG_PATH = log_path
    return log_path


def suppress_noisy_loggers() -> None:
    """Route stdlib logging from third-party libs through loguru and quieten noisy ones."""
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
