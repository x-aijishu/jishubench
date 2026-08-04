"""Host runtime configuration and environment helpers."""

from host.config.env import (
    claw_eval_env,
    judge_env,
    lmms_eval_env,
    tau_bench_env,
    terminal_bench_2_env,
    terminal_bench_env,
)
from host.config.logging_setup import (
    log_filename_for_run,
    resolve_log_path,
    setup_console_logging,
    setup_logging,
    suppress_noisy_loggers,
)
from host.config.profiles import (
    CONFIG_ENV_VAR,
    activate_profile,
    list_device_profiles,
    resolve_config_path,
)
from host.config.settings import REPO_ROOT, LoggingConfig, Settings, load_settings
from host.config.submodules import (
    CLAW_EVAL_DIR,
    LMMS_EVAL_DIR,
    SUBMODULES_DIR,
    TAU2_BENCH_DIR,
    ensure_claw_eval,
    ensure_lmms_eval,
    ensure_tau2_bench,
    submodule_status,
    tau2_data_dir,
)

__all__ = [
    "REPO_ROOT",
    "CONFIG_ENV_VAR",
    "LoggingConfig",
    "Settings",
    "CLAW_EVAL_DIR",
    "LMMS_EVAL_DIR",
    "SUBMODULES_DIR",
    "TAU2_BENCH_DIR",
    "activate_profile",
    "ensure_lmms_eval",
    "ensure_claw_eval",
    "ensure_tau2_bench",
    "claw_eval_env",
    "judge_env",
    "list_device_profiles",
    "lmms_eval_env",
    "load_settings",
    "log_filename_for_run",
    "resolve_config_path",
    "resolve_log_path",
    "setup_console_logging",
    "setup_logging",
    "submodule_status",
    "suppress_noisy_loggers",
    "tau_bench_env",
    "tau2_data_dir",
    "terminal_bench_2_env",
    "terminal_bench_env",
]
