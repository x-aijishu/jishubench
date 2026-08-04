//! Hardware Abstraction Layer for monitor.
//!
//! Platform backends:
//!   - Linux: [`OsLinuxProfiler`] (`/proc`, `/sys`)
//!   - macOS: [`OsDarwinProfiler`] (Mach/BSD APIs)
//!
//! Use [`PlatformProfiler`] (or [`platform_profiler`]) for the active target.

mod os_linux;

#[cfg(target_os = "macos")]
mod os_darwin;

#[cfg(feature = "plugin")]
pub mod cabi;

#[cfg(feature = "plugin")]
pub mod plugin;

#[cfg(any(feature = "plugin", feature = "nvidia"))]
pub mod composite;

#[cfg(feature = "nvidia")]
pub mod nvidia;

pub use os_linux::OsLinuxProfiler;

#[cfg(target_os = "macos")]
pub use os_darwin::OsDarwinProfiler;

/// Profiler implementation selected for the compilation target.
#[cfg(target_os = "linux")]
pub type PlatformProfiler = OsLinuxProfiler;

#[cfg(target_os = "macos")]
pub type PlatformProfiler = OsDarwinProfiler;

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub type PlatformProfiler = OsLinuxProfiler;

/// Construct the platform-appropriate system profiler.
pub fn platform_profiler() -> PlatformProfiler {
    PlatformProfiler::new()
}

use quanta::Instant;

/// Device-static information that does not change during the agent's lifetime.
/// Reported once at connection time via `/health` and per-trace metadata,
/// NOT duplicated in every sample.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct DeviceInfo {
    // ---- 系统内存 ----
    /// Total physical RAM in MB (from /proc/meminfo on Linux).
    pub mem_total_mb: Option<f64>,
    /// Total swap space in MB (0 = none configured).
    pub swap_total_mb: Option<f64>,

    // ---- CPU ----
    /// CPU model name string (e.g. "Neoverse V2", "Cortex-A76", "Intel Xeon Platinum 8480C").
    pub cpu_model_name: Option<String>,
    /// Number of CPU cores configured (sysconf _SC_NPROCESSORS_CONF).
    pub cpu_core_count: Option<u32>,
    /// CPU maximum frequency in MHz (from cpufreq sysfs).
    pub cpu_max_freq_mhz: Option<f64>,

    // ---- 板卡 / 设备 ----
    /// Board / device model (e.g. "NVIDIA Jetson Orin NX 16GB", "Rockchip RK3588 EVB").
    pub board_model: Option<String>,

    // ---- 运行时架构 ----
    /// CPU architecture (e.g. "aarch64", "x86_64").
    pub arch: Option<String>,

    // ---- GPU (NVIDIA) ----
    /// GPU VRAM total in MB (aggregate across all GPUs).
    pub gpu_mem_total_mb: Option<f64>,
    /// Human-readable GPU model name (e.g. "NVIDIA A100-SXM4-80GB").
    pub gpu_model: Option<String>,
    /// NVIDIA driver version (e.g. "535.154.05").
    pub gpu_driver_version: Option<String>,
    /// CUDA driver version (e.g. "12.2").
    pub cuda_version: Option<String>,
}

/// Per-sample hardware snapshot (dynamic fields only).
/// Static device properties are in [`DeviceInfo`].
#[derive(Debug, Clone)]
pub struct HardwareSample {
    /// Monotonic nanoseconds since the profiler's epoch (from `quanta::Clock`).
    pub t_mono_ns: u64,
    /// NPU utilisation 0–100 % (always `None` in the OS-level MVP).
    pub npu_util_pct: Option<f64>,
    /// System-wide CPU utilisation 0–100 % (`None` until a baseline exists).
    pub system_cpu_util_pct: Option<f64>,
    /// Available memory for new allocations in MB (includes reclaimable cache).
    pub mem_available_mb: Option<f64>,
    /// Used swap space in MB.
    pub swap_used_mb: Option<f64>,
    /// GPU utilisation 0–100 % (global on the currently supported backends).
    pub gpu_util_pct: Option<f64>,
    /// Used GPU VRAM in MB.
    pub gpu_mem_used_mb: Option<f64>,
    /// CPU temperature in degrees Celsius (`None` if unavailable).
    pub temp_celsius: Option<f64>,
    /// System load average over the last 1 minute.
    pub load_avg_1m: Option<f64>,
    /// System load average over the last 5 minutes.
    pub load_avg_5m: Option<f64>,
    /// System load average over the last 15 minutes.
    pub load_avg_15m: Option<f64>,
}

/// Which metrics this profiler can supply on the current platform/hardware.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Capabilities {
    pub npu_util: bool,
    pub system_cpu_util: bool,
    pub system_memory: bool,
    pub gpu_util: bool,
    pub gpu_memory: bool,
    pub temp_celsius: bool,
    pub load_avg: bool,
}

/// Non-fatal profiling errors (logged at startup and exposed via `/health`).
#[derive(Debug, thiserror::Error)]
pub enum HalError {
    #[error("permission denied reading {path}: {source}")]
    Permission {
        path: String,
        source: std::io::Error,
    },

    #[error("parse error for {path}: {msg}")]
    Parse { path: String, msg: String },

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// The single interface implemented by all profiler backends.
///
/// Implementations must be `Send + Sync` because the sampler loop holds an
/// `Arc<dyn HardwareProfiler>` across threads.
pub trait HardwareProfiler: Send + Sync {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> Capabilities;
    /// Device-static information collected once at startup (e.g. total RAM, CPU model).
    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            mem_total_mb: None,
            swap_total_mb: None,
            cpu_model_name: None,
            cpu_core_count: None,
            cpu_max_freq_mhz: None,
            board_model: None,
            arch: None,
            gpu_mem_total_mb: None,
            gpu_model: None,
            gpu_driver_version: None,
            cuda_version: None,
        }
    }
    /// Take a single hardware sample.  Non-fatal read errors should appear as
    /// `None` in the relevant field; see `errors()` for accumulated messages.
    fn sample(&self, clock: &quanta::Clock, epoch: Instant) -> HardwareSample;
    /// Accumulated non-fatal diagnostic messages (e.g. `EACCES` on a sysfs node).
    fn errors(&self) -> Vec<String>;
}
