//! Linux hardware profiler.
//!
//! Delegates each metric family to a sub-module:
//!   - cpu.rs    – system CPU utilisation from /proc/stat
//!   - memory.rs – /proc/meminfo system memory
//!   - temp.rs   – hwmon CPU temperature

mod cpu;
mod loadavg;
mod memory;
mod temp;

use std::sync::Mutex;

use quanta::Instant;

use crate::{Capabilities, DeviceInfo, HardwareProfiler, HardwareSample};

// ---------------------------------------------------------------------------
// OsLinuxProfiler
// ---------------------------------------------------------------------------

pub struct OsLinuxProfiler {
    caps: Capabilities,
    startup_errors: Vec<String>,
    di: DeviceInfo,
    cpu_ticks: Mutex<Option<cpu::CpuTicks>>,
}

impl Default for OsLinuxProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl OsLinuxProfiler {
    /// Build a profiler for system-level metrics.
    /// All sysfs probe failures are downgraded to entries in `startup_errors`.
    pub fn new() -> Self {
        let mut errs: Vec<String> = Vec::new();

        // Memory
        let sysmem = memory::read_sysmem_mb();
        let has_sysmem = sysmem.is_some();

        // Temperature (optional – probe once, cache path)
        let has_temp = temp::probe_temp_celsius().is_some();

        // Build DeviceInfo from cached /proc values
        let di = Self::collect_device_info(sysmem, &mut errs);

        OsLinuxProfiler {
            caps: Capabilities {
                npu_util: false,
                system_cpu_util: cpu::probe(),
                system_memory: has_sysmem,
                gpu_util: false, // Linux GPU requires plugin / nvidia feature
                gpu_memory: false,
                temp_celsius: has_temp,
                load_avg: true,
            },
            startup_errors: errs,
            di,
            cpu_ticks: Mutex::new(None),
        }
    }

    fn collect_device_info(
        sysmem: Option<(f64, f64, f64, f64)>,
        errs: &mut Vec<String>,
    ) -> DeviceInfo {
        let (mem_total_mb, swap_total_mb) = match sysmem {
            Some((total, _, _, swap_total)) => (Some(total), Some(swap_total)),
            None => (None, None),
        };

        let cpu_core_count = std::thread::available_parallelism()
            .ok()
            .map(|v| v.get() as u32);

        // CPU model name from /proc/cpuinfo
        let cpu_model_name = Self::read_cpu_model(errs);

        // Board model from device tree (ARM) or DMI (x86)
        let board_model = Self::read_board_model(errs);

        // CPU max frequency
        let cpu_max_freq_mhz = Self::read_cpu_max_freq(errs);

        // Runtime architecture
        let arch = Self::read_arch();

        DeviceInfo {
            mem_total_mb,
            swap_total_mb,
            cpu_model_name,
            cpu_core_count,
            cpu_max_freq_mhz,
            board_model,
            arch,
            gpu_mem_total_mb: None,
            gpu_model: None,
            gpu_driver_version: None,
            cuda_version: None,
        }
    }

    fn read_cpu_model(errs: &mut Vec<String>) -> Option<String> {
        let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        // On x86: look for "model name", on ARM: look for "Hardware" or "model name"
        for line in content.lines() {
            if let Some(val) = line.strip_prefix("model name\t: ") {
                return Some(val.trim().to_string());
            }
            if let Some(val) = line.strip_prefix("Hardware\t: ") {
                let trimmed = val.trim();
                if !trimmed.is_empty() && trimmed != "BCM2835" {
                    return Some(trimmed.to_string());
                }
            }
        }
        // Fallback: first "model name" or "Hardware" on aarch64
        for line in content.lines() {
            if let Some(val) = line.strip_prefix("model name\t: ") {
                return Some(val.trim().to_string());
            }
            if let Some(val) = line.strip_prefix("Hardware\t: ") {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
        errs.push("no cpu model found in /proc/cpuinfo".into());
        None
    }

    fn read_board_model(errs: &mut Vec<String>) -> Option<String> {
        // ARM device tree
        let dt_path = std::path::Path::new("/sys/firmware/devicetree/base/model");
        if dt_path.exists() {
            return std::fs::read_to_string(dt_path).ok().map(|s| {
                // device tree strings can have trailing null bytes
                s.trim_end_matches('\0').trim().to_string()
            });
        }
        // x86: DMI board name
        let dmi_path = std::path::Path::new("/sys/devices/virtual/dmi/id/board_name");
        if dmi_path.exists() {
            return std::fs::read_to_string(dmi_path)
                .ok()
                .map(|s| s.trim().to_string());
        }
        errs.push("no board model found (neither device-tree nor DMI)".into());
        None
    }

    fn read_cpu_max_freq(errs: &mut Vec<String>) -> Option<f64> {
        let path = std::path::Path::new("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
        if path.exists() {
            let khz: f64 = std::fs::read_to_string(path).ok()?.trim().parse().ok()?;
            return Some(khz / 1000.0);
        }
        errs.push("cpufreq not available".into());
        None
    }

    fn read_arch() -> Option<String> {
        // Try uname -m at runtime, fall back to compile-time const
        std::process::Command::new("uname")
            .arg("-m")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if s.is_empty() {
                    None
                } else {
                    Some(s)
                }
            })
            .or_else(|| Some(std::env::consts::ARCH.to_string()))
    }

    fn read_sysmem(&self) -> (Option<f64>, Option<f64>) {
        if let Some((_, avail, _, swap_used)) = memory::read_sysmem_mb() {
            (Some(avail), Some(swap_used))
        } else {
            (None, None)
        }
    }

    fn read_cpu_pct(&self) -> Option<f64> {
        cpu::read_cpu_pct(&mut self.cpu_ticks.lock().unwrap())
    }
}

impl HardwareProfiler for OsLinuxProfiler {
    fn name(&self) -> &'static str {
        "os_linux"
    }

    fn capabilities(&self) -> Capabilities {
        self.caps.clone()
    }

    fn device_info(&self) -> DeviceInfo {
        self.di.clone()
    }

    fn sample(&self, clock: &quanta::Clock, epoch: Instant) -> HardwareSample {
        let t_mono_ns = clock.now().duration_since(epoch).as_nanos() as u64;
        let (mem_avail, swap_used) = self.read_sysmem();
        let (load_1, load_5, load_15) = loadavg::read_load_avg()
            .map(|(a, b, c)| (Some(a), Some(b), Some(c)))
            .unwrap_or((None, None, None));

        HardwareSample {
            t_mono_ns,
            npu_util_pct: None,
            system_cpu_util_pct: self.read_cpu_pct(),
            mem_available_mb: mem_avail,
            swap_used_mb: swap_used,
            gpu_util_pct: None,
            gpu_mem_used_mb: None,
            temp_celsius: temp::probe_temp_celsius(),
            load_avg_1m: load_1,
            load_avg_5m: load_5,
            load_avg_15m: load_15,
        }
    }

    fn errors(&self) -> Vec<String> {
        self.startup_errors.clone()
    }
}
