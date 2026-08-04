//! macOS hardware profiler.
//!
//! Delegates each metric family to a sub-module:
//!   - cpu.rs    – system CPU utilisation from Mach host statistics
//!   - memory.rs – system memory via Mach/BSD
//!   - gpu.rs    – global GPU utilisation (IOGPU)

mod cpu;
mod gpu;
mod loadavg;
mod memory;

use quanta::Instant;

use crate::{Capabilities, DeviceInfo, HardwareProfiler, HardwareSample};

// ---------------------------------------------------------------------------
// OsDarwinProfiler
// ---------------------------------------------------------------------------

pub struct OsDarwinProfiler {
    caps: Capabilities,
    startup_errors: Vec<String>,
    di: DeviceInfo,
    cpu_ticks: std::sync::Mutex<Option<cpu::CpuTicks>>,
}

impl Default for OsDarwinProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl OsDarwinProfiler {
    /// Build a profiler for system-level metrics.
    pub fn new() -> Self {
        // System memory
        let sysmem = memory::read_sysmem_mb();
        let has_sysmem = sysmem.is_some();

        // GPU
        let has_gpu = gpu::read_gpu_util_pct().is_some();

        let di = Self::collect_device_info(sysmem);

        OsDarwinProfiler {
            caps: Capabilities {
                npu_util: false,
                system_cpu_util: cpu::probe(),
                system_memory: has_sysmem,
                gpu_util: has_gpu,
                gpu_memory: false,
                temp_celsius: false,
                load_avg: true,
            },
            startup_errors: Vec::new(),
            di,
            cpu_ticks: std::sync::Mutex::new(None),
        }
    }

    fn collect_device_info(sysmem: Option<(f64, f64, f64, f64)>) -> DeviceInfo {
        let (mem_total_mb, swap_total_mb) = match sysmem {
            Some((total, _, _, swap_total)) => (Some(total), Some(swap_total)),
            None => (None, None),
        };

        let cpu_core_count = std::thread::available_parallelism()
            .ok()
            .map(|v| v.get() as u32);

        // sysctl hw.model for macOS model identifier
        let cpu_model_name = Self::sysctl_string("hw.model");

        // sysctl hw.cpufrequency_max (may be 0 on some Macs)
        let cpu_max_freq_mhz = Self::sysctl_u64("hw.cpufrequency_max")
            .map(|hz| hz as f64 / 1_000_000.0)
            .filter(|&v| v > 0.0);

        // Runtime architecture
        let arch = Self::read_arch();

        DeviceInfo {
            mem_total_mb,
            swap_total_mb,
            cpu_model_name,
            cpu_core_count,
            cpu_max_freq_mhz,
            board_model: None, // macOS has no device-tree / DMI
            arch,
            gpu_mem_total_mb: None,
            gpu_model: None,
            gpu_driver_version: None,
            cuda_version: None,
        }
    }

    fn sysctl_string(key: &str) -> Option<String> {
        let c_key = std::ffi::CString::new(key).ok()?;
        let mut len: libc::size_t = 0;
        // First call: get required buffer size
        if unsafe {
            libc::sysctlbyname(
                c_key.as_ptr(),
                std::ptr::null_mut(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        } != 0
        {
            return None;
        }
        let mut buf = vec![0u8; len];
        if unsafe {
            libc::sysctlbyname(
                c_key.as_ptr(),
                buf.as_mut_ptr() as *mut _,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        } != 0
        {
            return None;
        }
        // Trim trailing null bytes
        let s = String::from_utf8_lossy(&buf)
            .trim_end_matches('\0')
            .to_string();
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    }

    fn sysctl_u64(key: &str) -> Option<u64> {
        let c_key = std::ffi::CString::new(key).ok()?;
        let mut val: u64 = 0;
        let mut len: libc::size_t = std::mem::size_of::<u64>();
        if unsafe {
            libc::sysctlbyname(
                c_key.as_ptr(),
                &mut val as *mut _ as *mut _,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        } != 0
        {
            return None;
        }
        Some(val)
    }

    fn read_arch() -> Option<String> {
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

impl HardwareProfiler for OsDarwinProfiler {
    fn name(&self) -> &'static str {
        "os_darwin"
    }

    fn capabilities(&self) -> Capabilities {
        self.caps.clone()
    }

    fn device_info(&self) -> DeviceInfo {
        self.di.clone()
    }

    fn sample(&self, clock: &quanta::Clock, epoch: Instant) -> HardwareSample {
        let (mem_avail, swap_used) = self.read_sysmem();
        let (load_1, load_5, load_15) = loadavg::read_load_avg()
            .map(|(a, b, c)| (Some(a), Some(b), Some(c)))
            .unwrap_or((None, None, None));

        HardwareSample {
            t_mono_ns: mono_ns(clock, epoch),
            npu_util_pct: None,
            system_cpu_util_pct: self.read_cpu_pct(),
            mem_available_mb: mem_avail,
            swap_used_mb: swap_used,
            gpu_util_pct: gpu::read_gpu_util_pct(),
            gpu_mem_used_mb: None,
            temp_celsius: None,
            load_avg_1m: load_1,
            load_avg_5m: load_5,
            load_avg_15m: load_15,
        }
    }

    fn errors(&self) -> Vec<String> {
        self.startup_errors.clone()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mono_ns(clock: &quanta::Clock, epoch: Instant) -> u64 {
    clock.now().duration_since(epoch).as_nanos() as u64
}
