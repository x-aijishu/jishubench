//! NVIDIA GPU profiler: NVML (dGPU) → Jetson sysfs (embedded) fallback.
//!
//! `NvidiaProfiler` first tries to load NVML for desktop/server GPUs
//! (A100/H100/RTX/etc.) and falls back to reading sysfs on Jetson platforms.

mod jetson;
mod nvml;

use quanta::{Clock, Instant};

use crate::{Capabilities, DeviceInfo, HardwareProfiler, HardwareSample};

/// Combine all GPUs into a single sample by averaging or summing metrics.
struct GpuSample {
    gpu_util_pct: Option<f64>,
    gpu_mem_used_mb: Option<f64>,
    temp_celsius: Option<f64>,
}

/// NVIDIA GPU profiler.
///
/// Tries NVML first for dGPUs, then Jetson sysfs for embedded devices.
pub struct NvidiaProfiler {
    backend: NvidiaBackend,
    device_info: DeviceInfo,
}

enum NvidiaBackend {
    Nvml(nvml::NvmlSession, Vec<nvml::NvmlDevice>),
    Jetson,
    Unavailable,
}

impl NvidiaProfiler {
    /// Create a new `NvidiaProfiler`, probing the available backends.
    pub fn new() -> Self {
        // Try NVML first (dGPU)
        if let Some(session) = nvml::NvmlSession::open() {
            let devices = session.devices();
            let n_gpus = devices.len();
            if n_gpus > 0 {
                let di = Self::nvml_device_info(&session, &devices);
                return NvidiaProfiler {
                    backend: NvidiaBackend::Nvml(session, devices),
                    device_info: di,
                };
            }
        }

        // Fallback to Jetson sysfs
        if jetson::detect_jetson() {
            let di = Self::jetson_device_info();
            return NvidiaProfiler {
                backend: NvidiaBackend::Jetson,
                device_info: di,
            };
        }

        // No GPU found
        NvidiaProfiler {
            backend: NvidiaBackend::Unavailable,
            device_info: DeviceInfo::default(),
        }
    }

    fn nvml_device_info(session: &nvml::NvmlSession, devices: &[nvml::NvmlDevice]) -> DeviceInfo {
        let primary = &devices[0];
        let model = primary.name(session);
        let mem = primary.memory_info(session);
        let cuda_cc = primary.cuda_compute_capability(session);

        DeviceInfo {
            gpu_model: model,
            gpu_mem_total_mb: mem.as_ref().map(|m| m.total_mb()),
            gpu_driver_version: None,
            cuda_version: cuda_cc,
            ..DeviceInfo::default()
        }
    }

    fn jetson_device_info() -> DeviceInfo {
        DeviceInfo {
            gpu_model: jetson::read_board_model()
                .or_else(|| Some("NVIDIA Jetson (unknown)".into())),
            gpu_mem_total_mb: jetson::read_gpu_mem_total_mb(),
            ..DeviceInfo::default()
        }
    }

    fn nvml_sample(&self, session: &nvml::NvmlSession, devices: &[nvml::NvmlDevice]) -> GpuSample {
        let mut gpu_util_sum = 0.0f64;
        let mut gpu_util_count = 0u32;
        let mut mem_used_sum = 0.0f64;
        let mut mem_used_count = 0u32;
        let mut temp_sum = 0.0f64;
        let mut temp_count = 0u32;

        for dev in devices {
            if let Some((util_gpu, _util_mem)) = dev.utilization(session) {
                gpu_util_sum += util_gpu as f64;
                gpu_util_count += 1;
            }
            if let Some(mem) = dev.memory_info(session) {
                mem_used_sum += mem.used_mb();
                mem_used_count += 1;
            }
            if let Some(temp) = dev.temperature(session) {
                temp_sum += temp;
                temp_count += 1;
            }
        }

        GpuSample {
            gpu_util_pct: if gpu_util_count > 0 {
                Some(gpu_util_sum / gpu_util_count as f64)
            } else {
                None
            },
            gpu_mem_used_mb: if mem_used_count > 0 {
                Some(mem_used_sum / mem_used_count as f64)
            } else {
                None
            },
            temp_celsius: if temp_count > 0 {
                Some(temp_sum / temp_count as f64)
            } else {
                None
            },
        }
    }

    fn jetson_sample(&self) -> GpuSample {
        GpuSample {
            gpu_util_pct: jetson::read_gpu_util_pct(),
            gpu_mem_used_mb: None, // Jetson doesn't expose per-sample GPU mem usage
            temp_celsius: jetson::read_gpu_temp_celsius(),
        }
    }
}

impl HardwareProfiler for NvidiaProfiler {
    fn name(&self) -> &'static str {
        match self.backend {
            NvidiaBackend::Nvml(..) => "nvidia_nvml",
            NvidiaBackend::Jetson => "nvidia_jetson",
            NvidiaBackend::Unavailable => "nvidia_unavail",
        }
    }

    fn capabilities(&self) -> Capabilities {
        match self.backend {
            NvidiaBackend::Nvml(..) => Capabilities {
                gpu_util: true,
                gpu_memory: true,
                temp_celsius: true,
                ..Capabilities::default()
            },
            NvidiaBackend::Jetson => Capabilities {
                gpu_util: true,
                gpu_memory: false,
                temp_celsius: true,
                ..Capabilities::default()
            },
            NvidiaBackend::Unavailable => Capabilities::default(),
        }
    }

    fn device_info(&self) -> DeviceInfo {
        self.device_info.clone()
    }

    fn sample(&self, _clock: &Clock, _epoch: Instant) -> HardwareSample {
        let gpu = match &self.backend {
            NvidiaBackend::Nvml(session, devices) => self.nvml_sample(session, devices),
            NvidiaBackend::Jetson => self.jetson_sample(),
            NvidiaBackend::Unavailable => GpuSample {
                gpu_util_pct: None,
                gpu_mem_used_mb: None,
                temp_celsius: None,
            },
        };

        HardwareSample {
            t_mono_ns: 0,
            npu_util_pct: None,
            system_cpu_util_pct: None,
            mem_available_mb: None,
            swap_used_mb: None,
            gpu_util_pct: gpu.gpu_util_pct,
            gpu_mem_used_mb: gpu.gpu_mem_used_mb,
            temp_celsius: gpu.temp_celsius,
            load_avg_1m: None,
            load_avg_5m: None,
            load_avg_15m: None,
        }
    }

    fn errors(&self) -> Vec<String> {
        match self.backend {
            NvidiaBackend::Unavailable => vec!["NVIDIA GPU not found (NVML + Jetson sysfs)".into()],
            _ => vec![],
        }
    }
}
