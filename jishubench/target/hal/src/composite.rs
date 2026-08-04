//! CompositeProfiler – merges the built-in OS profiler with optional
//! GPU (NVIDIA) and plugin profilers.
//!
//! Priority: plugin > gpu > builtin.  For each metric, the highest-priority
//! source that claims the capability wins.

use std::sync::Arc;

use quanta::Instant;

#[cfg(feature = "plugin")]
use crate::plugin::PluginProfiler;
use crate::{Capabilities, DeviceInfo, HardwareProfiler, HardwareSample};

/// A profiler that merges the built-in OS profiler with optional GPU and
/// plugin profilers.
pub struct CompositeProfiler {
    name: String,
    caps: Capabilities,
    builtin: Arc<dyn HardwareProfiler>,
    gpu: Option<Arc<dyn HardwareProfiler>>,
    #[cfg(feature = "plugin")]
    plugin: Option<PluginProfiler>,
    startup_errors: Vec<String>,
}

impl CompositeProfiler {
    /// Build a new composite profiler.
    ///
    /// `builtin` is typically `OsLinuxProfiler` or `OsDarwinProfiler`.
    /// `gpu` is an optional GPU profiler (e.g. NvidiaProfiler).
    /// `plugin` is an optional dynamically-loaded profiler plugin `.so`.
    pub fn new(
        builtin: Arc<dyn HardwareProfiler>,
        gpu: Option<Arc<dyn HardwareProfiler>>,
        #[cfg(feature = "plugin")] plugin: Option<PluginProfiler>,
    ) -> Self {
        let b_caps = builtin.capabilities();
        let gpu_name = gpu.as_ref().map(|g| g.name().to_string());

        let mut errs = vec![];
        if let Some(ref g) = gpu {
            errs.extend(g.errors());
        }

        #[cfg(feature = "plugin")]
        let plugin_name = plugin.as_ref().map(|p| p.name().to_string());
        #[cfg(feature = "plugin")]
        if let Some(ref p) = plugin {
            errs.extend(p.errors());
        }

        // Merge capabilities: gpu overrides builtin, plugin overrides gpu.
        let caps = {
            let mut c = b_caps.clone();

            if let Some(ref g) = gpu {
                let g_caps = g.capabilities();
                if g_caps.gpu_util {
                    c.gpu_util = true;
                }
                if g_caps.gpu_memory {
                    c.gpu_memory = true;
                }
                if g_caps.temp_celsius {
                    c.temp_celsius = true;
                }
            }

            #[cfg(feature = "plugin")]
            if let Some(ref p) = plugin {
                let p_caps = p.capabilities();
                if p_caps.npu_util {
                    c.npu_util = true;
                }
                if p_caps.system_cpu_util {
                    c.system_cpu_util = true;
                }
                if p_caps.system_memory {
                    c.system_memory = true;
                }
                if p_caps.gpu_util {
                    c.gpu_util = true;
                }
                if p_caps.gpu_memory {
                    c.gpu_memory = true;
                }
                if p_caps.temp_celsius {
                    c.temp_celsius = true;
                }
            }

            c
        };

        // Build composite name
        #[cfg(feature = "plugin")]
        let name = match (gpu_name, plugin_name) {
            (Some(g), Some(p)) => format!("{}+{}+{}", builtin.name(), g, p),
            (Some(g), None) => format!("{}+{}", builtin.name(), g),
            (None, Some(p)) => format!("{}+{}", builtin.name(), p),
            (None, None) => builtin.name().to_string(),
        };
        #[cfg(not(feature = "plugin"))]
        let name = match gpu_name {
            Some(g) => format!("{}+{}", builtin.name(), g),
            None => builtin.name().to_string(),
        };

        CompositeProfiler {
            name,
            caps,
            builtin,
            gpu,
            #[cfg(feature = "plugin")]
            plugin,
            startup_errors: errs,
        }
    }
}

impl HardwareProfiler for CompositeProfiler {
    fn name(&self) -> &'static str {
        Box::leak(self.name.clone().into_boxed_str())
    }

    fn capabilities(&self) -> Capabilities {
        self.caps.clone()
    }

    fn device_info(&self) -> DeviceInfo {
        let mut di = self.builtin.device_info();

        if let Some(ref g) = self.gpu {
            let g_di = g.device_info();
            if g_di.gpu_mem_total_mb.is_some() {
                di.gpu_mem_total_mb = g_di.gpu_mem_total_mb;
            }
            if g_di.gpu_model.is_some() {
                di.gpu_model = g_di.gpu_model;
            }
            if g_di.gpu_driver_version.is_some() {
                di.gpu_driver_version = g_di.gpu_driver_version;
            }
            if g_di.cuda_version.is_some() {
                di.cuda_version = g_di.cuda_version;
            }
            if g_di.board_model.is_some() {
                di.board_model = g_di.board_model;
            }
            if g_di.cpu_model_name.is_some() {
                di.cpu_model_name = g_di.cpu_model_name;
            }
            if g_di.cpu_core_count.is_some() {
                di.cpu_core_count = g_di.cpu_core_count;
            }
            if g_di.cpu_max_freq_mhz.is_some() {
                di.cpu_max_freq_mhz = g_di.cpu_max_freq_mhz;
            }
            if g_di.mem_total_mb.is_some() {
                di.mem_total_mb = g_di.mem_total_mb;
            }
            if g_di.swap_total_mb.is_some() {
                di.swap_total_mb = g_di.swap_total_mb;
            }
            if g_di.arch.is_some() {
                di.arch = g_di.arch;
            }
        }

        #[cfg(feature = "plugin")]
        if let Some(ref p) = self.plugin {
            let p_di = p.device_info();
            if p_di.mem_total_mb.is_some() {
                di.mem_total_mb = p_di.mem_total_mb;
            }
            if p_di.swap_total_mb.is_some() {
                di.swap_total_mb = p_di.swap_total_mb;
            }
            if p_di.cpu_model_name.is_some() {
                di.cpu_model_name = p_di.cpu_model_name;
            }
            if p_di.cpu_core_count.is_some() {
                di.cpu_core_count = p_di.cpu_core_count;
            }
            if p_di.cpu_max_freq_mhz.is_some() {
                di.cpu_max_freq_mhz = p_di.cpu_max_freq_mhz;
            }
            if p_di.board_model.is_some() {
                di.board_model = p_di.board_model;
            }
            if p_di.arch.is_some() {
                di.arch = p_di.arch;
            }
            if p_di.gpu_mem_total_mb.is_some() {
                di.gpu_mem_total_mb = p_di.gpu_mem_total_mb;
            }
            if p_di.gpu_model.is_some() {
                di.gpu_model = p_di.gpu_model;
            }
            if p_di.gpu_driver_version.is_some() {
                di.gpu_driver_version = p_di.gpu_driver_version;
            }
            if p_di.cuda_version.is_some() {
                di.cuda_version = p_di.cuda_version;
            }
        }

        di
    }

    fn sample(&self, clock: &quanta::Clock, epoch: Instant) -> HardwareSample {
        let mut sample = self.builtin.sample(clock, epoch);

        // Apply GPU profiler overrides
        if let Some(ref g) = self.gpu {
            let g_caps = g.capabilities();
            let g_sample = g.sample(clock, epoch);

            if g_caps.gpu_util {
                sample.gpu_util_pct = g_sample.gpu_util_pct;
            }
            if g_caps.gpu_memory {
                sample.gpu_mem_used_mb = g_sample.gpu_mem_used_mb;
            }
            if g_caps.temp_celsius && g_sample.temp_celsius.is_some() {
                sample.temp_celsius = g_sample.temp_celsius;
            }
        }

        // Apply plugin overrides (highest priority)
        #[cfg(feature = "plugin")]
        if let Some(ref p) = self.plugin {
            let p_caps = p.capabilities();
            let p_sample = p.sample(clock, epoch);

            if p_caps.npu_util {
                sample.npu_util_pct = p_sample.npu_util_pct;
            }
            if p_caps.system_cpu_util {
                sample.system_cpu_util_pct = p_sample.system_cpu_util_pct;
            }
            if p_caps.system_memory {
                sample.mem_available_mb = p_sample.mem_available_mb;
                sample.swap_used_mb = p_sample.swap_used_mb;
            }
            if p_caps.gpu_util {
                sample.gpu_util_pct = p_sample.gpu_util_pct;
            }
            if p_caps.gpu_memory {
                sample.gpu_mem_used_mb = p_sample.gpu_mem_used_mb;
            }
            if p_caps.temp_celsius && p_sample.temp_celsius.is_some() {
                sample.temp_celsius = p_sample.temp_celsius;
            }
        }

        sample
    }

    fn errors(&self) -> Vec<String> {
        let mut e = self.builtin.errors();
        if let Some(ref g) = self.gpu {
            e.extend(g.errors());
        }
        e.extend(self.startup_errors.clone());
        e
    }
}
