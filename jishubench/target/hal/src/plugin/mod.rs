//! PluginLoader – wraps a dynamically-loaded C ABI plugin as a `HardwareProfiler`.

use std::path::Path;

use quanta::{Clock, Instant};

use crate::{
    cabi::{self, CapiProfilerVtable, CapiSample},
    Capabilities, DeviceInfo, HardwareProfiler, HardwareSample,
};

/// A `HardwareProfiler` backed by a dynamically-loaded C plugin (.so).
pub struct PluginProfiler {
    name: String,
    caps: Capabilities,
    vtable: Box<CapiProfilerVtable>,
}

impl PluginProfiler {
    /// Load a plugin from a `.so` path.
    pub fn load(path: &Path) -> Result<Self, String> {
        let vtable = cabi::load_plugin(path)?;
        let name = cabi::plugin_name(&vtable);
        let caps = (vtable.capabilities)(vtable.ctx).into();
        Ok(PluginProfiler { name, caps, vtable })
    }

    /// Wrap an already-loaded vtable (for testing / static plugins).
    pub fn from_vtable(vtable: Box<CapiProfilerVtable>) -> Self {
        let name = cabi::plugin_name(&vtable);
        let caps = (vtable.capabilities)(vtable.ctx).into();
        PluginProfiler { name, caps, vtable }
    }
}

impl HardwareProfiler for PluginProfiler {
    fn name(&self) -> &'static str {
        Box::leak(self.name.clone().into_boxed_str())
    }

    fn capabilities(&self) -> Capabilities {
        self.caps.clone()
    }

    fn device_info(&self) -> DeviceInfo {
        // Plugin might export a device_info symbol; for now, return default.
        // TODO: look up optional `jishu_profiler_device_info` symbol if present.
        DeviceInfo::default()
    }

    fn sample(&self, clock: &Clock, epoch: Instant) -> HardwareSample {
        let t_mono_ns = clock.now().duration_since(epoch).as_nanos() as u64;
        let cs = (self.vtable.sample)(self.vtable.ctx, t_mono_ns);

        HardwareSample {
            t_mono_ns: cs.t_mono_ns,
            npu_util_pct: CapiSample::capi_sample_to_option_f64(cs.npu_util_pct),
            system_cpu_util_pct: CapiSample::capi_sample_to_option_f64(cs.system_cpu_util_pct),
            mem_available_mb: CapiSample::capi_sample_to_option_f64(cs.mem_available_mb),
            swap_used_mb: CapiSample::capi_sample_to_option_f64(cs.swap_used_mb),
            gpu_util_pct: CapiSample::capi_sample_to_option_f64(cs.gpu_util_pct),
            gpu_mem_used_mb: CapiSample::capi_sample_to_option_f64(cs.gpu_mem_used_mb),
            temp_celsius: CapiSample::capi_sample_to_option_f64(cs.temp_celsius),
            load_avg_1m: None,
            load_avg_5m: None,
            load_avg_15m: None,
        }
    }

    fn errors(&self) -> Vec<String> {
        vec![]
    }
}

impl Drop for PluginProfiler {
    fn drop(&mut self) {
        // Nothing extra needed – the vtable handle is dropped which may call
        // a user-provided drop if the cabi layer handles it.
    }
}
