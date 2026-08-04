//! Rust FFI bindings for `jishu_profiler.h` (the C ABI plugin interface).

use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::Path;

use crate::DeviceInfo;

/// Opaque C context pointer.
type CtxPtr = *mut std::ffi::c_void;

/// Per-sample hardware snapshot as seen on the C side.
/// Only dynamic fields — static info goes through DeviceInfo.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CapiSample {
    pub t_mono_ns: u64,
    pub npu_util_pct: f64,
    pub system_cpu_util_pct: f64,
    pub mem_available_mb: f64,
    pub swap_used_mb: f64,
    pub gpu_util_pct: f64,
    pub gpu_mem_used_mb: f64,
    pub temp_celsius: f64,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct CapiCapabilities {
    pub npu_util: bool,
    pub system_cpu_util: bool,
    pub system_memory: bool,
    pub gpu_util: bool,
    pub gpu_memory: bool,
    pub temp_celsius: bool,
}

/// DeviceInfo as seen on the C side.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CapiDeviceInfo {
    pub mem_total_mb: f64,                 // NaN = unavailable
    pub swap_total_mb: f64,                // NaN = unavailable
    pub gpu_mem_total_mb: f64,             // NaN = unavailable
    pub gpu_model: *const c_char,          // NULL = unavailable
    pub gpu_driver_version: *const c_char, // NULL = unavailable
    pub cuda_version: *const c_char,       // NULL = unavailable
}

/// The C vtable loaded from a plugin `.so`.
#[repr(C)]
pub struct CapiProfilerVtable {
    pub name: extern "C" fn(ctx: CtxPtr) -> *const c_char,
    pub capabilities: extern "C" fn(ctx: CtxPtr) -> CapiCapabilities,
    pub sample: extern "C" fn(ctx: CtxPtr, epoch_ns: u64) -> CapiSample,
    pub destroy: Option<extern "C" fn(ctx: CtxPtr)>,
    pub ctx: CtxPtr,
}

// SAFETY: The plugin .so is trusted and the vtable is designed for cross-thread
// use. The `ctx: *mut c_void` field is opaque and managed by the plugin; we
// assert that the plugin's implementation is thread-safe.
unsafe impl Send for CapiProfilerVtable {}
unsafe impl Sync for CapiProfilerVtable {}

/// C entry point signature – every plugin must export this.
pub type CreateFn = extern "C" fn() -> *mut CapiProfilerVtable;

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

impl CapiSample {
    /// Convert NaN sentinels to `None`.
    pub fn capi_sample_to_option_f64(val: f64) -> Option<f64> {
        if val.is_nan() {
            None
        } else {
            Some(val)
        }
    }
}

impl From<CapiCapabilities> for crate::Capabilities {
    fn from(c: CapiCapabilities) -> Self {
        crate::Capabilities {
            npu_util: c.npu_util,
            system_cpu_util: c.system_cpu_util,
            system_memory: c.system_memory,
            gpu_util: c.gpu_util,
            gpu_memory: c.gpu_memory,
            temp_celsius: c.temp_celsius,
            load_avg: false,
        }
    }
}

impl From<CapiDeviceInfo> for DeviceInfo {
    fn from(c: CapiDeviceInfo) -> Self {
        let nan_to_none = |v: f64| if v.is_nan() { None } else { Some(v) };
        let ptr_to_string = |p: *const c_char| {
            if p.is_null() {
                None
            } else {
                unsafe { CStr::from_ptr(p) }
                    .to_string_lossy()
                    .into_owned()
                    .into()
            }
        };

        DeviceInfo {
            mem_total_mb: nan_to_none(c.mem_total_mb),
            swap_total_mb: nan_to_none(c.swap_total_mb),
            cpu_model_name: None,
            cpu_core_count: None,
            cpu_max_freq_mhz: None,
            board_model: None,
            arch: None,
            gpu_mem_total_mb: nan_to_none(c.gpu_mem_total_mb),
            gpu_model: ptr_to_string(c.gpu_model),
            gpu_driver_version: ptr_to_string(c.gpu_driver_version),
            cuda_version: ptr_to_string(c.cuda_version),
        }
    }
}

/// Load a dynamic library and look up the `jishu_profiler_create` symbol.
///
/// Returns the vtable pointer on success, or an error description on failure.
pub fn load_plugin(path: &Path) -> Result<Box<CapiProfilerVtable>, String> {
    let lib_path = path;
    let lib_name = lib_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("plugin");

    // SAFETY: libloading is used to dlopen a trusted .so.
    // The plugin is expected to have been compiled against the same ABI.
    let lib = unsafe {
        libloading::Library::new(lib_path)
            .map_err(|e| format!("failed to load plugin `{lib_name}`: {e}"))?
    };

    // Dereference immediately to release the borrow on `lib`.
    let create: CreateFn = unsafe {
        *lib.get(b"jishu_profiler_create")
            .map_err(|e| format!("symbol `jishu_profiler_create` not found in `{lib_name}`: {e}"))?
    };

    // Leak the Library handle so it stays loaded for the lifetime of the process.
    let _leaked = Box::into_raw(Box::new(lib));

    let vtable_ptr = create();
    if vtable_ptr.is_null() {
        return Err(format!(
            "plugin `{lib_name}`: jishu_profiler_create returned NULL"
        ));
    }

    // SAFETY: the plugin guarantees the vtable lives until destroy() is called.
    let vtable = unsafe { Box::from_raw(vtable_ptr) };
    Ok(vtable)
}

/// Extract the plugin name from its vtable.
pub fn plugin_name(vtable: &CapiProfilerVtable) -> String {
    let c_str = unsafe { CStr::from_ptr((vtable.name)(vtable.ctx)) };
    c_str.to_string_lossy().into_owned()
}

/// Free a plugin vtable (calls destroy if defined, then deallocates).
pub fn destroy_plugin(vtable: Box<CapiProfilerVtable>) {
    if let Some(destroy) = vtable.destroy {
        (destroy)(vtable.ctx);
    }
    drop(vtable);
}
