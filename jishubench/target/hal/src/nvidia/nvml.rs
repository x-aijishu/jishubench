//! NVML runtime dynamic loading via `libloading`.
//!
//! Provides safe wrappers around a subset of the NVML C API needed for
//! GPU monitoring (memory, utilization, temperature, device info).
//! The `libnvidia-ml.so.1` library is loaded at runtime — no build-time
//! dependency on the CUDA / NVML SDK.

use std::ffi::{c_uint, c_ulong, CStr};
use std::os::raw::c_int;
use std::sync::Mutex;

use libloading::Library;

// ---------------------------------------------------------------------------
// NVML types (matching the C ABI)
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
type nvmlReturn_t = c_uint;
#[allow(non_camel_case_types)]
type nvmlDevice_t = u64;

const NVML_SUCCESS: nvmlReturn_t = 0;

// ---------------------------------------------------------------------------
// NVML API function pointers (loaded at runtime)
// ---------------------------------------------------------------------------

struct NvmlFunctions {
    init: extern "C" fn() -> nvmlReturn_t,
    shutdown: extern "C" fn() -> nvmlReturn_t,
    device_get_handle_by_index: extern "C" fn(c_uint, *mut nvmlDevice_t) -> nvmlReturn_t,
    device_get_name: extern "C" fn(nvmlDevice_t, *mut i8, c_uint) -> nvmlReturn_t,
    device_get_memory_info: extern "C" fn(nvmlDevice_t, *mut nvmlMemory_t) -> nvmlReturn_t,
    device_get_utilization_rates:
        extern "C" fn(nvmlDevice_t, *mut nvmlUtilization_t) -> nvmlReturn_t,
    device_get_temperature: extern "C" fn(nvmlDevice_t, c_uint, *mut c_uint) -> nvmlReturn_t,
    device_get_cuda_compute_capability:
        extern "C" fn(nvmlDevice_t, *mut c_int, *mut c_int) -> nvmlReturn_t,
}

#[repr(C)]
struct nvmlMemory_t {
    pub total: c_ulong,
    pub free: c_ulong,
    pub used: c_ulong,
}

#[repr(C)]
struct nvmlUtilization_t {
    pub gpu: c_uint,
    pub memory: c_uint,
}

/// Per-GPU handle returned by NVML.
pub struct NvmlDevice {
    handle: nvmlDevice_t,
    #[allow(dead_code)]
    index: u32,
}

// ---------------------------------------------------------------------------
// NVML session handle
// ---------------------------------------------------------------------------

/// RAII handle for the NVML library.  Call `NvmlSession::open()` to initialise.
pub struct NvmlSession {
    _lib: Mutex<Library>,
    fns: NvmlFunctions,
    device_count: u32,
}

unsafe impl Send for NvmlSession {}
unsafe impl Sync for NvmlSession {}

impl NvmlSession {
    /// Try to open `libnvidia-ml.so.1` and call `nvmlInit_v2`.
    /// Returns `None` if the library is not available or initialisation fails.
    pub fn open() -> Option<Self> {
        let lib_paths = ["libnvidia-ml.so.1", "libnvidia-ml.so"];
        let lib = unsafe { lib_paths.iter().find_map(|name| Library::new(name).ok())? };

        let fns = NvmlFunctions {
            init: unsafe { *lib.get(b"nvmlInit_v2").ok()? },
            shutdown: unsafe { *lib.get(b"nvmlShutdown").ok()? },
            device_get_handle_by_index: unsafe { *lib.get(b"nvmlDeviceGetHandleByIndex_v2").ok()? },
            device_get_name: unsafe { *lib.get(b"nvmlDeviceGetName").ok()? },
            device_get_memory_info: unsafe { *lib.get(b"nvmlDeviceGetMemoryInfo").ok()? },
            device_get_utilization_rates: unsafe {
                *lib.get(b"nvmlDeviceGetUtilizationRates").ok()?
            },
            device_get_temperature: unsafe { *lib.get(b"nvmlDeviceGetTemperature").ok()? },
            device_get_cuda_compute_capability: unsafe {
                *lib.get(b"nvmlDeviceGetCudaComputeCapability").ok()?
            },
        };

        let ret = (fns.init)();
        if ret != NVML_SUCCESS {
            return None;
        }

        // Count devices: probe with increasing indices until one fails
        // A more robust approach would use nvmlDeviceGetCount_v2, but we keep
        // the symbol table minimal and just try indices.
        let mut device_count = 0u32;
        for i in 0..64 {
            let mut handle: nvmlDevice_t = 0;
            let ret = (fns.device_get_handle_by_index)(i, &mut handle);
            if ret != NVML_SUCCESS {
                break;
            }
            device_count = i + 1;
        }

        Some(NvmlSession {
            _lib: Mutex::new(lib),
            fns,
            device_count,
        })
    }

    /// Return the number of visible GPUs.
    #[allow(dead_code)]
    pub fn device_count(&self) -> u32 {
        self.device_count
    }

    /// Open a handle to the i-th GPU.
    pub fn device_by_index(&self, index: u32) -> Option<NvmlDevice> {
        let mut handle: nvmlDevice_t = 0;
        let ret = (self.fns.device_get_handle_by_index)(index, &mut handle);
        if ret != NVML_SUCCESS {
            return None;
        }
        Some(NvmlDevice { handle, index })
    }

    /// Iterate over all visible GPUs.
    pub fn devices(&self) -> Vec<NvmlDevice> {
        (0..self.device_count)
            .filter_map(|i| self.device_by_index(i))
            .collect()
    }
}

impl Drop for NvmlSession {
    fn drop(&mut self) {
        let _ = (self.fns.shutdown)();
    }
}

impl NvmlDevice {
    /// GPU model name (e.g. "NVIDIA A100 80GB PCIe").
    pub fn name(&self, session: &NvmlSession) -> Option<String> {
        let mut buf = [0i8; 256];
        let ret = (session.fns.device_get_name)(self.handle, buf.as_mut_ptr(), buf.len() as c_uint);
        if ret != NVML_SUCCESS {
            return None;
        }
        // c_char is i8 on x86_64 / macOS, u8 on aarch64 Linux — cast for portability.
        let cstr = unsafe { CStr::from_ptr(buf.as_ptr().cast()) };
        Some(cstr.to_string_lossy().into_owned())
    }

    /// Memory info in bytes.
    pub fn memory_info(&self, session: &NvmlSession) -> Option<NvmlMemoryInfo> {
        let mut mem = nvmlMemory_t {
            total: 0,
            free: 0,
            used: 0,
        };
        let ret = (session.fns.device_get_memory_info)(self.handle, &mut mem);
        if ret != NVML_SUCCESS {
            return None;
        }
        Some(NvmlMemoryInfo {
            total_bytes: mem.total,
            free_bytes: mem.free,
            used_bytes: mem.used,
        })
    }

    /// GPU utilisation in percent (0-100).
    pub fn utilization(&self, session: &NvmlSession) -> Option<(u32, u32)> {
        let mut util = nvmlUtilization_t { gpu: 0, memory: 0 };
        let ret = (session.fns.device_get_utilization_rates)(self.handle, &mut util);
        if ret != NVML_SUCCESS {
            return None;
        }
        Some((util.gpu, util.memory))
    }

    /// GPU temperature in Celsius.
    pub fn temperature(&self, session: &NvmlSession) -> Option<f64> {
        const NVML_TEMP_GPU: c_uint = 0;
        let mut temp: c_uint = 0;
        let ret = (session.fns.device_get_temperature)(self.handle, NVML_TEMP_GPU, &mut temp);
        if ret != NVML_SUCCESS {
            return None;
        }
        Some(temp as f64)
    }

    /// CUDA compute capability as a version string (e.g. "8.0").
    pub fn cuda_compute_capability(&self, session: &NvmlSession) -> Option<String> {
        let mut major: c_int = 0;
        let mut minor: c_int = 0;
        let ret =
            (session.fns.device_get_cuda_compute_capability)(self.handle, &mut major, &mut minor);
        if ret != NVML_SUCCESS {
            return None;
        }
        Some(format!("{}.{}", major, minor))
    }
}

/// Memory info wrapper.
pub struct NvmlMemoryInfo {
    pub total_bytes: u64,
    #[allow(dead_code)]
    pub free_bytes: u64,
    pub used_bytes: u64,
}

impl NvmlMemoryInfo {
    pub fn total_mb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0)
    }
    pub fn used_mb(&self) -> f64 {
        self.used_bytes as f64 / (1024.0 * 1024.0)
    }
}
