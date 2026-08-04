//! macOS global GPU utilisation via IOKit IOGPU PerformanceStatistics.
//!
//! # Bugfix note
//!
//! The IOKit IOGPU `"Device Utilization %"` value is a **Float32** in the
//! CFNumber.  Previously the code read it as `kCFNumberSInt64Type` (which was
//! also defined with the *wrong* constant — `12` is actually `kCFNumberFloatType`,
//! not `kCFNumberSInt64Type` = 4).  This caused `CFNumberGetValue` to write 4
//! float32 bytes into an 8-byte `i64` buffer without type conversion, producing
//! the raw float32 bit pattern reinterpreted as an integer (e.g. 1.0 % → 1065353216).
//!
//! The fix: read as `kCFNumberFloat64Type (6)` so CFNumberGetValue performs
//! a proper float32→float64 conversion and the result is a usable 0–100 value.

use core_foundation::base::TCFType;
use core_foundation::string::CFString;
use std::ffi::c_void;

// ---------------------------------------------------------------------------
// Raw CoreFoundation FFI declarations needed for dictionary traversal.
// We avoid the generic CFDictionary<CFString, CFType> wrapper because it
// doesn't implement ConcreteCFType for arbitrary type parameters.
// ---------------------------------------------------------------------------

type CFTypeID = usize;
type CFIndex = isize;
type CFNumberType = CFIndex;

#[allow(non_upper_case_globals)]
const kCFNumberFloat64Type: CFNumberType = 6;

extern "C" {
    fn CFDictionaryGetValue(theDict: *const c_void, key: *const c_void) -> *const c_void;
    fn CFGetTypeID(cf: *const c_void) -> CFTypeID;
    fn CFNumberGetTypeID() -> CFTypeID;
    fn CFNumberGetValue(number: *const c_void, theType: CFNumberType, valuePtr: *mut c_void) -> u8; // Boolean (unsigned char)
    fn CFRelease(cf: *const c_void);
}

/// Read global GPU utilisation percentage via IOKit IOGPU.
///
/// Data source: IOKit `IOGPU` -> `PerformanceStatistics` -> `"Device Utilization %"`
/// This returns the **system-wide** GPU load, not per-process.
/// Works on Apple Silicon and Intel Macs with a GPU. Returns `None` if
/// the IOGPU service is not available or the query fails.
///
/// No `sudo` required – same interface used by htop and Activity Monitor.
pub(super) fn read_gpu_util_pct() -> Option<f64> {
    unsafe {
        // kIOMasterPortDefault is just MACH_PORT_NULL (0)
        let service = IOServiceGetMatchingService(0, IOServiceMatching(c"IOGPU".as_ptr()));
        if service == 0 {
            return None;
        }

        let mut props: CFDictionaryRef = std::ptr::null();
        // kCFAllocatorDefault is NULL
        let ret = IORegistryEntryCreateCFProperties(service, &mut props, std::ptr::null(), 0);
        IOObjectRelease(service);

        if ret != libc::KERN_SUCCESS || props.is_null() {
            return None;
        }

        // Look up PerformanceStatistics dictionary.
        let stats_key = CFString::new("PerformanceStatistics");
        let stats_raw = CFDictionaryGetValue(props, stats_key.as_CFTypeRef());
        if stats_raw.is_null() {
            CFRelease(props);
            return None;
        }

        // Look up "Device Utilization %" inside the stats dictionary.
        let util_key = CFString::new("Device Utilization %");
        let util_cf = CFDictionaryGetValue(stats_raw, util_key.as_CFTypeRef());
        if util_cf.is_null() {
            CFRelease(props);
            return None;
        }

        // Verify it's a CFNumber and read as Float64 (double).
        // IOKit stores this value as Float32; CFNumberGetValue handles the
        // conversion to Float64 transparently.
        if CFGetTypeID(util_cf) != CFNumberGetTypeID() {
            CFRelease(props);
            return None;
        }

        let mut val: f64 = 0.0;
        let ok = CFNumberGetValue(
            util_cf,
            kCFNumberFloat64Type,
            &mut val as *mut f64 as *mut c_void,
        );
        CFRelease(props);

        if ok != 0 {
            Some(val)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// IOKit FFI declarations + helper type aliases
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
type CFDictionaryRef = *const std::ffi::c_void;
#[allow(non_camel_case_types)]
type io_service_t = libc::c_uint;
#[allow(non_camel_case_types)]
type mach_port_t = libc::c_uint;
#[allow(non_camel_case_types)]
type CFAllocatorRef = *const std::ffi::c_void;
#[allow(non_camel_case_types)]
type CFMutableDictionaryRef = *const std::ffi::c_void;
#[allow(non_camel_case_types)]
type IOOptionBits = u32;

#[link(name = "IOKit", kind = "framework")]
extern "C" {
    fn IOServiceGetMatchingService(
        masterPort: mach_port_t,
        matching: CFMutableDictionaryRef,
    ) -> io_service_t;

    fn IOServiceMatching(name: *const libc::c_char) -> CFMutableDictionaryRef;

    fn IORegistryEntryCreateCFProperties(
        entry: io_service_t,
        properties: *mut CFDictionaryRef,
        allocator: CFAllocatorRef,
        options: IOOptionBits,
    ) -> libc::c_int;

    fn IOObjectRelease(object: io_service_t) -> libc::c_int;
}
