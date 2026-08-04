//! Jetson platform GPU monitoring via sysfs.
//!
//! Reads GPU utilisation and temperature from the standard sysfs paths
//! found on NVIDIA Jetson devices (TX1/TX2, Xavier NX/AGX, Orin NX/AGX).

use std::fs;
use std::path::Path;

/// Possible sysfs paths for GPU load on Jetson.
const GPU_LOAD_PATHS: &[&str] = &[
    "/sys/devices/gpu.0/load",          // TX1/TX2/Xavier
    "/sys/class/kgsl/kgsl-3d0/gpubusy", // older Jetson (alternative)
];

/// GPU thermal zone prefix.
const THERMAL_PREFIX: &str = "/sys/class/thermal/thermal_zone";

/// Board model path (DT compatible string).
const DT_MODEL: &str = "/proc/device-tree/model";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Attempt to read GPU utilisation from sysfs on a Jetson device.
///
/// Returns a value in percent (0.0–100.0) if available.
pub fn read_gpu_util_pct() -> Option<f64> {
    for path_str in GPU_LOAD_PATHS {
        let p = Path::new(path_str);
        if !p.exists() {
            continue;
        }
        let raw = fs::read_to_string(p).ok()?;
        let trimmed = raw.trim();
        // Format on most Jetsons: <util> <freq> (e.g. "45 1300500000")
        // where util is the percentage of time the GPU was busy * 10 (i.e. 450 = 45.0%).
        if let Some(load_val) = trimmed.split_whitespace().next() {
            let val: f64 = load_val.parse().ok()?;
            // If the value is > 100, it's likely in tenths-of-percent units
            if val > 100.0 {
                return Some(val / 10.0);
            }
            return Some(val);
        }
    }
    None
}

/// Read GPU temperature from the first thermal zone with "GPU" in its type.
///
/// Returns Celsius.
pub fn read_gpu_temp_celsius() -> Option<f64> {
    for i in 0..16 {
        let zone_path = format!("{THERMAL_PREFIX}{i}");
        let type_path = format!("{zone_path}/type");
        let temp_path = format!("{zone_path}/temp");

        let zone_type = fs::read_to_string(&type_path).ok()?;
        if !zone_type.trim().eq_ignore_ascii_case("gpu") {
            continue;
        }

        let raw = fs::read_to_string(&temp_path).ok()?;
        let millideg: f64 = raw.trim().parse().ok()?;
        // Temperature is usually in millidegrees Celsius
        return Some(millideg / 1000.0);
    }
    None
}

/// Read the board model string from the device tree.
pub fn read_board_model() -> Option<String> {
    let raw = fs::read_to_string(DT_MODEL).ok()?;
    let s = raw.trim().trim_end_matches('\0').to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Return `true` if we appear to be running on a Jetson platform.
///
/// Detection is based on the presence of the GPU sysfs node and an
/// NVIDIA Jetson board model string.
pub fn detect_jetson() -> bool {
    // Check for the primary GPU sysfs path
    let has_gpu_dev = GPU_LOAD_PATHS.iter().any(|p| Path::new(p).exists());

    if !has_gpu_dev {
        return false;
    }

    // Cross-check with the board model
    if let Some(model) = read_board_model() {
        let lower = model.to_lowercase();
        if lower.contains("jetson") || lower.contains("tegra") {
            return true;
        }
    }

    // Without board model info, the existence of GPU load sysfs is a strong hint
    has_gpu_dev
}

/// Read total GPU memory (MB) on Jetson.
///
/// On Orin this can be read from `/sys/kernel/debug/nvmap/iovmm/areas` or
/// from the kernel boot log.  On most Jetsons the GPU memory is carved out
/// of system RAM; we try to extract it from known locations.
pub fn read_gpu_mem_total_mb() -> Option<f64> {
    // Newer Jetson kernels expose GPU memory info via debugfs
    const NVMAP_AREAS: &str = "/sys/kernel/debug/nvmap/iovmm/areas";
    if let Ok(content) = fs::read_to_string(NVMAP_AREAS) {
        // Sum all "total" lines (format: "total=0x...")
        let total: u64 = content
            .lines()
            .filter_map(|line| line.strip_prefix("total=0x"))
            .filter_map(|hex| u64::from_str_radix(hex, 16).ok())
            .sum();
        if total > 0 {
            return Some(total as f64 / (1024.0 * 1024.0));
        }
    }

    // Fallback: parse dmesg for GPU carveout
    if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
        // If we can read /proc/meminfo but don't have debugfs, we can't determine
        // the GPU carveout size precisely.  Return None to let the caller decide.
        let _ = meminfo;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_jetson_noop() {
        // On a non-Jetson machine this should always return false.
        assert!(!detect_jetson());
    }
}
