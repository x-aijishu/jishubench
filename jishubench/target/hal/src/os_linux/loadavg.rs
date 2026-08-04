//! Linux system load average via `/proc/loadavg`.

use std::fs;

/// Read system load averages (1, 5, 15 minutes) from `/proc/loadavg`.
pub(super) fn read_load_avg() -> Option<(f64, f64, f64)> {
    let content = fs::read_to_string("/proc/loadavg").ok()?;
    let parts: Vec<&str> = content.split_whitespace().collect();
    if parts.len() < 3 {
        return None;
    }
    let l1: f64 = parts[0].parse().ok()?;
    let l5: f64 = parts[1].parse().ok()?;
    let l15: f64 = parts[2].parse().ok()?;
    Some((l1, l5, l15))
}
