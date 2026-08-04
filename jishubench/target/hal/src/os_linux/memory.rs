//! Linux system-level memory measurement.

use std::fs;

/// Read system-level memory info from `/proc/meminfo`.
/// Returns (mem_total_mb, mem_available_mb, swap_total_mb, swap_used_mb).
pub(super) fn read_sysmem_mb() -> Option<(f64, f64, f64, f64)> {
    let content = fs::read_to_string("/proc/meminfo").ok()?;
    let mut total = None;
    let mut avail = None;
    let mut swap_total = None;
    let mut swap_free = None;
    for line in content.lines() {
        if let Some(kb) = parse_meminfo_line(line, "MemTotal:") {
            total = Some(kb);
        } else if let Some(kb) = parse_meminfo_line(line, "MemAvailable:") {
            avail = Some(kb);
        } else if let Some(kb) = parse_meminfo_line(line, "SwapTotal:") {
            swap_total = Some(kb);
        } else if let Some(kb) = parse_meminfo_line(line, "SwapFree:") {
            swap_free = Some(kb);
        }
    }
    match (total, avail, swap_total, swap_free) {
        (Some(t), Some(a), Some(st), Some(sf)) => {
            Some((t / 1024.0, a / 1024.0, st / 1024.0, (st - sf) / 1024.0))
        }
        _ => None,
    }
}

fn parse_meminfo_line(line: &str, prefix: &str) -> Option<f64> {
    if line.starts_with(prefix) {
        line.split_whitespace().nth(1)?.parse::<f64>().ok()
    } else {
        None
    }
}
