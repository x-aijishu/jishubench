//! macOS system load average via `getloadavg`.

/// Read system load averages (1, 5, 15 minutes) via libc::getloadavg.
pub(super) fn read_load_avg() -> Option<(f64, f64, f64)> {
    let mut load: [f64; 3] = [0.0; 3];
    let ret = unsafe { libc::getloadavg(load.as_mut_ptr(), 3) };
    if ret != 3 {
        return None;
    }
    Some((load[0], load[1], load[2]))
}
