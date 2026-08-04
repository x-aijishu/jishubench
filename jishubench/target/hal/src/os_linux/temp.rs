//! Linux CPU temperature via hwmon sysfs nodes.

use std::{fs, path::PathBuf};

/// Find the first `temp*_input` file under `/sys/class/hwmon/hwmon*/`.
fn find_hwmon_temp_input() -> Option<PathBuf> {
    #[cfg(target_os = "linux")]
    {
        let dir = std::path::Path::new("/sys/class/hwmon");
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Ok(inner) = fs::read_dir(entry.path()) {
                    for file in inner.flatten() {
                        let name = file.file_name();
                        let s = name.to_string_lossy();
                        if s.starts_with("temp") && s.ends_with("_input") {
                            return Some(file.path());
                        }
                    }
                }
            }
        }
    }
    None
}

/// Read CPU temperature from hwmon and return degrees Celsius.
///
/// hwmon temp*_input values are in millidegrees Celsius, so we divide by 1000.
pub(super) fn probe_temp_celsius() -> Option<f64> {
    let path = find_hwmon_temp_input()?;
    let content = fs::read_to_string(&path).ok()?;
    let milli: f64 = content.trim().parse().ok()?;
    Some(milli / 1000.0)
}
