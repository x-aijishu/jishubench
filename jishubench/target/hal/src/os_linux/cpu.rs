//! Linux system-wide CPU utilisation via `/proc/stat`.

#[derive(Clone, Copy)]
pub(super) struct CpuTicks {
    total: u64,
    idle: u64,
}

pub(super) fn probe() -> bool {
    read_ticks().is_some()
}

pub(super) fn read_cpu_pct(state: &mut Option<CpuTicks>) -> Option<f64> {
    let current = read_ticks()?;
    let previous = state.replace(current)?;
    utilization(previous, current)
}

fn read_ticks() -> Option<CpuTicks> {
    let content = std::fs::read_to_string("/proc/stat").ok()?;
    parse_cpu_line(content.lines().next()?)
}

fn parse_cpu_line(line: &str) -> Option<CpuTicks> {
    let mut fields = line.split_whitespace();
    if fields.next()? != "cpu" {
        return None;
    }

    let values: Vec<u64> = fields
        .take(8)
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    if values.len() < 4 {
        return None;
    }

    Some(CpuTicks {
        total: values.iter().sum(),
        idle: values[3] + values.get(4).copied().unwrap_or(0),
    })
}

fn utilization(previous: CpuTicks, current: CpuTicks) -> Option<f64> {
    let total = current.total.checked_sub(previous.total)?;
    if total == 0 {
        return None;
    }
    let idle = current.idle.saturating_sub(previous.idle).min(total);
    Some((total - idle) as f64 / total as f64 * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_aggregate_cpu_ticks() {
        let ticks = parse_cpu_line("cpu  100 20 30 400 50 6 7 8 0 0").unwrap();
        assert_eq!(ticks.total, 621);
        assert_eq!(ticks.idle, 450);
    }

    #[test]
    fn computes_system_utilization() {
        let previous = CpuTicks {
            total: 100,
            idle: 60,
        };
        let current = CpuTicks {
            total: 200,
            idle: 80,
        };
        assert_eq!(utilization(previous, current), Some(80.0));
    }
}
