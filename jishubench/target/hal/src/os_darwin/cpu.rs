//! macOS system-wide CPU utilisation via Mach host statistics.

use std::mem::MaybeUninit;

const HOST_CPU_LOAD_INFO: libc::c_int = 3;
const CPU_STATE_USER: usize = 0;
const CPU_STATE_SYSTEM: usize = 1;
const CPU_STATE_IDLE: usize = 2;
const CPU_STATE_NICE: usize = 3;

#[repr(C)]
struct HostCpuLoadInfo {
    cpu_ticks: [libc::c_uint; 4],
}

#[derive(Clone, Copy)]
pub(super) struct CpuTicks {
    total: u64,
    idle: u64,
}

extern "C" {
    fn host_statistics(
        host: libc::mach_port_t,
        flavor: libc::c_int,
        info: *mut libc::c_void,
        count: *mut libc::c_uint,
    ) -> libc::c_int;
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
    let mut info = MaybeUninit::<HostCpuLoadInfo>::uninit();
    let mut count = 4;
    #[allow(deprecated)]
    let host = unsafe { libc::mach_host_self() };
    let result = unsafe {
        host_statistics(
            host,
            HOST_CPU_LOAD_INFO,
            info.as_mut_ptr().cast(),
            &mut count,
        )
    };
    if result != libc::KERN_SUCCESS {
        return None;
    }

    let ticks = unsafe { info.assume_init() }.cpu_ticks;
    let idle = ticks[CPU_STATE_IDLE] as u64;
    let total = ticks[CPU_STATE_USER] as u64
        + ticks[CPU_STATE_SYSTEM] as u64
        + idle
        + ticks[CPU_STATE_NICE] as u64;
    Some(CpuTicks { total, idle })
}

fn utilization(previous: CpuTicks, current: CpuTicks) -> Option<f64> {
    let total = current.total.checked_sub(previous.total)?;
    if total == 0 {
        return None;
    }
    let idle = current.idle.saturating_sub(previous.idle).min(total);
    Some((total - idle) as f64 / total as f64 * 100.0)
}
