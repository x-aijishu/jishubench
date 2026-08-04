//! macOS system-level memory measurement.

use std::mem::MaybeUninit;

// ---------------------------------------------------------------------------
// System-level memory via Mach / BSD interfaces.
// ---------------------------------------------------------------------------

/// Read system memory info via Mach `host_statistics64` and `sysctl`.
///
/// Returns (mem_total_mb, mem_available_mb, swap_total_mb, swap_used_mb).
///
/// Available memory is conservatively estimated as
///   (free + inactive + speculative) * page_size.
/// Swap info comes from `sysctl VM_SWAPUSAGE`.
pub(super) fn read_sysmem_mb() -> Option<(f64, f64, f64, f64)> {
    // Total physical RAM via host_info
    let total_bytes = host_max_mem()?;
    let total_mb = total_bytes as f64 / (1024.0 * 1024.0);

    // Available memory via host_statistics64
    let avail_mb = host_available_mb()?;

    // Swap via sysctl
    let (swap_total_mb, swap_used_mb) = sysctl_swap_mb()?;

    Some((total_mb, avail_mb, swap_total_mb, swap_used_mb))
}

fn host_max_mem() -> Option<u64> {
    #[repr(C)]
    struct HostBasicInfo {
        max_mem: u64,
        // Other fields omitted – we only need max_mem.
    }

    extern "C" {
        fn host_info(
            host: libc::mach_port_t,
            flavor: libc::c_int,
            info: *mut libc::c_void,
            count: *mut libc::c_uint,
        ) -> libc::c_int;
    }

    const HOST_BASIC_INFO: libc::c_int = 1; // mach_host.h
    const HOST_BASIC_INFO_COUNT: libc::c_uint = 4; // 4 u32 = 2 u64 words

    unsafe {
        let mut info = MaybeUninit::<HostBasicInfo>::uninit();
        let mut count = HOST_BASIC_INFO_COUNT;
        #[allow(deprecated)]
        let host_self = libc::mach_host_self();
        let ret = host_info(
            host_self,
            HOST_BASIC_INFO,
            info.as_mut_ptr().cast(),
            &mut count,
        );
        if ret != libc::KERN_SUCCESS {
            return None;
        }
        Some(info.assume_init().max_mem)
    }
}

fn host_available_mb() -> Option<f64> {
    // Import mach_host.h constants and types
    #[repr(C)]
    struct VmStatistics64 {
        free_count: u32,
        active_count: u32,
        inactive_count: u32,
        wired_count: u32,
        zero_fill_count: u64,
        reactivations: u64,
        pageins: u64,
        pageouts: u64,
        faults: u64,
        cow_faults: u64,
        lookups: u64,
        hits: u64,
        purges: u64,
        purgeable_count: u32,
        speculative_count: u32,
        compressor_page_count: u32,
        throttled_count: u32,
        filebacked_count: u32,
        // vm_statistics64_data_t has additional fields but we stop at the
        // standard count definition used by host_statistics64.
        pages_encrypted: u32,
        pages_decrypted: u32,
        total_uncompressed_pages_in_compressor: u64,
    }

    extern "C" {
        fn host_statistics64(
            host: libc::mach_port_t,
            flavor: libc::c_int,
            info: *mut libc::c_void,
            count: *mut libc::c_uint,
        ) -> libc::c_int;
        fn host_page_size(host: libc::mach_port_t, page_size: *mut libc::vm_size_t) -> libc::c_int;
    }

    const HOST_VM_INFO64: libc::c_int = 4;
    // Number of u32 fields in vm_statistics64_data_t up to (and including)
    // total_uncompressed_pages_in_compressor
    const COUNT: libc::c_uint = 26;

    unsafe {
        let mut stats = MaybeUninit::<VmStatistics64>::uninit();
        let mut count = COUNT;
        #[allow(deprecated)]
        let host_self = libc::mach_host_self();
        let ret = host_statistics64(
            host_self,
            HOST_VM_INFO64,
            stats.as_mut_ptr().cast(),
            &mut count,
        );
        if ret != libc::KERN_SUCCESS {
            return None;
        }
        let s = stats.assume_init();

        let mut page_size: libc::vm_size_t = 0;
        if host_page_size(host_self, &mut page_size) != libc::KERN_SUCCESS {
            return None;
        }

        let free_pages = s.free_count as u64 + s.inactive_count as u64 + s.speculative_count as u64;
        let avail_bytes = free_pages * page_size as u64;
        Some(avail_bytes as f64 / (1024.0 * 1024.0))
    }
}

fn sysctl_swap_mb() -> Option<(f64, f64)> {
    #[repr(C)]
    struct XswUsage {
        xsu_gen: u64,
        xsu_total: u64,
        xsu_avail: u64,
        xsu_used: u64,
    }

    const VM_SWAPUSAGE: libc::c_int = 35;
    const CTL_VM: libc::c_int = 2;

    let mut mib = [CTL_VM, VM_SWAPUSAGE];

    unsafe {
        let mut usage = MaybeUninit::<XswUsage>::uninit();
        let mut size = std::mem::size_of::<XswUsage>() as libc::size_t;
        let ret = libc::sysctl(
            mib.as_mut_ptr(),
            mib.len() as libc::c_uint,
            usage.as_mut_ptr().cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        );
        if ret != 0 {
            return None;
        }
        let u = usage.assume_init();
        let total_mb = u.xsu_total as f64 / (1024.0 * 1024.0);
        let used_mb = u.xsu_used as f64 / (1024.0 * 1024.0);
        Some((total_mb, used_mb))
    }
}
