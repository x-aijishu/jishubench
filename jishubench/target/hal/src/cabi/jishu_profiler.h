/// @file jishu_profiler.h
/// @brief C ABI for third-party profiler plugins (chip vendors).
///
/// Vendors implement a shared library (.so) that exports a single entry point:
///
/// ```c
/// jishu_profiler_t* jishu_profiler_create(void);
/// ```
///
/// The returned vtable is called by jishubench's sampler loop.  Every
/// function pointer in the vtable MUST be non-NULL.

#ifndef JISHU_PROFILER_H
#define JISHU_PROFILER_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Per-sample hardware snapshot returned by jishu_profiler_t::sample().
/// Only dynamic fields — static device info is reported via DeviceInfo.
typedef struct {
    uint64_t t_mono_ns;     ///< monotonic ns since epoch (filled by framework)
    double   npu_util_pct;  ///< NPU util 0-100, NaN = unavailable
    double   system_cpu_util_pct; ///< system CPU util 0-100, NaN = unavailable
    double   mem_available_mb;  ///< available RAM MB, NaN = unavailable
    double   swap_used_mb;      ///< used swap MB, NaN = unavailable
    double   gpu_util_pct;      ///< GPU util 0-100, NaN = unavailable
    double   gpu_mem_used_mb;   ///< used GPU VRAM MB, NaN = unavailable
    double   temp_celsius;      ///< degrees Celsius, NaN = unavailable
} jishu_hw_sample_t;

/// Capability flags reported by jishu_profiler_t::capabilities().
typedef struct {
    bool npu_util;
    bool system_cpu_util;
    bool system_memory;
    bool gpu_util;
    bool gpu_memory;
    bool temp_celsius;
} jishu_capabilities_t;

/// Static device information, reported once (not per sample).
typedef struct {
    double mem_total_mb;        ///< total physical RAM MB, NaN = unavailable
    double swap_total_mb;       ///< total swap MB, NaN = unavailable
    double gpu_mem_total_mb;    ///< GPU VRAM total MB, NaN = unavailable
    const char* gpu_model;      ///< GPU model name, NULL = unavailable
    const char* gpu_driver_version; ///< driver version, NULL = unavailable
    const char* cuda_version;   ///< CUDA version, NULL = unavailable
} jishu_device_info_t;

// ---------------------------------------------------------------------------
// Profiler vtable
// ---------------------------------------------------------------------------

/// Opaque context pointer managed by the plugin.
typedef struct jishu_profiler {
    /// Human-readable backend name (static string).
    const char* (*name)(void* ctx);

    /// Capability flags.
    jishu_capabilities_t (*capabilities)(void* ctx);

    /// Take a single sample.  The framework fills `t_mono_ns` before calling.
    /// Fields the plugin does not support should be set to NaN.
    jishu_hw_sample_t (*sample)(void* ctx, uint64_t epoch_ns);

    /// Free all resources.  May be NULL if no cleanup is needed.
    void (*destroy)(void* ctx);

    /// Opaque plugin-internal state.
    void* ctx;
} jishu_profiler_t;

// ---------------------------------------------------------------------------
// Entry points (exported by plugin .so)
// ---------------------------------------------------------------------------

/// Create a profiler instance.
/// Returns NULL on failure (e.g. unsupported hardware, permission denied).
jishu_profiler_t* jishu_profiler_create(void);

/// Optional: return static device info for this plugin.
/// May return NULL if the plugin has no device info to report.
jishu_device_info_t* jishu_profiler_device_info(void* ctx);

#ifdef __cplusplus
}
#endif

#endif /* JISHU_PROFILER_H */
