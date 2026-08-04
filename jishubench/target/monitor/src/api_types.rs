//! Wire types shared by the HTTP layer and the core state machine.
//! All types derive `Serialize`; request bodies additionally derive
//! `Deserialize`.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Request payloads
// ---------------------------------------------------------------------------

/// Body of `POST /v1/eval/traces`.
#[derive(Debug, Clone, Deserialize)]
pub struct CreateTraceRequest {
    pub trace_id: String,
    pub benchmark: String,
    pub sample_id: String,
    pub model: String,
    #[serde(default = "default_phase")]
    pub phase: String,
    /// Arbitrary extra fields forwarded from Host (e.g. `run_id`).
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

fn default_phase() -> String {
    "measured".into()
}

/// Body of `POST /v1/eval/traces/{id}/mark`.
#[derive(Debug, Clone, Deserialize)]
pub struct MarkRequest {
    pub stage: String,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Response bodies
// ---------------------------------------------------------------------------

/// `GET /v1/eval/health` response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub profiler: String,
    pub capabilities: hal::Capabilities,
    pub device_info: hal::DeviceInfo,
    pub sample_hz: u32,
    pub errors: Vec<String>,
}

/// A single timestamped mark event.
#[derive(Debug, Clone, Serialize)]
pub struct MarkRecord {
    pub stage: String,
    pub t_mono_ns: u64,
    pub t_wall: DateTime<Utc>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// A single hardware sample in the timeseries (dynamic fields only).
#[derive(Debug, Clone, Serialize)]
pub struct SampleRecord {
    pub t_mono_ns: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub npu_util_pct: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_cpu_util_pct: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mem_available_mb: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub swap_used_mb: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_util_pct: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_mem_used_mb: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp_celsius: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_avg_1m: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Timeseries {
    pub interval_ms: u32,
    pub samples: Vec<SampleRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StatAggregate {
    pub mean: f64,
    pub max: f64,
}

/// Aggregate of available memory over the trace.
#[derive(Debug, Clone, Serialize)]
pub struct MemAvailableAggregate {
    pub mean: f64,
    /// Minimum available memory – peak memory pressure during inference.
    pub min: f64,
}

/// Swap usage delta over the trace.
#[derive(Debug, Clone, Serialize)]
pub struct SwapAggregate {
    pub initial_mb: f64,
    pub final_mb: f64,
    /// final - initial difference.  > 0 means swap-in occurred (memory pressure).
    pub delta_mb: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Aggregates {
    pub duration_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_cpu_util_pct: Option<StatAggregate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mem_available_mb: Option<MemAvailableAggregate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub swap: Option<SwapAggregate>,
}

/// Full `GET /v1/eval/traces/{id}` response.
#[derive(Debug, Clone, Serialize)]
pub struct TraceResponse {
    pub trace_id: String,
    pub benchmark: String,
    pub sample_id: String,
    pub model: String,
    pub phase: String,
    pub status: String,
    pub profiler: String,
    pub marks: Vec<MarkRecord>,
    pub timeseries: Timeseries,
    pub aggregates: Aggregates,
    pub capabilities: hal::Capabilities,
    pub device_info: hal::DeviceInfo,
    pub errors: Vec<String>,
}

/// Generic acknowledgement for mutating endpoints.
#[derive(Debug, Serialize)]
pub struct AckResponse {
    pub trace_id: String,
    pub status: String,
}
