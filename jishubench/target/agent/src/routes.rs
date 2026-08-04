//! Axum route handlers for all `/v1/eval/*` endpoints.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use monitor::{
    api_types::{CreateTraceRequest, MarkRequest},
    CoreError,
};
use serde_json::json;

use crate::AppState;

// ---------------------------------------------------------------------------
// Error → HTTP mapping
// ---------------------------------------------------------------------------

fn core_err(err: CoreError) -> Response {
    let (status, msg) = match &err {
        CoreError::NotFound(id) => (StatusCode::NOT_FOUND, format!("trace '{id}' not found")),
        CoreError::Conflict(id) => (StatusCode::CONFLICT, format!("conflict on '{id}'")),
        CoreError::BadRequest(m) => (StatusCode::BAD_REQUEST, m.clone()),
    };
    (status, Json(json!({"error": msg}))).into_response()
}

// ---------------------------------------------------------------------------
// GET /v1/eval/health
// ---------------------------------------------------------------------------

pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let caps = state.profiler.capabilities();
    let di = state.profiler.device_info();
    let errors = state.profiler.errors();
    Json(json!({
        "profiler":           state.profiler.name(),
        "capabilities":       caps,
        "device_info":        di,
        "sample_hz":          state.sample_hz,
        "sample_interval_ms": state.sample_interval_ms,
        "errors":             errors,
    }))
}

// ---------------------------------------------------------------------------
// POST /v1/eval/traces
// ---------------------------------------------------------------------------

pub async fn create_trace(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateTraceRequest>,
) -> Response {
    match state.store.create(req) {
        Ok(ack) => (StatusCode::CREATED, Json(ack)).into_response(),
        Err(e) => core_err(e),
    }
}

// ---------------------------------------------------------------------------
// POST /v1/eval/traces/{trace_id}/start
// ---------------------------------------------------------------------------

pub async fn start_trace(
    State(state): State<Arc<AppState>>,
    Path(trace_id): Path<String>,
) -> Response {
    match state.store.start(&trace_id) {
        Ok(ack) => Json(ack).into_response(),
        Err(e) => core_err(e),
    }
}

// ---------------------------------------------------------------------------
// POST /v1/eval/traces/{trace_id}/mark
// ---------------------------------------------------------------------------

pub async fn mark_trace(
    State(state): State<Arc<AppState>>,
    Path(trace_id): Path<String>,
    Json(req): Json<MarkRequest>,
) -> Response {
    match state.store.mark(&trace_id, req) {
        Ok(ack) => Json(ack).into_response(),
        Err(e) => core_err(e),
    }
}

// ---------------------------------------------------------------------------
// POST /v1/eval/traces/{trace_id}/stop
// ---------------------------------------------------------------------------

pub async fn stop_trace(
    State(state): State<Arc<AppState>>,
    Path(trace_id): Path<String>,
) -> Response {
    match state.store.stop(&trace_id) {
        Ok(ack) => Json(ack).into_response(),
        Err(e) => core_err(e),
    }
}

// ---------------------------------------------------------------------------
// GET /v1/eval/traces/{trace_id}
// ---------------------------------------------------------------------------

pub async fn get_trace(
    State(state): State<Arc<AppState>>,
    Path(trace_id): Path<String>,
) -> Response {
    match state.store.get(&trace_id, state.sample_interval_ms) {
        Ok(resp) => Json(resp).into_response(),
        Err(e) => core_err(e),
    }
}
