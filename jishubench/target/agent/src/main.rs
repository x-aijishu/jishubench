//! agent — hardware monitoring sidecar for jishubench.
//!
//! Exposes `/v1/eval/*` compatible with the Host Python `EvalMonitorClient`.

mod routes;

use std::{net::SocketAddr, sync::Arc};

use anyhow::Context;
use axum::{routing::get, routing::post, Router};
use clap::Parser;
use hal::{platform_profiler, HardwareProfiler};
use monitor::TraceStore;
use tower_http::trace::TraceLayer;
use tracing::info;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "edge-eval-agent",
    version,
    about = "Hardware monitoring sidecar for jishubench (Target side)"
)]
struct Cli {
    /// Listen address (host:port).
    #[arg(long, env = "EEA_BIND", default_value = "127.0.0.1:9090")]
    bind: SocketAddr,

    /// Hardware sampling rate in Hz (1–1000).
    #[arg(long, env = "EEA_SAMPLE_HZ", default_value_t = 1)]
    sample_hz: u32,

    /// Maximum hardware samples retained per trace (unlimited when omitted).
    #[arg(long, env = "EEA_MAX_SAMPLES")]
    max_samples: Option<usize>,

    /// Health-check mode: print capabilities JSON and exit.
    #[arg(long)]
    health_check: bool,

    /// Path to a profiler plugin shared library (.so) for extended metrics.
    /// Requires the agent to be compiled with `--features plugin`.
    #[cfg(feature = "plugin")]
    #[arg(long, env = "EEA_PROFILER_PLUGIN")]
    profiler_plugin: Option<std::path::PathBuf>,
}

// ---------------------------------------------------------------------------
// App state shared across route handlers via Arc
// ---------------------------------------------------------------------------

pub struct AppState {
    pub store: TraceStore,
    /// Kept separately so the health handler can call profiler methods without
    /// going through TraceStore.
    pub profiler: Arc<dyn HardwareProfiler>,
    pub sample_hz: u32,
    pub sample_interval_ms: u32,
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "agent=info,tower_http=warn".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    let sample_hz = cli.sample_hz.clamp(1, 1000);
    let interval_ms = (1000 / sample_hz).max(1);

    // Build the profiler stack: builtin OS, optional NVIDIA GPU, optional plugin
    let profiler: Arc<dyn HardwareProfiler> = {
        let builtin = Arc::new(platform_profiler());

        #[cfg(feature = "nvidia")]
        let gpu: Option<Arc<dyn HardwareProfiler>> =
            Some(Arc::new(hal::nvidia::NvidiaProfiler::new()));
        #[cfg(not(feature = "nvidia"))]
        #[allow(unused_variables)]
        let gpu: Option<Arc<dyn HardwareProfiler>> = None;

        #[cfg(feature = "plugin")]
        {
            let plugin =
                cli.profiler_plugin.as_ref().and_then(
                    |path| match hal::plugin::PluginProfiler::load(path) {
                        Ok(p) => {
                            info!(path = %path.display(), "profiler plugin loaded");
                            Some(p)
                        }
                        Err(e) => {
                            tracing::warn!(
                                "failed to load profiler plugin `{}`: {e}; falling back to builtin",
                                path.display()
                            );
                            None
                        }
                    },
                );
            make_composite(builtin, gpu, plugin)
        }
        #[cfg(not(feature = "plugin"))]
        {
            #[cfg(feature = "nvidia")]
            {
                Arc::new(hal::composite::CompositeProfiler::new(builtin, gpu))
            }
            #[cfg(not(feature = "nvidia"))]
            {
                builtin as Arc<dyn HardwareProfiler>
            }
        }
    };

    if cli.health_check {
        let caps = profiler.capabilities();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "profiler":     profiler.name(),
                "capabilities": caps,
                "sample_hz":    sample_hz,
                "errors":       profiler.errors(),
            }))
            .unwrap()
        );
        return Ok(());
    }

    let store = TraceStore::new(Arc::clone(&profiler), sample_hz, cli.max_samples);
    let state = Arc::new(AppState {
        store,
        profiler,
        sample_hz,
        sample_interval_ms: interval_ms,
    });

    let app = build_router(Arc::clone(&state));

    info!(
        bind      = %cli.bind,
        sample_hz = sample_hz,
        "edge-eval-agent starting"
    );

    let listener = tokio::net::TcpListener::bind(cli.bind)
        .await
        .with_context(|| format!("failed to bind {}", cli.bind))?;

    axum::serve(listener, app).await.context("server error")?;

    Ok(())
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/eval/health", get(routes::health))
        .route("/v1/eval/traces", post(routes::create_trace))
        .route("/v1/eval/traces/:trace_id/start", post(routes::start_trace))
        .route("/v1/eval/traces/:trace_id/mark", post(routes::mark_trace))
        .route("/v1/eval/traces/:trace_id/stop", post(routes::stop_trace))
        .route("/v1/eval/traces/:trace_id", get(routes::get_trace))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Helper to build a CompositeProfiler when plugin feature is enabled.
#[cfg(feature = "plugin")]
fn make_composite(
    builtin: Arc<dyn HardwareProfiler>,
    gpu: Option<Arc<dyn HardwareProfiler>>,
    plugin: Option<hal::plugin::PluginProfiler>,
) -> Arc<dyn HardwareProfiler> {
    if gpu.is_some() || plugin.is_some() {
        Arc::new(hal::composite::CompositeProfiler::new(builtin, gpu, plugin))
    } else {
        builtin
    }
}

// ---------------------------------------------------------------------------
// Integration tests (axum in-process)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use serde_json::{json, Value};
    use tower::ServiceExt; // oneshot

    fn test_state() -> Arc<AppState> {
        let profiler: Arc<dyn HardwareProfiler> = Arc::new(platform_profiler());
        let store = TraceStore::new(Arc::clone(&profiler), 2, Some(100));
        Arc::new(AppState {
            store,
            profiler,
            sample_hz: 2,
            sample_interval_ms: 500,
        })
    }

    async fn call(
        app: Router,
        method: Method,
        uri: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let bytes = match body {
            Some(v) => serde_json::to_vec(&v).unwrap(),
            None => Vec::new(),
        };
        let req = Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(bytes))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let val: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        (status, val)
    }

    #[tokio::test]
    async fn health_returns_200_with_profiler_field() {
        let (status, body) = call(
            build_router(test_state()),
            Method::GET,
            "/v1/eval/health",
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            body.get("profiler").is_some(),
            "missing `profiler` field: {body}"
        );
        assert!(body.get("capabilities").is_some());
        assert!(body.get("sample_hz").is_some());
    }

    #[tokio::test]
    async fn full_trace_lifecycle() {
        let state = test_state();
        let trace_body = json!({
            "trace_id": "lifecycle-001",
            "benchmark": "MMMU",
            "sample_id": "mmmu_0001",
            "model": "Qwen3.5-4B-Q4_K_M",
        });

        let (s, _) = call(
            build_router(Arc::clone(&state)),
            Method::POST,
            "/v1/eval/traces",
            Some(trace_body),
        )
        .await;
        assert_eq!(s, StatusCode::CREATED);

        let (s, _) = call(
            build_router(Arc::clone(&state)),
            Method::POST,
            "/v1/eval/traces/lifecycle-001/start",
            None,
        )
        .await;
        assert_eq!(s, StatusCode::OK);

        let (s, _) = call(
            build_router(Arc::clone(&state)),
            Method::POST,
            "/v1/eval/traces/lifecycle-001/mark",
            Some(json!({"stage": "request_sent"})),
        )
        .await;
        assert_eq!(s, StatusCode::OK);

        let (s, _) = call(
            build_router(Arc::clone(&state)),
            Method::POST,
            "/v1/eval/traces/lifecycle-001/stop",
            None,
        )
        .await;
        assert_eq!(s, StatusCode::OK);

        let (s, body) = call(
            build_router(Arc::clone(&state)),
            Method::GET,
            "/v1/eval/traces/lifecycle-001",
            None,
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(body["status"], "stopped");
        assert!(body.get("aggregates").is_some());
    }

    #[tokio::test]
    async fn duplicate_create_returns_409() {
        let state = test_state();
        let body =
            json!({ "trace_id": "dup-001", "benchmark": "B", "sample_id": "s1", "model": "m1" });
        call(
            build_router(Arc::clone(&state)),
            Method::POST,
            "/v1/eval/traces",
            Some(body.clone()),
        )
        .await;
        let (s, _) = call(
            build_router(Arc::clone(&state)),
            Method::POST,
            "/v1/eval/traces",
            Some(body),
        )
        .await;
        assert_eq!(s, StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn unknown_trace_returns_404() {
        let (s, _) = call(
            build_router(test_state()),
            Method::GET,
            "/v1/eval/traces/no-such-id",
            None,
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }
}
