//! Trace state machine and in-memory store.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use chrono::Utc;
use dashmap::DashMap;
use parking_lot::Mutex;
use quanta::{Clock, Instant};
use tracing::{debug, warn};

use hal::HardwareProfiler;

use crate::{
    aggregate,
    api_types::{
        AckResponse, Aggregates, CreateTraceRequest, MarkRecord, MarkRequest, SampleRecord,
        Timeseries, TraceResponse,
    },
    error::CoreError,
};

// ---------------------------------------------------------------------------
// TraceStatus
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceStatus {
    Created,
    Running,
    Stopped,
}

impl TraceStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Running => "running",
            Self::Stopped => "stopped",
        }
    }
}

// ---------------------------------------------------------------------------
// SamplerHandle — owned by TraceData while the sampler is running
// ---------------------------------------------------------------------------

struct SamplerHandle {
    stop_flag: Arc<AtomicBool>,
    /// Buffer filled by the background sampler thread; drained on stop.
    shared: Arc<Mutex<Vec<SampleRecord>>>,
}

// ---------------------------------------------------------------------------
// TraceData
// ---------------------------------------------------------------------------

struct TraceData {
    req: CreateTraceRequest,
    status: TraceStatus,
    marks: Vec<MarkRecord>,
    samples: Vec<SampleRecord>,
    started_ns: Option<u64>,
    stopped_ns: Option<u64>,
    aggregates: Option<Aggregates>,
    /// Present only while status == Running.
    sampler: Option<SamplerHandle>,
}

// ---------------------------------------------------------------------------
// TraceStore
// ---------------------------------------------------------------------------

pub struct TraceStore {
    map: DashMap<String, Mutex<TraceData>>,
    clock: Clock,
    /// Epoch used to compute relative monotonic timestamps for all marks and
    /// samples within this agent's lifetime.
    epoch: Instant,
    profiler: Arc<dyn HardwareProfiler>,
    sample_interval: Duration,
    max_samples: Option<usize>,
}

impl TraceStore {
    pub fn new(
        profiler: Arc<dyn HardwareProfiler>,
        sample_hz: u32,
        max_trace_samples: Option<usize>,
    ) -> Self {
        let clock = Clock::new();
        let epoch = clock.now();
        let interval_us = 1_000_000u64 / sample_hz.max(1) as u64;
        Self {
            map: DashMap::new(),
            clock,
            epoch,
            profiler,
            sample_interval: Duration::from_micros(interval_us),
            max_samples: max_trace_samples,
        }
    }

    /// Monotonic nanoseconds relative to the store epoch.
    fn now_ns(&self) -> u64 {
        self.clock.now().duration_since(self.epoch).as_nanos() as u64
    }

    /// Expose profiler so the health route can build its response.
    pub fn profiler(&self) -> &dyn HardwareProfiler {
        self.profiler.as_ref()
    }

    // -----------------------------------------------------------------------
    // POST /v1/eval/traces
    // -----------------------------------------------------------------------

    pub fn create(&self, req: CreateTraceRequest) -> Result<AckResponse, CoreError> {
        let id = req.trace_id.clone();
        if self.map.contains_key(&id) {
            return Err(CoreError::Conflict(id));
        }
        self.map.insert(
            id.clone(),
            Mutex::new(TraceData {
                req,
                status: TraceStatus::Created,
                marks: Vec::new(),
                samples: Vec::new(),
                started_ns: None,
                stopped_ns: None,
                aggregates: None,
                sampler: None,
            }),
        );
        debug!(trace_id = %id, "trace created");
        Ok(AckResponse {
            trace_id: id,
            status: "created".into(),
        })
    }

    // -----------------------------------------------------------------------
    // POST /v1/eval/traces/{id}/start
    // -----------------------------------------------------------------------

    pub fn start(&self, trace_id: &str) -> Result<AckResponse, CoreError> {
        let entry = self
            .map
            .get(trace_id)
            .ok_or_else(|| CoreError::NotFound(trace_id.into()))?;
        let mut data = entry.lock();

        match data.status {
            TraceStatus::Running => return Err(CoreError::Conflict(trace_id.into())),
            TraceStatus::Stopped => {
                return Err(CoreError::BadRequest("trace already stopped".into()))
            }
            TraceStatus::Created => {}
        }

        let now = self.now_ns();
        data.status = TraceStatus::Running;
        data.started_ns = Some(now);

        // Set up the stop flag and sample buffer shared with the sampler thread.
        let stop_flag = Arc::new(AtomicBool::new(false));
        let shared: Arc<Mutex<Vec<SampleRecord>>> = Arc::new(Mutex::new(Vec::new()));

        let stop_clone = Arc::clone(&stop_flag);
        let shared_clone = Arc::clone(&shared);
        let profiler = Arc::clone(&self.profiler);
        let clock = self.clock.clone();
        let epoch = self.epoch;
        let interval = self.sample_interval;
        let max_samples = self.max_samples;
        let id_str = trace_id.to_owned();

        thread::Builder::new()
            .name(format!("sampler-{trace_id}"))
            .spawn(move || {
                debug!(trace_id = %id_str, "sampler started");
                let mut count = 0usize;
                loop {
                    if stop_clone.load(Ordering::Relaxed) {
                        break;
                    }
                    thread::sleep(interval);
                    if stop_clone.load(Ordering::Relaxed) {
                        break;
                    }
                    if let Some(limit) = max_samples {
                        if count >= limit {
                            warn!(
                                trace_id = %id_str,
                                max_samples = limit,
                                "sample limit reached; stopping sampler"
                            );
                            break;
                        }
                    }
                    let hw = profiler.sample(&clock, epoch);
                    shared_clone.lock().push(SampleRecord {
                        t_mono_ns: hw.t_mono_ns,
                        npu_util_pct: hw.npu_util_pct,
                        system_cpu_util_pct: hw.system_cpu_util_pct,
                        mem_available_mb: hw.mem_available_mb,
                        swap_used_mb: hw.swap_used_mb,
                        gpu_util_pct: hw.gpu_util_pct,
                        gpu_mem_used_mb: hw.gpu_mem_used_mb,
                        temp_celsius: hw.temp_celsius,
                        load_avg_1m: hw.load_avg_1m,
                    });
                    count += 1;
                }
                debug!(trace_id = %id_str, samples = count, "sampler finished");
            })
            .expect("failed to spawn sampler thread");

        data.sampler = Some(SamplerHandle { stop_flag, shared });

        Ok(AckResponse {
            trace_id: trace_id.into(),
            status: "running".into(),
        })
    }

    // -----------------------------------------------------------------------
    // POST /v1/eval/traces/{id}/mark
    // -----------------------------------------------------------------------

    pub fn mark(&self, trace_id: &str, req: MarkRequest) -> Result<AckResponse, CoreError> {
        let entry = self
            .map
            .get(trace_id)
            .ok_or_else(|| CoreError::NotFound(trace_id.into()))?;
        let mut data = entry.lock();
        if data.status == TraceStatus::Stopped {
            return Err(CoreError::BadRequest("trace already stopped".into()));
        }
        let now = self.now_ns();
        data.marks.push(MarkRecord {
            stage: req.stage,
            t_mono_ns: now,
            t_wall: Utc::now(),
            extra: req.extra,
        });
        Ok(AckResponse {
            trace_id: trace_id.into(),
            status: data.status.as_str().into(),
        })
    }

    // -----------------------------------------------------------------------
    // POST /v1/eval/traces/{id}/stop
    //
    // Three-phase design to avoid holding the DashMap shard lock during sleep:
    //   Phase 1 – validate state and take the SamplerHandle (drops shard lock)
    //   Phase 2 – signal the sampler and drain its buffer  (no locks held)
    //   Phase 3 – re-acquire shard lock and finalise trace data
    // -----------------------------------------------------------------------

    pub fn stop(&self, trace_id: &str) -> Result<AckResponse, CoreError> {
        // Phase 1
        let handle = {
            let entry = self
                .map
                .get(trace_id)
                .ok_or_else(|| CoreError::NotFound(trace_id.into()))?;
            let mut data = entry.lock();
            match data.status {
                TraceStatus::Created => {
                    return Err(CoreError::BadRequest("trace not started".into()))
                }
                TraceStatus::Stopped => {
                    return Err(CoreError::BadRequest("trace already stopped".into()))
                }
                TraceStatus::Running => {}
            }
            data.sampler.take()
            // `entry` (shard lock) dropped here
        };

        // Phase 2 — no shard lock held
        let samples = if let Some(h) = handle {
            h.stop_flag.store(true, Ordering::Relaxed);
            // Give the sampler thread one extra interval + small margin to notice.
            thread::sleep(self.sample_interval + Duration::from_millis(10));
            h.shared.lock().clone()
        } else {
            Vec::new()
        };

        // Phase 3
        let entry = self
            .map
            .get(trace_id)
            .ok_or_else(|| CoreError::NotFound(trace_id.into()))?;
        let mut data = entry.lock();
        let stopped_ns = self.now_ns();
        data.status = TraceStatus::Stopped;
        data.stopped_ns = Some(stopped_ns);
        data.samples = samples;
        let duration_ns = stopped_ns.saturating_sub(data.started_ns.unwrap_or(stopped_ns));
        data.aggregates = Some(aggregate::compute(duration_ns, &data.samples));

        Ok(AckResponse {
            trace_id: trace_id.into(),
            status: "stopped".into(),
        })
    }

    // -----------------------------------------------------------------------
    // GET /v1/eval/traces/{id}
    // -----------------------------------------------------------------------

    pub fn get(&self, trace_id: &str, interval_ms: u32) -> Result<TraceResponse, CoreError> {
        let entry = self
            .map
            .get(trace_id)
            .ok_or_else(|| CoreError::NotFound(trace_id.into()))?;
        let data = entry.lock();

        let aggregates = data.aggregates.clone().unwrap_or_else(|| {
            let dur = match (data.started_ns, data.stopped_ns) {
                (Some(s), Some(e)) => e.saturating_sub(s),
                _ => 0,
            };
            aggregate::compute(dur, &data.samples)
        });

        Ok(TraceResponse {
            trace_id: trace_id.into(),
            benchmark: data.req.benchmark.clone(),
            sample_id: data.req.sample_id.clone(),
            model: data.req.model.clone(),
            phase: data.req.phase.clone(),
            status: data.status.as_str().into(),
            profiler: self.profiler.name().into(),
            marks: data.marks.clone(),
            timeseries: Timeseries {
                interval_ms,
                samples: data.samples.clone(),
            },
            aggregates,
            capabilities: self.profiler.capabilities(),
            device_info: self.profiler.device_info(),
            errors: self.profiler.errors(),
        })
    }

    /// Remove a trace (future retention cleanup hook).
    pub fn remove(&self, trace_id: &str) {
        self.map.remove(trace_id);
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use hal::{Capabilities, DeviceInfo, HardwareSample};

    struct NoopProfiler;

    impl HardwareProfiler for NoopProfiler {
        fn name(&self) -> &'static str {
            "noop"
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                npu_util: false,
                system_cpu_util: false,
                system_memory: false,
                gpu_util: false,
                gpu_memory: false,
                temp_celsius: false,
                load_avg: false,
            }
        }
        fn device_info(&self) -> DeviceInfo {
            DeviceInfo::default()
        }
        fn sample(&self, clock: &Clock, epoch: Instant) -> HardwareSample {
            HardwareSample {
                t_mono_ns: clock.now().duration_since(epoch).as_nanos() as u64,
                npu_util_pct: None,
                system_cpu_util_pct: None,
                mem_available_mb: None,
                swap_used_mb: None,
                gpu_util_pct: None,
                gpu_mem_used_mb: None,
                temp_celsius: None,
                load_avg_1m: None,
                load_avg_5m: None,
                load_avg_15m: None,
            }
        }
        fn errors(&self) -> Vec<String> {
            vec![]
        }
    }

    fn make_store() -> TraceStore {
        TraceStore::new(Arc::new(NoopProfiler), 20, Some(1200))
    }

    fn req(id: &str) -> CreateTraceRequest {
        CreateTraceRequest {
            trace_id: id.into(),
            benchmark: "MMMU".into(),
            sample_id: "s001".into(),
            model: "test-model".into(),
            phase: "measured".into(),
            extra: HashMap::new(),
        }
    }

    #[test]
    fn happy_path_lifecycle() {
        let store = make_store();
        store.create(req("t1")).unwrap();
        store.start("t1").unwrap();
        store
            .mark(
                "t1",
                MarkRequest {
                    stage: "request_sent".into(),
                    extra: HashMap::new(),
                },
            )
            .unwrap();
        store
            .mark(
                "t1",
                MarkRequest {
                    stage: "response_received".into(),
                    extra: HashMap::new(),
                },
            )
            .unwrap();
        store.stop("t1").unwrap();
        let resp = store.get("t1", 50).unwrap();
        assert_eq!(resp.status, "stopped");
        assert_eq!(resp.marks.len(), 2);
        assert!(resp.aggregates.duration_ms >= 0.0);
    }

    #[test]
    fn duplicate_create_is_conflict() {
        let store = make_store();
        store.create(req("t2")).unwrap();
        assert!(matches!(
            store.create(req("t2")).unwrap_err(),
            CoreError::Conflict(_)
        ));
    }

    #[test]
    fn stop_before_start_is_bad_request() {
        let store = make_store();
        store.create(req("t3")).unwrap();
        assert!(matches!(
            store.stop("t3").unwrap_err(),
            CoreError::BadRequest(_)
        ));
    }

    #[test]
    fn get_unknown_is_not_found() {
        let store = make_store();
        assert!(matches!(
            store.get("x", 50).unwrap_err(),
            CoreError::NotFound(_)
        ));
    }

    #[test]
    fn start_running_trace_is_conflict() {
        let store = make_store();
        store.create(req("t4")).unwrap();
        store.start("t4").unwrap();
        assert!(matches!(
            store.start("t4").unwrap_err(),
            CoreError::Conflict(_)
        ));
        // clean up
        store.stop("t4").unwrap();
    }
}
