# CLI & API Reference

## CLI reference

```
edge-eval-agent [OPTIONS]
```

| Option | Env var | Default | Description |
|--------|---------|---------|-------------|
| `--bind <ADDR>` | `EEA_BIND` | `127.0.0.1:9090` | Listen address |
| `--sample-hz <HZ>` | `EEA_SAMPLE_HZ` | `2` | Hardware sampling rate (1–1000) |
| `--max-samples <N>` | `EEA_MAX_SAMPLES` | `1200` | Max samples per trace (OOM guard) |
| `--health-check` | -- | flag | Print capabilities JSON and exit |
| `--profiler-plugin <PATH>` | `EEA_PROFILER_PLUGIN` | optional | Path to a `.so` profiler plugin (requires `plugin` feature) |
| `-h`, `--help` | -- | -- | Print help |
| `-V`, `--version` | -- | -- | Print version |

### Build features

| Feature | Command | Default | Description |
|---------|---------|---------|-------------|
| `nvidia` | `cargo build --features nvidia` | off | NVIDIA GPU monitoring (NVML + Jetson sysfs) |
| `plugin` | `cargo build --features plugin` | off | `.so` profiler plugin loading |
| `nvidia+plugin` | `cargo build --features nvidia,plugin` | off | Both GPU and plugin support |

## API reference

All endpoints are mounted under `/v1/eval/`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/eval/health` | Profiler name, capabilities, device info, sample rate, errors |
| POST | `/v1/eval/traces` | Create a trace |
| POST | `/v1/eval/traces/{id}/start` | Start background sampling |
| POST | `/v1/eval/traces/{id}/mark` | Record a named timestamp event |
| POST | `/v1/eval/traces/{id}/stop` | Stop sampling and compute aggregates |
| GET | `/v1/eval/traces/{id}` | Retrieve the full trace report |

### HTTP status codes

| Status | Meaning |
|--------|---------|
| 200 OK | Request succeeded |
| 201 Created | Trace created successfully |
| 400 Bad Request | Invalid state transition (e.g. stop before start) |
| 404 Not Found | Trace ID does not exist |
| 409 Conflict | Duplicate trace ID |

### GET /v1/eval/health

Response:

```json
{
  "profiler": "os_linux",
  "capabilities": {
    "npu_util": false,
    "system_cpu_util": true,
    "system_memory": true,
    "gpu_util": false,
    "gpu_memory": false,
    "temp_celsius": true,
    "load_avg": true
  },
  "device_info": {
    "mem_total_mb": 8192.0,
    "swap_total_mb": 1024.0,
    "cpu_model_name": "Neoverse V2",
    "cpu_core_count": 8,
    "cpu_max_freq_mhz": 2800.0,
    "board_model": "NVIDIA Jetson Orin NX 16GB",
    "arch": "aarch64",
    "gpu_mem_total_mb": null,
    "gpu_model": null,
    "gpu_driver_version": null,
    "cuda_version": null
  },
  "sample_hz": 2,
  "sample_interval_ms": 500,
  "errors": []
}
```

Fields:

| Field | Description |
|-------|-------------|
| `profiler` | Composite profiler name (e.g. `"os_linux"`, `"os_linux+nvidia"`, `"os_linux+nvidia+my_plugin"`) |
| `capabilities` | Which metrics the profiler can supply on this platform |
| `device_info` | Static device information collected once at startup (RAM, CPU, GPU, board, arch) |
| `sample_hz` | Current sampling rate in Hz |
| `sample_interval_ms` | Current sampling interval in ms |
| `errors` | Non-fatal startup diagnostic messages |

### POST /v1/eval/traces

Request body:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `trace_id` | string | yes | Unique trace identifier |
| `benchmark` | string | yes | Benchmark name (e.g. `MMMU`) |
| `sample_id` | string | yes | Sample identifier within the benchmark |
| `model` | string | yes | Model name and quantization |
| `phase` | string | no | Defaults to `"measured"`. Use `"dry_run"` for warmup traces |
| (extra) | any | no | Any additional fields forwarded to the trace report |

Response: `201 Created`

```json
{
  "trace_id": "run_20260607_mmmu_0001",
  "status": "created"
}
```

### POST /v1/eval/traces/{id}/start

No request body.

Response: `200 OK`

```json
{
  "trace_id": "run_20260607_mmmu_0001",
  "status": "running"
}
```

### POST /v1/eval/traces/{id}/mark

Request body:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `stage` | string | yes | Event name (e.g. `"request_sent"`, `"response_received"`) |
| (extra) | any | no | Any additional fields forwarded to the mark record |

Response: `200 OK`

```json
{
  "trace_id": "run_20260607_mmmu_0001",
  "status": "running"
}
```

### POST /v1/eval/traces/{id}/stop

No request body.

Response: `200 OK`

```json
{
  "trace_id": "run_20260607_mmmu_0001",
  "status": "stopped"
}
```

### GET /v1/eval/traces/{id}

No request body. Response consists of:

| Field | Description |
|-------|-------------|
| `trace_id` | Unique trace identifier |
| `benchmark` | Benchmark name |
| `sample_id` | Sample identifier |
| `model` | Model name and quantization |
| `phase` | Phase (e.g. `"measured"`, `"dry_run"`) |
| `status` | `"created"`, `"running"`, or `"stopped"` |
| `profiler` | Backend name (e.g. `"os_linux"`) |
| `marks` | Timestamped mark events |
| `timeseries` | Hardware sample array with `interval_ms` |
| `aggregates` | Computed summary statistics |
| `capabilities` | Available metrics for this run |
| `device_info` | Static device information (collected once at startup) |
| `errors` | Non-fatal diagnostic messages |

#### Example trace report

```json
{
  "trace_id": "run_20260607_mmmu_0001",
  "benchmark": "MMMU",
  "sample_id": "mmmu_0001",
  "model": "Qwen3.5-4B-Q4_K_M",
  "phase": "measured",
  "status": "stopped",
  "profiler": "os_linux",
  "marks": [
    {"stage": "request_sent",      "t_mono_ns": 1000000,   "t_wall": "2026-06-07T12:00:00.001Z"},
    {"stage": "response_received", "t_mono_ns": 850000000, "t_wall": "2026-06-07T12:00:00.850Z"}
  ],
  "timeseries": {
    "interval_ms": 500,
    "samples": [
      {
        "t_mono_ns": 5000000,
        "npu_util_pct": null,
        "system_cpu_util_pct": 45.0,
        "mem_available_mb": 4096.0,
        "swap_used_mb": 128.0,
        "gpu_util_pct": null,
        "gpu_mem_used_mb": null,
        "temp_celsius": 65.0,
        "load_avg_1m": 1.2
      }
    ]
  },
  "aggregates": {
    "duration_ms": 849.0,
    "system_cpu_util_pct": {"mean": 42.0, "max": 65.0},
    "mem_available_mb": {"mean": 4000.0, "min": 3800.0},
    "swap": {"initial_mb": 100.0, "final_mb": 128.0, "delta_mb": 28.0}
  },
  "capabilities": {
    "npu_util": false,
    "system_cpu_util": true,
    "system_memory": true,
    "gpu_util": false,
    "gpu_memory": false,
    "temp_celsius": true,
    "load_avg": true
  },
  "device_info": {
    "mem_total_mb": 8192.0,
    "swap_total_mb": 1024.0,
    "cpu_model_name": "Neoverse V2",
    "cpu_core_count": 8,
    "cpu_max_freq_mhz": 2800.0,
    "board_model": "NVIDIA Jetson Orin NX 16GB",
    "arch": "aarch64",
    "gpu_mem_total_mb": null,
    "gpu_model": null,
    "gpu_driver_version": null,
    "cuda_version": null
  },
  "errors": []
}
```

> Note: `mem_total_mb` and `swap_total_mb` are **static** fields reported once in `device_info`, not duplicated per sample. Per-sample samples contain only dynamic fields (`mem_available_mb`, `swap_used_mb`).

#### Aggregates breakdown

| Field | Struct | Fields | Description |
|-------|--------|--------|-------------|
| `system_cpu_util_pct` | `StatAggregate` | `mean`, `max` | System-wide CPU utilisation |
| `mem_available_mb` | `MemAvailableAggregate` | `mean`, `min` | System memory pressure (min = peak pressure) |
| `swap` | `SwapAggregate` | `initial_mb`, `final_mb`, `delta_mb` | Swap usage change; `delta_mb > 0` indicates swap-in |

## Environment check list

| Check | Command | Expected |
|-------|---------|----------|
| Toolchain | `rustc --version` | stable, no errors |
| Native build (default) | `cargo build -p agent` | succeeds |
| Native build (NVIDIA GPU) | `cargo build -p agent --features nvidia` | succeeds |
| Native build (plugin) | `cargo build -p agent --features plugin` | succeeds |
| Native build (all features) | `cargo build -p agent --features nvidia,plugin` | succeeds |
| Local API | `curl localhost:9090/v1/eval/health` | HTTP 200 + JSON |
| Cross-compile (aarch64) | `cargo build --release --target aarch64-unknown-linux-gnu` | succeeds |
| Cross-compile (x86_64) | `cargo build --release --target x86_64-unknown-linux-gnu` | succeeds |
| Host integration | `jishubench preflight` | `monitor_reachable: true` |
| pytest | `pytest tests/test_monitor_agent.py -v` | all pass (requires running agent) |
