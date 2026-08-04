# Architecture

jishubench uses a **Host–Target** split so benchmark scoring and hardware sampling stay on separate channels and the Target inference path stays clean.

## Component diagram

```
Host (jishubench)
  ├─ HostController → LlamaCppClient
  │    ├─ LlamaCppControlPlane: /health /models /props (control plane)
  │    └─ OpenAI SDK:           /v1/models reachability check (data plane)
    ├─ LmmsEvalRunner → simple_evaluate()
       ├─ TaskManager: load benchmark YAML + HF dataset
       ├─ async_openai: POST /v1/chat/completions → Target
       └─ metrics: score on Host (Judge via OPENAI_API_URL if needed)
    └─ Agent benchmark runners
      ├─ TauBenchRunner: in-process user/agent simulation
      ├─ TerminalBenchRunner: tb Docker task harness
      ├─ TerminalBench2Runner: Harbor task harness
      └─ ClawEvalRunner: run-local config + upstream batch handler

Target (edge board)
  ├─ llama-server router (no -m; models loaded on demand)
  └─ edge-eval-agent sidecar (optional) — hardware metrics at :9090
```

## Responsibilities

| Layer | Runs on | Role |
|-------|---------|------|
| **lmms-eval** | Host | Benchmark tasks, dataset loading, scoring |
| **Terminal-Bench 1** | Host | Legacy `tb` Docker sandbox agent harness |
| **Terminal-Bench 2.0** | Host | Harbor task harness |
| **Tau-bench** | Host | In-process tool-agent-user simulation |
| **Claw-Eval** | Host | Autonomous agent harness, mock services, sandbox, judge, traces |
| **Host controller** | Host | Preflight, Target lifecycle, run orchestration |
| **Inference server** | Target | OpenAI-compatible chat completions (agent) |
| **Remote user LLM** | Remote API | User simulator for tau-bench |
| **edge-eval-agent** | Target | Optional `/v1/eval/*` hardware traces |

Hardware monitoring (`edge-eval-agent`) is a separate sidecar channel. It is not wired into the per-sample scoring loop today (Phase B TTFT tracing is planned).

## Directory layout

```
jishubench/
├── jishubench/
│   ├── host/
│   │   ├── main.py                 # jishubench CLI
│   │   ├── controller.py           # preflight, lifecycle, traces, manifests
│   │   ├── cli/                    # parser and command handlers
│   │   ├── client/                 # llama.cpp and monitor clients
│   │   ├── config/                 # settings, profiles, env, logging, submodules
│   │   └── adapters/
│   │       ├── __init__.py          # Runner protocol and shared result types
│   │       ├── lmms_eval.py        # LmmsEvalRunner
│   │       ├── tau_bench.py        # TauBenchRunner + LLM patch
│   │       ├── terminal_bench.py   # TerminalBenchRunner (tb v0.1.1)
│   │       ├── terminal_bench_2.py # TerminalBench2Runner (Harbor)
│   │       ├── claw_eval.py        # ClawEvalRunner + trace archival
│   │       └── utils.py            # shared adapter preflight helpers
│   └── target/                     # edge-eval-agent Rust workspace
│       ├── agent/                  # HTTP API
│       ├── hal/                    # hardware abstraction layer
│       └── monitor/                # metric probes
├── submodules/
│   ├── lmms-eval/                  # Vision-language benchmarks
│   ├── tau2-bench/                 # Tool-agent-user simulation tasks
│   └── claw-eval/                  # Autonomous agent tasks, fixtures, sandbox
├── configs/
│   ├── base.yaml                   # shared defaults
│   └── devices/                    # device profiles
├── docs/
└── tests/                          # one test module per component/backend
```

## Further reading

- [Strategic design document](design/benchmark_infra_new.md) (Chinese, internal)
- [edge-eval-agent](edge-eval-agent/index.md) — Rust sidecar API and build
- [llama.cpp router](deployment/llama-cpp-router.md) — model load lifecycle
