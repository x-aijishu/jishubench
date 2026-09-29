<!-- 建议素材：assets/JishuBenchBanner.png、assets/host-target-architecture.png -->

# JishuBench

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENCE)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.12-blue?logo=python)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active%20development-orange)]()

**让端侧大模型与硬件选型有据可依。**  
**Reproducible evaluation for edge AI models, devices, and inference setups.**

JishuBench 是一套端侧 AI 评测基础设施。它把模型效果、真实任务表现和硬件运行数据放进同一套评测流程，帮助开发者比较不同模型、设备与推理方案，而不是只凭单一跑分做决定。

JishuBench connects model quality, task performance, and hardware observations in one Host–Target workflow, giving edge AI teams comparable evidence for model and device selection.

[官方网站](https://aijishu.com/jishubench) · [快速开始](docs/quickstart.md) · [架构说明](docs/architecture.md) · [CLI 文档](docs/cli.md) · [安全策略](SECURITY.md)


<img src="./assets/AijishuBanner.png" alt="AIJISHU" width="100%">

## Overview / 项目简介

在边缘设备上评测模型，常见问题不是“跑不起来”，而是完整评测框架、数据集和评分流程本身就会占用大量资源，影响被测模型的运行环境。JishuBench 使用 **Host–Target 分离架构**解决这个问题：

- **Host** 负责下载数据集、运行 Benchmark、编排任务、调用 Judge 并计算得分
- **Target** 只负责运行被测模型的推理服务，以及可选的硬件指标采集
- Host 和 Target 通过 OpenAI 兼容接口连接

这样既能让资源受限的 Target 轻量接入，也能用同一套 Host 流程比较不同模型、设备和推理配置。

JishuBench keeps benchmark harnesses, datasets, orchestration, and scoring on the Host. The Target stays focused on inference and optional hardware monitoring, making it easier to compare edge deployments without placing the full evaluation stack on every device.

## Key Capabilities / 核心能力

### Model and Hardware Evaluation / 模型与硬件联合评测

- 在同一流程中记录模型任务得分与设备运行信息
- 比较不同模型、量化版本、推理服务和硬件组合
- 使用一致的任务和配置减少人工测试带来的偏差
- 为端侧模型部署和设备选型提供可复查的结果

### Multiple Benchmarks, One Workflow / 多 Benchmark 统一编排

- 统一执行视觉语言、工具调用、终端操作和自主 Agent 任务
- 使用同一套 CLI 完成预检、数据准备、运行和结果归档
- 按 Benchmark 安装依赖，避免为单一任务安装全部框架
- 支持限制样本数量和指定任务，方便先做小规模验证

### Host–Target Separation / Host–Target 分离

- Host 在工作站或服务器上运行完整评测框架
- Target 只需提供 OpenAI 兼容推理端点
- 支持使用 llama.cpp Router 管理 Target 上的模型加载和切换
- Target 无需重复安装数据集、评分器和全部 Python 依赖

### Preflight and Reproducibility / 预检与复现

- 运行前检查 Benchmark、子模块、Target 连接和模型状态
- 使用 YAML 设备配置保存推理地址、模型和缓存路径
- 记录运行日志、配置和结果文件，便于后续核对
- 支持预下载数据集并迁移到离线或受限网络环境

### Optional Hardware Monitoring / 可选硬件监控

- 通过 `edge-eval-agent` 采集 Target 硬件指标
- 使用独立监控通道，避免与模型推理接口混在一起
- 可将监控地址写入设备配置，随评测流程收集运行信息

## How It Works / 工作方式

```text
Host
├─ Benchmark frameworks
├─ Datasets
├─ Task orchestration
├─ Judge and scoring
└─ Run logs and results
          │
          │ OpenAI-compatible API
          ▼
Target
├─ Inference service, such as llama.cpp
└─ edge-eval-agent for optional hardware metrics
```

Host 可以保持不变，只需切换设备配置，就能把同一组任务运行到不同 Target 上；Target 也不需要承载完整评测环境。

## Supported Benchmarks / 支持的评测

| Backend | 评测内容 | Integration |
| --- | --- | --- |
| **lmms-eval** | MMMU、RealWorldQA 等视觉语言模型任务 | `submodules/lmms-eval` |
| **tau-bench** | Agent、工具和模拟用户之间的多轮交互 | `submodules/tau2-bench` |
| **terminal-bench** | 通过传统 `tb` Harness 运行 Terminal-Bench v1 | Python package |
| **terminal-bench-2** | 通过 Harbor 运行 Terminal-Bench 2.0 | Python package |
| **claw-eval** | 带 Docker Sandbox、任务筛选和 Judge 的 Agent 评测 | `submodules/claw-eval` |

## Quick Start / 快速开始

### Requirements / 前置要求

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Git
- 可访问 Target 推理服务的 Host
- 运行 Terminal-Bench 或 Claw-Eval Sandbox 时需要 Docker

### Install / 安装 Host

```bash
git clone https://github.com/x-aijishu/jishubench.git --recursive
cd jishubench
uv sync --extra essentials
uv run jishubench preflight
```

`essentials` 安装除 Harbor 外的主要评测后端和开发工具。只需要部分能力时，可以按需安装：

```bash
uv sync --extra lmms-eval
uv sync --extra tau
uv sync --extra terminal
uv sync --extra harbor
uv sync --extra claw-eval
uv sync --extra dev
```

> `harbor` 与 `tau` / `essentials` 存在依赖冲突，请勿组合安装。

### Configure the Target / 配置 Target

选择或创建 `configs/devices/` 下的设备配置，至少填写 Target 的推理地址和模型：

```yaml
target:
  inference_base_url: http://<target-ip>:8080/v1
  model: Qwen3.5-4B-Q4_K_M
  monitor_base_url: http://<target-ip>:9090
```

Target 需要提供 OpenAI 兼容推理接口。使用 llama.cpp 时，可以在边缘设备上启动：

```bash
llama-server \
  --host 0.0.0.0 \
  --port 8080 \
  --models-dir /path/to/models \
  -np 4
```

硬件监控为可选能力：

```bash
edge-eval-agent --bind 0.0.0.0:9090
```

### Run / 运行评测

```bash
uv run jishubench preflight --config configs/devices/macstudio.yaml
uv run jishubench run mmmu_val --config macstudio --limit 10
```

也可以设置默认设备配置：

```bash
uv run jishubench config use
uv run jishubench run mmmu_val --limit 10
```

## Usage / 使用方式

### 查看可用评测

```bash
uv run jishubench benchmarks
uv run jishubench benchmarks --search mmmu
uv run jishubench benchmarks --json
```

### 同时运行多个任务

```bash
uv run jishubench run mmmu_val realworldqa --limit 10
```

### 提前下载数据集

```bash
uv run jishubench download mmmu_val realworldqa
uv run jishubench download mmmu_val --check
```

### 管理 Target 模型

配置 llama.cpp Router 后，可以从 Host 查看、加载和卸载模型：

```bash
uv run jishubench target models
uv run jishubench target status
uv run jishubench target load --model Qwen3.5-4B-Q4_K_M
uv run jishubench target unload --model Qwen3.5-4B-Q4_K_M
```

## Reading the Results / 如何理解结果

JishuBench 帮助你在相同条件下获得可比较的数据，但单次得分不等于对模型或硬件的最终结论。比较结果时应同时记录：

- 模型名称、版本和量化方式
- 推理服务及其启动参数
- Target 设备、系统和驱动版本
- Benchmark、数据版本和样本范围
- 并发、上下文长度和其他运行配置
- Judge 模型及评分设置

官网展示的 Phase 1 得分是特定设备与配置下的单次内部测试结果，仅供参考，不应直接当作通用排行榜。

## Documentation / 文档

| Topic / 主题 | Link / 链接 |
| --- | --- |
| Quick start / 快速开始 | [docs/quickstart.md](docs/quickstart.md) |
| Architecture / 架构 | [docs/architecture.md](docs/architecture.md) |
| Configuration / 配置 | [docs/configuration/overview.md](docs/configuration/overview.md) |
| CLI reference / CLI 参考 | [docs/cli.md](docs/cli.md) |
| API keys / 密钥配置 | [docs/configuration/api-keys.md](docs/configuration/api-keys.md) |
| Tau-Bench | [docs/configuration/tau-bench.md](docs/configuration/tau-bench.md) |
| Terminal-Bench v1 | [docs/configuration/terminal-bench.md](docs/configuration/terminal-bench.md) |
| Terminal-Bench 2.0 | [docs/configuration/terminal-bench-2.md](docs/configuration/terminal-bench-2.md) |
| Claw-Eval | [docs/configuration/claw-eval.md](docs/configuration/claw-eval.md) |
| edge-eval-agent | [docs/edge-eval-agent/index.md](docs/edge-eval-agent/index.md) |
| llama.cpp Router | [docs/deployment/llama-cpp-router.md](docs/deployment/llama-cpp-router.md) |

本地预览文档：

```bash
uv run mkdocs serve
```

## Boundaries and Security / 边界与安全

- JishuBench 是评测工具，不是用于直接暴露到公网的网络服务
- Host 会读取用户提供的 YAML 配置，并连接其中指定的 Target 地址
- Target 推理端点和监控端点应只开放给可信网络，并按需配置认证
- Judge API Key 与 Target 推理凭据应分开管理，不要提交到仓库
- Terminal-Bench 和 Claw-Eval 会运行外部 Harness 或 Docker Sandbox，执行前应确认任务来源和隔离环境
- `edge-eval-agent` 当前使用独立监控通道，不参与每个样本的评分闭环
- 预检通过只表示运行条件满足，不代表评测结果有效或选型已经完成
- 第三方 Benchmark、数据集和依赖的许可及安全问题应遵循对应上游项目规则

安全问题请按照 [SECURITY.md](SECURITY.md) 私下报告，不要直接创建公开 Issue。

## Development / 开发

```bash
uv sync --extra essentials
uv run pre-commit install
uv run pytest
uv run ruff check .
```

完整贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## Project Status / 项目状态

**Active development · 0.1.x**

JishuBench 当前支持版本为 0.1.x，核心 Host–Target 流程和多种 Benchmark 接入已经可用，接口、配置和硬件监控能力仍在持续完善。

## Acknowledgments

JishuBench integrates with and builds on:

- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- [tau2-bench](https://github.com/sierra-research/tau2-bench)
- [terminal-bench](https://github.com/harbor-framework/terminal-bench)
- [claw-eval](https://github.com/claw-eval/claw-eval)

## License

[Apache License 2.0](LICENCE)

## About AIJISHU

JishuBench 由 [AIJISHU](https://aijishu.com/) 团队打造。我们关注 AI Agent、知识工具、评测系统，以及 AI 与真实设备结合的开发体验。

**AIJISHU builds practical AI tools for agents, knowledge, evaluation, and real-world development.**
