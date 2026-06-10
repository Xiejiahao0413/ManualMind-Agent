# ManualMind-Agent

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## English

**ManualMind-Agent** is a production-style multi-agent fault diagnosis system for complex equipment manuals.

It demonstrates how to connect document ingestion, hybrid retrieval, controlled tool execution, safety guardrails, tracing, evaluation, and human handoff in one local Agent engineering project.

> **Synthetic data statement:** this repository uses synthetic equipment manuals and synthetic evaluation samples only. It does not include real manufacturer manuals, private industrial documents, or production customer data.

---

## Core Features

- **LangGraph-style multi-agent workflow** with supervisor, diagnosis, retrieval, safety/report, circuit breaker, and handoff nodes.
- **MCP-style tool execution layer** with tool registry, executor, structured schemas, and controlled tool access.
- **Tool control layer** with Tool Router, Tool Call Guard, retry/fallback handling, duplicate-call reuse, and circuit breaker logic.
- **Hybrid retrieval** with BM25 sparse retrieval, dense retrieval interface, metadata filters, candidate merge/dedup, and rerank abstraction.
- **Document ingestion** for Markdown/text manuals, section splitting, sensitive-data sanitization, chunk metadata, and local indexing.
- **Sensitive data guard** for user input, tool arguments, retrieval results, and SSE streaming output.
- **Human handoff** for high-risk operations, unsafe user requests, insufficient evidence, and retry-limit failures.
- **Trace & Evaluation** for workflow events, tool calls, retrieval evidence, fallback decisions, safety behavior, and handoff decisions.

---

## Current Scope

The current repository uses in-memory and mock/local components so the full demo can run on a laptop without external services.

Production extension points:

- Real LLM calls
- Real Milvus deployment
- Production BGE rerank model loading
- PDF/OCR parsing
- Persistent memory and trace storage
- Real MCP client/server transport
- Enterprise authentication, authorization, and audit storage

This project focuses on explicit Python control logic instead of hiding business behavior inside prompts.

---

## Data Scale

Current local demo data:

- `data/demo_manuals/a100_manual.md`: one focused A100 compressor manual.
- `data/manuals/`: 10 synthetic Markdown manuals across multiple equipment types.
- `data/eval_samples/equipment_fault_eval_50.json`: 50 standard evaluation samples.
- `data/eval_samples/equipment_fault_adversarial_30.json`: 30 adversarial/boundary evaluation samples.
- `data/eval_samples/equipment_fault_multi_manuals_50.json`: 50 multi-manual evaluation samples.

Default evaluation currently runs **130 samples**.

---

## Quick Start

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run tests:

```powershell
.venv\Scripts\python.exe -m pytest
```

Run the single-manual demo:

```powershell
.venv\Scripts\python.exe scripts\demo_run.py
```

Run the multi-manual demo:

```powershell
.venv\Scripts\python.exe scripts\demo_multi_manuals.py
```

Run evaluation:

```powershell
.venv\Scripts\python.exe scripts\run_eval.py
```

Start the API server:

```powershell
uvicorn app.main:app --reload
```

---

## Evaluation Summary

The evaluation framework measures:

- Tool selection accuracy
- Fault code extraction accuracy
- Source evidence coverage
- Safety reminder coverage
- Handoff decision accuracy
- Scenario breakdown for normal, boundary, adversarial, failure, high-risk, and security cases

The metrics are based on local synthetic manuals, mock/in-memory retrieval components, and deterministic workflow logic. They validate the engineering chain rather than claim production accuracy.

---

## Docs

- [Architecture](docs/architecture.md)
- [Demo Guide](docs/demo.md)
- [Interview Notes](docs/interview_notes.md)

---

## Project Structure

```text
app/
  agents/        multi-agent workflow and report formatter
  api/           FastAPI route modules
  core/          shared app configuration and dependencies
  evaluation/    evaluation schemas, loader, and runner
  ingestion/     manual parser, splitter, metadata, and indexer
  mcp_server/    MCP-style tool schemas, registry, executor, tools
  memory/        in-memory memory manager
  retrieval/     BM25, dense retriever interface, hybrid retrieval, rerank
  schemas/       Pydantic schemas
  security/      sensitive-data detector, sanitizer, streaming guard
  tools/         tool router, guard, fallback, validator, circuit breaker
  tracing/       in-memory trace manager
data/
  demo_manuals/  focused A100 demo manual
  manuals/       synthetic multi-equipment manuals
  eval_samples/  standard, adversarial, and multi-manual eval sets
docs/            architecture, demo, and interview notes
scripts/         demo and evaluation scripts
tests/           pytest test suite
```

---

<a id="中文"></a>

## 中文

**ManualMind-Agent** 是一个面向复杂设备手册的生产化风格多 Agent 故障诊断系统。

它把文档入库、混合检索、受控工具调用、安全防护、Trace、Evaluation 和人工接管串成一条可运行的本地工程链路，重点展示 Agent 系统工程能力，而不是普通 RAG Demo。

> **合成数据声明：** 本仓库只使用模拟设备手册和模拟评测样本，不包含真实厂家手册、企业内部文档或生产客户数据。

---

## 核心功能

- **LangGraph 风格多 Agent 工作流**：包含 supervisor、diagnosis、retrieval、safety/report、circuit breaker、handoff 节点。
- **MCP 风格工具执行层**：包含工具注册、统一执行器、结构化 schema 和受控工具访问。
- **工具控制层**：包含 Tool Router、Tool Call Guard、重试降级、重复调用复用和熔断逻辑。
- **混合检索**：包含 BM25、Dense Retriever 接口、metadata filter、候选合并去重和 rerank 抽象。
- **文档入库**：支持 Markdown/text 手册解析、结构化切分、脱敏、chunk 元数据和本地索引。
- **敏感信息防护**：覆盖用户输入、工具参数、检索结果和 SSE 流式输出。
- **人工接管**：用于高风险操作、不安全请求、证据不足和重试失败场景。
- **Trace & Evaluation**：记录节点流转、工具调用、检索证据、降级决策、安全行为和接管原因。

---

## 当前范围

该仓库目前采用内存型组件，使完整 demo 可以在本地直接运行，而不依赖外部服务。

真实 LLM 调用、真实 Milvus 部署、生产级 BGE rerank 模型加载、PDF/OCR 解析以及持久化记忆等能力，被设计为生产化扩展点，便于后续替换和接入。

本项目强调用显式 Python 逻辑实现路由、工具防护、重试降级、熔断、脱敏和人工接管，而不是把关键业务逻辑隐藏在 prompt 中。

---

## 数据规模

当前本地数据包括：

- `data/demo_manuals/a100_manual.md`：一份 A100 空压机示例手册。
- `data/manuals/`：10 份多设备模拟 Markdown 手册。
- `data/eval_samples/equipment_fault_eval_50.json`：50 条标准评测样本。
- `data/eval_samples/equipment_fault_adversarial_30.json`：30 条对抗/边界评测样本。
- `data/eval_samples/equipment_fault_multi_manuals_50.json`：50 条多手册评测样本。

默认评测共运行 **130 条样本**。

---

## 快速开始

创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

安装依赖：

```powershell
pip install -r requirements.txt
```

运行测试：

```powershell
.venv\Scripts\python.exe -m pytest
```

运行单手册 Demo：

```powershell
.venv\Scripts\python.exe scripts\demo_run.py
```

运行多手册 Demo：

```powershell
.venv\Scripts\python.exe scripts\demo_multi_manuals.py
```

运行评测：

```powershell
.venv\Scripts\python.exe scripts\run_eval.py
```

启动 API：

```powershell
uvicorn app.main:app --reload
```

---

## 评测结果摘要

当前评测框架覆盖：

- 工具选择准确率
- 故障码识别准确率
- 证据来源覆盖率
- 安全提醒覆盖率
- 人工接管决策准确率
- normal、boundary、adversarial、failure、high_risk、security 等场景统计

这些指标基于本地模拟手册、mock/in-memory 检索组件和确定性工作流逻辑，用于验证系统链路和评测框架，不代表真实生产准确率。

---

## 文档链接

- [架构文档](docs/architecture.md)
- [Demo 文档](docs/demo.md)
- [面试讲解笔记](docs/interview_notes.md)
