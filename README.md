# ManualMind-Agent

A production-style multi-agent fault diagnosis system for complex equipment manuals.

## Overview / 项目概述

ManualMind-Agent is a Python/FastAPI engineering project for diagnosing industrial equipment faults from manuals, fault code tables, maintenance instructions, safety rules, and historical cases. It is designed as more than a simple RAG demo: the project emphasizes controlled tool calling, hybrid retrieval, streaming safety filtering, traceability, retry/fallback handling, circuit breakers, and human handoff.

中文说明：ManualMind-Agent 面向复杂设备手册和工业故障诊断场景，目标不是做一个简单的“文档问答 RAG”，而是搭建一个具备工程控制能力的多 Agent 系统。它覆盖文档入库、混合检索、受控工具调用、安全脱敏、SSE 流式输出、Trace 追踪、评测闭环和人工接管，适合作为 Agent 工程化项目展示。

## Core Features / 核心功能

- LangGraph supervisor-worker multi-agent workflow.
- MCP-style tool execution layer with registry, executor, schemas, and structured errors.
- BM25 + dense vector + rerank hybrid retrieval pipeline.
- Sensitive data guard for input, tool arguments, document ingestion, and SSE output.
- Tool Router / Tool Call Guard / Retry / Fallback / Circuit Breaker control layer.
- Human handoff payload generation for unsafe or unresolved cases.
- Trace & Evaluation layer for workflow events, tool calls, retrieval quality, fallback, and handoff decisions.
- FastAPI API layer with SSE streaming diagnosis responses.
- Runnable end-to-end demo using an A100 air compressor manual and E03 fault code.

中文说明：核心设计重点是“可控”和“可追踪”。Agent 不直接调用工具，而是通过 Tool Router、Tool Call Guard、Retry/Fallback 和 MCP Tool Executor 完成受控调用；检索层同时支持故障码等精确匹配和语义检索；高风险或证据不足时可以触发人工接管；Trace 和 Evaluation 用于排查链路问题和做回归评测。

## Architecture / 系统架构

```mermaid
flowchart TD
    Client[Client / Demo Script] --> API[FastAPI Stateless API]
    API --> Guard[Sensitive Data Guard]
    Guard --> StreamGuard[SSE Streaming Output Guard]
    Guard --> Graph[LangGraph Multi-Agent Workflow]

    Graph --> Supervisor[Supervisor Node]
    Supervisor --> Diagnosis[Diagnosis Node]
    Supervisor --> Retrieval[Retrieval Node]
    Supervisor --> Safety[Safety Report Node]
    Supervisor --> Breaker[Circuit Breaker Node]
    Supervisor --> Handoff[Handoff Node]

    Retrieval --> Router[Tool Router]
    Router --> ToolGuard[Tool Call Guard]
    ToolGuard --> Retry[Retry & Fallback Manager]
    Retry --> Executor[MCP Tool Executor]
    Executor --> Tools[MCP Tool Server]
    Tools --> Hybrid[Hybrid Retrieval]

    Hybrid --> BM25[BM25 Sparse Retrieval]
    Hybrid --> Dense[In-Memory Dense / Milvus Interface]
    Hybrid --> Rerank[BGE-Rerank Interface]

    Graph --> Memory[Memory Manager]
    Executor --> Memory
    Graph --> Trace[Trace Manager]
    Trace --> Eval[Evaluation Runner]
```

中文说明：FastAPI 层保持 stateless，任务状态、工具调用记录、安全审计和 Trace 都由独立 Manager 管理。LangGraph 负责多 Agent 状态流转，Retrieval Node 只能通过 Tool Control Layer 调用 MCP 工具，MCP 工具再访问 Hybrid Retrieval。这样可以避免 Agent 绕过工具控制层，也便于后续替换真实 Milvus、BGE-rerank 或远程 MCP Server。

## End-To-End Flow

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI
    participant Ingestion as ManualIndexer
    participant Workflow as LangGraph Workflow
    participant Tool as Tool Control Layer
    participant MCP as MCP Tool Executor
    participant Retrieval as Hybrid Retriever
    participant Trace as Trace Manager

    User->>API: Upload demo manual
    API->>Ingestion: Parse, sanitize, split, build chunks
    Ingestion->>Retrieval: Add chunks to BM25 + dense indexes
    User->>API: POST /api/diagnosis/chat
    API->>Workflow: Create DiagnosisState
    Workflow->>Tool: Route and guard tool calls
    Tool->>MCP: Execute manual_hybrid_search / fault_code_lookup
    MCP->>Retrieval: Search indexed manual chunks
    Retrieval-->>Workflow: Evidence + source_refs
    Workflow->>Trace: Record nodes and tool events
    Workflow-->>API: Structured final_answer
    API-->>User: SSE events with trace_id
```

## Quick Start / 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

中文说明：当前 demo 不依赖真实 LLM、真实 Milvus 或外部服务，安装依赖后即可在本地运行。

Run the local demo without starting the API server:

```powershell
.\.venv\Scripts\python.exe scripts\demo_run.py
```

Start FastAPI:

```powershell
uvicorn app.main:app --reload
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run the evaluation sets:

```powershell
.\.venv\Scripts\python.exe scripts\run_eval.py --all
```

Expected test result:

```text
105 passed, 1 warning
```

The warning is from the FastAPI/TestClient `httpx` deprecation path and does not affect the current demo.

## Demo / 示例演示

`scripts/demo_run.py` indexes `data/demo_manuals/a100_manual.md`, runs an E03 diagnosis, and prints the trace id, source references, and final report.

中文说明：示例手册是“空压机 A100”设备手册，其中包含故障码 `E03`、温度传感器异常说明、可能原因、排查步骤、安全提醒和维护周期。运行 demo 后，可以看到从文档入库、检索、工具调用、报告生成到 Trace 输出的完整链路。

```text
indexed_doc_id: demo-a100-manual
chunks_count: 5
trace_id: trace-<generated>
source_refs: ['a100_manual.md', 'a100_manual.md:1']
final_answer:
故障识别：
E03 表示温度传感器异常。控制器检测到温度传感器信号超出正常范围，可能导致温度读数不稳定、保护停机或过热报警

可能原因：
- 温度传感器接线松动或端子氧化
- 温度传感器探头损坏
- 冷却风扇堵塞导致局部温度异常
- 控制器采样通道异常

排查步骤：
1. 先断电并等待设备冷却
2. 检查温度传感器插头、线束和端子
3. 清理冷却风道和风扇滤网
4. 复位后观察 E03 是否再次出现
5. 如 E03 持续出现，更换温度传感器并记录维修结果

安全提醒：
- 处理 E03 前必须断电、等待冷却，禁止带压拆卸温度传感器

引用来源：
- a100_manual.md
- a100_manual.md:1
```

See [docs/demo.md](docs/demo.md) for a longer walkthrough.

## Evaluation Samples / 评测样本

The repository includes `data/eval_samples/equipment_fault_eval_50.json`, a 50-sample standard evaluation set for the A100 demo manual. It also includes `data/eval_samples/equipment_fault_adversarial_30.json`, a 30-sample adversarial and boundary evaluation set. The multi-manual dataset adds `data/eval_samples/equipment_fault_multi_manuals_50.json`, covering 10 simulated equipment manuals under `data/manuals/`.

Multi-manual local indexing and demo commands:

```powershell
.\.venv\Scripts\python.exe scripts\index_manuals.py
.\.venv\Scripts\python.exe scripts\demo_multi_manuals.py
.\.venv\Scripts\python.exe scripts\run_eval.py --multi-manuals
```

`scripts\run_eval.py --all` now runs standard + adversarial + multi-manual samples, for 130 local evaluation samples.

中文说明：标准 50 条样本覆盖故障码识别、故障现象诊断、参数查询、安全规范和人工接管五类问题；对抗/边界 30 条样本覆盖缺少设备型号、故障码格式变体、未知故障码、提示词注入、安全绕过、敏感信息混入和高风险维修等场景。Evaluation Runner 会输出 category 和 scenario_type 两个维度的分类统计。

## API Examples

Upload the demo manual:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/manual/upload `
  -F "file=@data/demo_manuals/a100_manual.md;type=text/markdown"
```

Index the returned `doc_id`:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/manual/index `
  -H "Content-Type: application/json" `
  -d "{\"doc_id\":\"<doc_id>\",\"device_name\":\"空压机 A100\",\"device_model\":\"A100\"}"
```

Run diagnosis chat with SSE:

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/api/diagnosis/chat `
  -H "Content-Type: application/json" `
  -d "{\"session_id\":\"demo-session\",\"message\":\"空压机 A100 报 E03，应该如何排查？\"}"
```

Query trace events:

```powershell
curl.exe http://127.0.0.1:8000/api/trace/<trace_id>/events
```

## Project Structure

```text
app/
  agents/        LangGraph workflow, tool-service integration, report formatter
  api/           FastAPI routers for manual, diagnosis, trace, eval, health
  core/          Config, dependencies, SSE helpers
  evaluation/    Evaluation schemas and runner
  ingestion/     Manual parsing, splitting, metadata, in-memory indexing
  mcp_server/    MCP-style tool schemas, registry, executor, tool implementations
  memory/        Memory interfaces and in-memory implementation
  retrieval/     BM25, dense retriever interface, hybrid search, reranker
  schemas/       Pydantic schemas for diagnosis, tools, retrieval, traces, documents
  security/      Sensitive data detector, sanitizer, streaming output guard
  tools/         Tool router, guard, retry/fallback, validator, circuit breaker
  tracing/       In-memory trace manager
data/
  demo_manuals/  A100 demo equipment manual
  manuals/       10 simulated equipment manuals for local retrieval experiments
  eval_samples/  Standard, adversarial, and multi-manual evaluation samples
docs/
  architecture.md
  demo.md
  interview_notes.md
scripts/
  demo_run.py
  demo_multi_manuals.py
  index_manuals.py
  run_eval.py
tests/
```

## Current Scope / 当前范围

This repository currently uses in-memory components so the full demo can run locally without external services. Real LLM calls, real Milvus deployment, production BGE rerank model loading, PDF/OCR parsing, and durable memory are intentionally left for later iterations.

该仓库目前采用内存型组件，使完整 demo 可以在本地直接运行，而不依赖外部服务。真实 LLM 调用、真实 Milvus 部署、生产级 BGE rerank 模型加载、PDF/OCR 解析以及持久化记忆等能力，被设计为生产化扩展点，便于后续替换和接入。

## Roadmap

- Add real Milvus persistence and collection management.
- Add production BGE embedding and rerank implementations.
- Add PDF parsing and OCR adapters for scanned manuals.
- Add a real MCP transport layer around the local tool server.
- Add durable memory backends for session, task, tool, safety, and case memory.
- Expand evaluation datasets and regression metrics.
- Add authentication and tenant-aware access control for production API usage.
