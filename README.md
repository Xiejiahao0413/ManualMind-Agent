# ManualMind-Agent

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## English

**ManualMind-Agent** is a multi-agent system for question answering and fault diagnosis over complex equipment manuals.

It demonstrates a production-style Agent architecture with controlled tool calling, hybrid retrieval, safety guardrails, streaming sanitization, traceability, evaluation, and human handoff.

> **Synthetic data only:** this repository uses synthetic demo manuals and synthetic evaluation samples. It does not include real manufacturer manuals, private industrial documents, or production customer data.

## Project Highlights

- **Multi-agent diagnosis workflow:** supervisor, diagnosis, retrieval, safety/report, circuit breaker, and handoff nodes.
- **Hybrid retrieval:** BM25 sparse retrieval, dense retrieval interface, metadata filtering, candidate deduplication, and rerank abstraction.
- **MCP-style tool execution layer:** tool registry, executor, schemas, result validation, and guarded tool access.
- **Tool control:** Tool Router, Tool Call Guard, retry/fallback handling, duplicate-call reuse, and circuit breaker logic.
- **Streaming safety:** Sensitive Data Guard plus SSE Streaming Output Guard with a rolling buffer to prevent cross-chunk leakage.
- **Document ingestion:** Markdown, TXT, and text-based PDF manual parsing with chunk metadata and local indexing.
- **Optional LLM report generation:** real OpenAI-compatible report generation can be enabled, with template fallback by default.
- **Optional Milvus backend:** vector store and embedding interfaces support Milvus experiments, with in-memory fallback by default.
- **Trace and evaluation:** workflow events, tool calls, retrieval evidence, safety behavior, and handoff decisions are observable and testable.

## Version Roadmap

| Version | Capability |
| --- | --- |
| v0.1-local-demo | Local end-to-end multi-agent diagnosis demo |
| v0.2-streaming-safety | Rolling-buffer SSE output sanitization |
| v0.3-pdf-ingestion | Text-based PDF manual ingestion |
| v0.4-real-llm-report | Optional real LLM report generation with template fallback |
| v0.5-milvus-retrieval | Optional Milvus/vector backend with memory fallback |

## Current Scope

The default project runs locally with mock or in-memory components. No API key, Milvus service, or external LLM is required for the local demos.

Production extension points include real LLM governance, real Milvus deployment, production embeddings, production rerank models, scanned PDF/OCR parsing, persistent memory, enterprise auth, and real MCP transport.

This project is intentionally explicit: routing, retry, fallback, circuit breaker, validation, sanitization, and handoff logic are implemented in Python rather than hidden inside prompts.

## Data and Evaluation

Local synthetic assets:

- `data/demo_manuals/a100_manual.md`: focused A100 compressor demo manual.
- `data/manuals/`: 10 synthetic Markdown manuals across multiple equipment types.
- `data/pdf_manuals/`: 2 synthetic text-based PDF manuals.
- `data/eval_samples/equipment_fault_eval_50.json`: 50 standard samples.
- `data/eval_samples/equipment_fault_adversarial_30.json`: 30 adversarial and boundary samples.
- `data/eval_samples/equipment_fault_multi_manuals_50.json`: 50 multi-manual samples.

Default evaluation covers **130 samples**. Current local validation: **141 tests passed**.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run tests and demos:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\demo_run.py
.\.venv\Scripts\python.exe scripts\demo_multi_manuals.py
.\.venv\Scripts\python.exe scripts\demo_pdf_manuals.py
.\.venv\Scripts\python.exe scripts\demo_llm_report.py
.\.venv\Scripts\python.exe scripts\demo_milvus_retrieval.py
.\.venv\Scripts\python.exe scripts\run_eval.py
```

Start the API:

```powershell
uvicorn app.main:app --reload
```

Core endpoints:

- `GET /api/health`
- `POST /api/manual/upload`
- `POST /api/manual/index`
- `POST /api/diagnosis/chat`
- `GET /api/diagnosis/tasks/{task_id}`
- `GET /api/trace/{trace_id}`
- `POST /api/eval/run`

## Documentation

- [Architecture](docs/architecture.md)
- [Demo Guide](docs/demo.md)
- [Interview Notes](docs/interview_notes.md)
- [Resume Bullets](docs/resume_bullets.md)
- [Version History](docs/version_history.md)

## Project Structure

```text
app/
  agents/        multi-agent workflow and tool service
  api/           FastAPI routes
  embeddings/    embedding client interfaces
  evaluation/    evaluation runner and schemas
  ingestion/     manual parsing, splitting, indexing
  llm/           optional LLM client abstraction
  mcp_server/    MCP-style tool registry and executor
  memory/        in-memory session/task/tool/safety memory
  retrieval/     BM25, dense, hybrid retrieval, rerank interface
  security/      sensitive data detection and streaming guard
  tracing/       in-memory trace manager
  vectorstore/   optional vector backend abstraction
data/            synthetic manuals and evaluation samples
docs/            architecture, demo, interview, resume docs
scripts/         local demo and indexing scripts
tests/           pytest coverage for core behavior
```

---

<a id="中文"></a>

## 中文

**ManualMind-Agent** 是一个面向复杂设备手册问答与故障诊断的多 Agent 系统。

它用一个本地可运行的项目，展示文档入库、混合检索、受控工具调用、安全脱敏、流式输出、Trace、Evaluation 和人工接管如何组合成一条完整诊断链路。

> **数据声明：** 本仓库只使用合成 demo 手册和合成评测样本，不包含真实厂家手册、企业私有文档或生产客户数据。

## 核心亮点

- **多 Agent 诊断工作流：** 包含 Supervisor、Diagnosis、Retrieval、Safety Report、Circuit Breaker、Handoff 等节点。
- **混合检索：** BM25 精确匹配、dense retrieval 接口、metadata filter、候选去重和 rerank 抽象。
- **MCP-style 工具层：** 工具注册、统一执行、结构化 schema、结果校验和受控工具访问。
- **工具调用控制：** Tool Router、Tool Call Guard、Retry/Fallback、重复签名复用和熔断逻辑。
- **流式安全：** Sensitive Data Guard 与 rolling buffer SSE 输出脱敏，避免敏感信息被分块绕过检测。
- **文档入库：** 支持 Markdown、TXT、text-based PDF 解析、切分、元数据构建和本地索引。
- **可选真实 LLM 报告：** 默认走模板报告；显式配置后可尝试真实 LLM，失败自动 fallback。
- **可选 Milvus 后端：** 默认 memory backend，可选接入 Milvus 做向量检索实验。
- **Trace 与评测：** 记录节点流转、工具调用、检索证据、安全行为和人工接管决策。

## 当前范围

当前默认采用本地 mock / in-memory 组件，完整 demo 可以直接在本机运行，不依赖 API key、Milvus 服务或外部 LLM。

OpenAI、Milvus、生产 embedding、生产 rerank、扫描 PDF/OCR、持久化记忆、真实 MCP transport、企业认证和审计存储，都是后续生产化扩展点。

项目重点不是宣称已经生产落地，而是展示一个“生产级架构风格”的 Agent 工程：关键控制逻辑用显式 Python 实现，而不是藏在 prompt 里。

## 数据规模

本地合成数据包括：

- 1 份 A100 空压机 demo 手册。
- 10 份多设备 Markdown 合成手册。
- 2 份 text-based PDF 合成手册。
- 50 条标准评测样本。
- 30 条对抗 / 边界评测样本。
- 50 条多设备评测样本。

默认评测共 **130 条样本**。当前本地验证结果：**141 tests passed**。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

常用命令：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\demo_run.py
.\.venv\Scripts\python.exe scripts\demo_multi_manuals.py
.\.venv\Scripts\python.exe scripts\demo_pdf_manuals.py
.\.venv\Scripts\python.exe scripts\demo_llm_report.py
.\.venv\Scripts\python.exe scripts\demo_milvus_retrieval.py
.\.venv\Scripts\python.exe scripts\run_eval.py
```

启动 API：

```powershell
uvicorn app.main:app --reload
```

## 文档

- [架构说明](docs/architecture.md)
- [Demo 指南](docs/demo.md)
- [面试讲解笔记](docs/interview_notes.md)
- [简历项目描述](docs/resume_bullets.md)
- [版本历史](docs/version_history.md)
