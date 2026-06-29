# Resume Bullets

## 中文 3 条版本

- 设计并实现 ManualMind-Agent，一个面向复杂设备手册问答与故障诊断的生产级架构风格多 Agent 项目，基于 Python / FastAPI 构建 API、SSE 流式响应、任务状态、Trace 和 Evaluation 闭环。
- 实现 BM25 + dense retrieval + rerank interface 的混合检索链路，支持 Markdown / TXT / text-based PDF 文档入库、chunk 元数据构建、source citation，并提供可选 Milvus backend 与 memory fallback。
- 构建 Tool Router / Tool Call Guard / Retry-Fallback / Circuit Breaker / Human Handoff 控制层，覆盖敏感信息脱敏、rolling-buffer SSE 输出防泄漏、可选 OpenAI 报告生成与模板 fallback；当前本地验证 141 tests passed。

## 中文 5 条版本

- 基于 Python / FastAPI 实现 ManualMind-Agent，多 Agent 设备故障诊断系统，覆盖文档入库、诊断工作流、SSE 输出、Trace 查询和 Evaluation API。
- 设计 Supervisor、Diagnosis、Retrieval、Safety Report、Circuit Breaker、Handoff 等节点，明确拆分意图识别、证据检索、安全审查、报告生成和人工接管职责。
- 实现 BM25 + dense retrieval + rerank interface 的混合检索架构，支持 metadata filter、chunk_id 去重、source_refs 引用，并预留 Milvus / embedding backend 扩展。
- 构建受控工具调用链路：Tool Router、Tool Call Guard、Retry/Fallback、MCP-style Executor、Tool Result Validator 和 Tool Memory，避免 Agent 直接调用工具和重复调用死循环。
- 实现 Sensitive Data Guard、rolling-buffer Streaming Output Guard、可选 OpenAI LLM 报告生成与模板 fallback，并用 130 条合成评测样本和 141 个 pytest 用例验证核心链路。

## English 3-Bullet Version

- Built **ManualMind-Agent**, a production-style multi-agent fault diagnosis project for complex equipment manuals, using Python, FastAPI, SSE streaming, trace management, and an evaluation loop.
- Implemented a hybrid retrieval pipeline with BM25, dense retrieval interfaces, rerank abstraction, metadata filtering, citations, text-based PDF ingestion, and optional Milvus backend with memory fallback.
- Designed controlled tool execution with Tool Router, Tool Call Guard, retry/fallback handling, circuit breaker, human handoff, sensitive data guard, rolling-buffer SSE sanitization, optional OpenAI report generation, and template fallback; validated locally with 141 passing tests.

## English 5-Bullet Version

- Developed **ManualMind-Agent**, a production-style multi-agent system for equipment manual QA and fault diagnosis with Python, FastAPI, SSE streaming, trace APIs, and evaluation workflows.
- Designed a Supervisor-Worker workflow with diagnosis, retrieval, safety report, circuit breaker, and handoff nodes to make reasoning stages explicit and testable.
- Built a hybrid retrieval layer combining BM25, dense retrieval interfaces, metadata filters, candidate deduplication, rerank abstraction, citations, and optional Milvus vector backend with in-memory fallback.
- Implemented a controlled tool execution path with Tool Router, Tool Call Guard, retry/fallback manager, MCP-style tool executor, result validator, and tool memory to prevent unsafe or repeated tool calls.
- Added sensitive data protection, rolling-buffer streaming sanitizer, optional OpenAI report generation with template fallback, text-based PDF ingestion, and synthetic evaluation datasets covering 130 samples; current local validation: 141 tests passed.
