# Architecture Notes

ManualMind-Agent is organized as a production-style diagnosis pipeline for complex equipment manuals. The current implementation uses local and in-memory components so the system can be tested end to end without external services.

## System Overview

The target architecture is:

```text
FastAPI stateless API
-> Sensitive Data Guard
-> LangGraph multi-agent workflow
-> Memory Manager
-> Tool Router / Tool Call Guard / Retry & Fallback / Circuit Breaker
-> MCP Tool Executor
-> MCP Tool Server
-> Hybrid Retrieval
-> Safety review and report generation
-> SSE streaming response
```

FastAPI remains stateless. Session state, diagnosis task state, tool call records, safety events, and traces live in dedicated manager classes.

## Document Ingestion Pipeline

Current path:

```text
Manual Upload
-> TextManualParser / PDFManualParser
-> Sensitive Data Guard
-> SectionSplitter
-> ChunkMetadataBuilder
-> ManualIndexer
-> HybridRetrieverImpl
```

Implemented behavior:

- `POST /api/manual/upload` stores uploaded bytes in an in-memory manual store and returns a `doc_id`.
- `POST /api/manual/index` parses txt/md/text-based PDF files, masks sensitive values, splits sections, builds retrieval metadata, and adds chunks to the shared in-memory retriever.
- `PDFManualParser` supports text-based PDF manuals through `pypdf`. It preserves page numbers for extracted text and returns `unsupported_pdf_type` for PDFs without extractable text. OCR and scanned PDF parsing are intentionally out of scope.

Each chunk contains:

- `chunk_id`
- `doc_id`
- `device_name`
- `device_model`
- `section_title`
- `page`
- `content_type`
- `fault_code`
- `source_file`
- `text`
- `metadata`

`chunk_id` is stable for repeated indexing of the same document content.

The repository also includes a local multi-manual corpus under `data/manuals/`.
It contains 10 original simulated Markdown manuals for A100/B200/C300/D400/P500/G600/H800/W700/T100/K900 equipment.
`scripts/index_manuals.py` batch-indexes these files into the current in-memory `ManualIndexer` and prints document, chunk, fault-code, and device-model counts.
This dataset is intended for local hybrid retrieval tests and later Milvus collection experiments without using real vendor manuals.

`data/pdf_manuals/` contains synthetic text-based PDF manuals for local ingestion tests. These files are generated demo data, not real manufacturer manuals. Real vendor PDFs should be kept only in local private folders such as `local_data/private_pdfs/` and must not be committed to the public repository.

## Retrieval Pipeline

The hybrid retrieval layer combines precise sparse retrieval and semantic-style dense retrieval:

```text
Query
-> LocalBM25Retriever
-> InMemoryDenseRetriever / MilvusDenseRetriever interface
-> Candidate merge by chunk_id
-> MockReranker / BGE-rerank interface
-> RetrievalResult[]
```

BM25 is useful for exact industrial signals:

- fault codes such as `E03`, `F12`, `P001`
- device models
- parameter names
- section titles

Dense retrieval is represented by `InMemoryDenseRetriever` and `MockEmbeddingProvider` for tests. The interface is designed so a real Milvus-backed retriever can replace it later.

Multi-manual retrieval experiments should index `data/manuals/` first, then query with device models such as `B200`, `C300`, or `H800`.
The workflow performs lightweight device-model extraction so retrieval metadata can limit evidence to the matching manual where possible.

The reranker is represented by an interface plus `MockReranker`. If rerank fails, `HybridRetrieverImpl` falls back to fused retrieval score ordering.

Metadata filters support:

- `device_name`
- `device_model`
- `content_type`
- `fault_code`
- `doc_id`

## Multi-Agent Workflow

The LangGraph workflow uses a supervisor-worker shape around `DiagnosisState`.

Current nodes:

- `supervisor_node`
- `diagnosis_node`
- `retrieval_node`
- `safety_report_node`
- `circuit_breaker_node`
- `handoff_node`

Routing rules are explicit Python logic:

- `retry_count >= max_retry` -> circuit breaker
- `handoff_required` -> handoff
- missing `query_type` -> diagnosis
- no retrieval result yet -> retrieval
- retrieved evidence -> safety report

The diagnosis node uses rule-based extraction only. It detects fault codes, parameter-style questions, and high-risk keywords. No LLM is called.

The safety report node delegates report text construction to `build_diagnosis_report`, which outputs a structured Chinese diagnosis report with:

- fault identification
- possible causes
- troubleshooting steps
- safety notices
- source references

## Tool Control Layer

Agents must not directly call tools. The workflow calls `WorkflowToolService`, which owns the controlled path:

```text
Tool Intent
-> ToolRouter
-> ToolCallGuard
-> MCPToolExecutor
-> ToolResultValidator
-> RetryFallbackManager
-> Memory Manager
-> DiagnosisState
```

`ToolCallGuard` checks:

- whitelist membership
- required arguments
- argument schema readiness
- sensitive data in tool args
- duplicate normalized args signatures
- retry count
- max tool calls per task

Successful calls are remembered in Tool Memory. If the same tool and normalized args have already succeeded, the service can reuse the cached result instead of executing the tool again.

## MCP Tool Server

The local MCP-style tool server contains:

- `ToolRegistry`
- `MCPToolExecutor`
- structured tool input and output schemas
- tool implementations

Implemented tools:

- `manual_hybrid_search`
- `fault_code_lookup`
- `parameter_lookup`
- `safety_rule_search`
- `source_trace`
- `handoff_ticket_create`

Tools currently run in-process. A real MCP transport layer can wrap the same registry and executor later.

## Sensitive Data Guard

The security layer uses regex and dictionary-based detection first.

Detected and masked values include:

- phone numbers
- emails
- IP addresses
- device IDs
- work order IDs
- internal URLs
- location keywords

Coverage points:

- user input
- document ingestion
- tool arguments
- SSE streaming output

`StreamingOutputGuard` uses a rolling buffer to prevent sensitive data leakage across SSE chunks.
Streaming Output Guard 使用滚动缓冲机制，避免敏感信息被 SSE 分块输出绕过检测。

## Memory Manager

`InMemoryMemoryManager` currently stores:

- Session Memory
- Task Memory
- Tool Memory
- Safety Memory
- Case Memory

The in-memory implementation is intentionally simple, but route code accesses it through dependencies so a durable backend can replace it later.

## Retry, Fallback, And Circuit Breaker

Known fallback mappings include:

- `milvus_timeout` -> `bm25_only`
- `bm25_error` -> `dense_only`
- `rerank_error` -> `fused_score`
- `llm_error` -> `template_report`
- `missing_device_model` -> `clarification`
- `high_risk_without_evidence` -> `human_handoff`
- `retry_count >= 3` -> `circuit_breaker_handoff`

The circuit breaker also checks repeated tool calls and max tool-call budgets. When triggered, it creates a sanitized `HandoffPayload`.

## Human Handoff

Handoff is used when the system cannot safely answer. Current handoff payload fields include:

- `task_id`
- `handoff_reason`
- `risk_level`
- `device_name`
- `fault_code`
- `symptoms`
- `tool_trace`
- `retrieved_sources`

The demo does not connect to a real ticketing system. `handoff_ticket_create` returns a structured local handoff id.

## Trace And Evaluation

Trace is separate from Memory.

Memory is operational state. Trace is observability data.

`InMemoryTraceManager` records events such as:

- supervisor started/completed
- diagnosis started/completed
- tool routing started/completed
- tool call started/completed/failed
- retrieval trace
- fallback decisions
- safety review
- circuit breaker trigger
- handoff creation
- final answer generation

Trace APIs:

- `GET /api/trace/{trace_id}`
- `GET /api/trace/{trace_id}/events`

The evaluation layer defines `EvalSample` and computes basic metrics without an LLM:

- tool selection accuracy
- fault code extraction accuracy
- handoff accuracy
- source coverage
- safety coverage

The repository includes `data/eval_samples/equipment_fault_eval_50.json`, a 50-sample A100 evaluation set covering:

- `fault_code`
- `symptom`
- `parameter`
- `safety`
- `handoff`

`scripts/run_eval.py` loads this JSON file, runs the current workflow, and prints aggregate metrics plus category-level breakdown.

The repository also includes `data/eval_samples/equipment_fault_adversarial_30.json`, a 30-sample adversarial and boundary evaluation set. It covers:

- `boundary`: missing model, fault-code variants, multi-code input, ambiguous questions
- `adversarial`: prompt injection, safety bypass, fake tool invocation
- `failure`: unknown fault code, out-of-manual requests, model conflicts
- `high_risk`: live/pressurized/hot repair requests
- `security`: phone, IP, work order, device id, email, internal URL, and location masking

`scripts/run_eval.py --all` runs both sets and reports aggregate metrics, `category_breakdown`, and `scenario_breakdown`.

## Current Limitations

- No real LLM integration.
- No real Milvus service required.
- No production BGE embedding or rerank model loaded.
- Text-based PDF parsing is implemented. Scanned PDF/OCR parsing is not implemented.
- Memory and trace data are in-memory only.
- MCP tools run in-process instead of through a remote MCP transport.
