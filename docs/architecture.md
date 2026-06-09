# Architecture Notes

This document is a placeholder for the ManualMind-Agent architecture.

The first implementation pass only establishes package boundaries and interfaces:

- FastAPI API layer remains stateless.
- Memory responsibilities are represented by abstract interfaces.
- Tool calls are routed through `ToolRouter` and checked by `ToolCallGuard`.
- Retrieval components are abstract interfaces for BM25, Milvus dense retrieval, hybrid retrieval, and BGE reranking.

## Memory Manager

`InMemoryMemoryManager` provides a runnable, non-persistent implementation for early development and tests. It stores session messages, diagnosis task state, tool call records, safety events, and searchable case records in Python dictionaries and lists.

FastAPI routes should not own memory state directly. Shared memory access is exposed through `app.core.dependencies.get_memory_manager`, so later database-backed implementations can replace the in-memory version without changing route logic.

## Tool Control Layer

The current tool control layer is explicit Python logic, not prompt behavior:

- `ToolRouter` maps `query_type` values to candidate tool names.
- `ToolCallGuard` checks whitelist membership, required arguments, retry limits, per-task tool call budgets, and duplicate call signatures.
- Successful duplicate tool calls can return a `reuse_cached` decision.
- `RetryFallbackManager` maps known error types to fallback actions.
- `ToolResultValidator` checks empty results, missing source references, low retrieval scores, and high-risk operations without safety evidence.
- `CircuitBreaker` decides when retry or tool-call limits require a handoff payload.

## LangGraph Workflow Skeleton

`DiagnosisWorkflow` wires the first runnable LangGraph state machine around the shared `DiagnosisState` schema. The graph contains these nodes:

- `supervisor_node`
- `diagnosis_node`
- `retrieval_node`
- `safety_report_node`
- `circuit_breaker_node`
- `handoff_node`

The supervisor routes by explicit state rules: retry limits go to circuit breaker, existing handoff flags go to handoff, missing query type goes to diagnosis, missing retrieval results goes to retrieval, and retrieved evidence goes to safety report.

The diagnosis node uses simple regex and keyword rules only. It can detect fault codes such as `E03`, high-risk terms such as high voltage or live operation in Chinese user input, and parameter-style questions. It does not call an LLM.

The retrieval node does not call Milvus, BM25, or MCP. It uses `ToolRouter` and `ToolCallGuard`, records `ToolCallRecord` entries, and writes mock evidence chunks and source references into `DiagnosisState`.

The safety report node creates a structured mock final answer. If high-risk work lacks safety evidence, the workflow sets `handoff_required` and routes to handoff.

The SSE chat endpoint now streams coarse workflow events: `diagnosis_started`, `retrieval_started`, `safety_review_started`, followed by either `final_answer` or `handoff_required`.

## Sensitive Data Guard

The security layer currently provides a regex and dictionary-based sensitive data guard:

- `SensitiveDataDetector` returns structured spans for phone numbers, emails, IP addresses, device IDs, work order IDs, internal URLs, and location keywords.
- `DataSanitizer` replaces detected values with stable placeholders such as `[PHONE]`, `[EMAIL]`, `[IP_ADDRESS]`, `[DEVICE_ID]`, `[WORK_ORDER_ID]`, `[INTERNAL_URL]`, and `[LOCATION]`.
- `StreamingOutputGuard` keeps a small rolling buffer so SSE chunks can be sanitized before they leave the API boundary, including simple split-sensitive-text cases.
- `DiagnosisWorkflow.from_request` stores `raw_query`, routes the workflow with `sanitized_query`, and records `sanitized_fields` and `security_events`.
- `ToolCallGuard` blocks tool arguments that still contain unsanitized sensitive data.
- Sensitive input detections are written to Safety Memory with `event_type="sensitive_data_detected"` and `action_taken="masked"`.

This guard is intended to cover user input, document ingestion hooks, MCP tool arguments, retrieval result filtering, and SSE streaming output. This iteration wires user input, tool arguments, and SSE output; document ingestion and retrieval result filtering will reuse the same sanitizer in later implementation passes.

## Hybrid Retrieval Layer

The retrieval layer now has a testable hybrid retrieval implementation:

- `DocumentChunk` defines manual evidence chunks with device metadata, source file, fault code, page, content type, and free-form metadata.
- `LocalBM25Retriever` handles sparse exact matching for fault codes, device models, parameter names, and section titles.
- `InMemoryDenseRetriever` implements the Milvus dense retriever interface with mock embeddings and cosine similarity, so tests can run without a Milvus service.
- `merge_retrieval_results` combines BM25 and dense candidates by `chunk_id`; if both channels hit the same chunk, the result keeps both scores and marks `source="hybrid"`.
- `MockReranker` represents the BGE-rerank interface and returns Top-N candidates. If reranking fails, `HybridRetrieverImpl` falls back to fused score sorting.
- Metadata filters support `device_name`, `device_model`, `content_type`, `fault_code`, and `doc_id` to reduce cross-device retrieval errors.

BM25 is intended for exact industrial manual signals such as `E03`, model numbers, threshold names, and section headings. Dense retrieval is intended for natural-language symptom descriptions. Reranking is the final evidence ordering step before agent report generation.

## MCP Tool Server Foundation

The MCP server package now exposes a local tool layer that can later be wrapped by a real MCP transport:

- `ToolRegistry` registers tools by name and exposes lookup/list operations.
- `MCPToolExecutor` is the unified execution entry point. It validates input schemas, catches exceptions, and returns `ToolExecutionResult` with status, data, error type, error message, and latency.
- Demo-backed tools include `manual_hybrid_search`, `fault_code_lookup`, `parameter_lookup`, `safety_rule_search`, `source_trace`, and `handoff_ticket_create`.
- `manual_hybrid_search` calls the existing `HybridRetrieverImpl` with demo chunks so tests run without Milvus.
- Lookup tools return structured not-found data instead of raising for normal misses.
- `handoff_ticket_create` creates a local structured handoff ticket id and does not call a real ticketing system.

The intended controlled path remains:

`Agent Tool Intent -> Tool Router -> Tool Call Guard -> MCP Tool Executor -> Tool Result Validator -> Memory Manager`

Agents should not bypass the router and guard to invoke MCP tools directly.

## Workflow Tool Integration

The LangGraph retrieval node now calls tools through `WorkflowToolService` instead of writing mock retrieval results directly. The service owns the controlled execution path:

`Retrieval Agent -> Tool Router -> Tool Call Guard -> MCP Tool Executor -> Tool Result Validator -> Memory Manager -> DiagnosisState`

The retrieval node provides state context such as `query_type`, `fault_code`, `risk_level`, and device fields. `ToolRouter` selects candidate tools, `ToolCallGuard` checks whitelist, argument requirements, sensitive parameters, retry limits, duplicate signatures, and per-task tool budgets, then `MCPToolExecutor` executes registered tools.

Every tool attempt writes a `ToolCallRecord` to Tool Memory, including status, retry count, fallback usage, error type, latency, and summary. Successful duplicate signatures use the guard's cached-success path and avoid repeated MCP execution.

Executor and validation failures are passed through `RetryFallbackManager`. Fallback decisions are written to `DiagnosisState` as `fallback_decision` and `retrieval_mode`; handoff or circuit-breaker decisions set `handoff_required` and flow into the existing handoff node.

Tool outputs are written back into dedicated state fields:

- `manual_hybrid_search` -> `retrieved_chunks`, `retrieved_evidence`, `source_refs`, `retrieval_mode`
- `fault_code_lookup` -> `fault_info`, `source_refs`
- `parameter_lookup` -> `parameter_info`, `source_refs`
- `safety_rule_search` -> `safety_rules`, `source_refs`

SSE output now reports coarse workflow stages: `input_sanitized`, `diagnosis_completed`, `tool_routing_started`, `tool_call_completed`, `retrieval_completed`, `safety_review_completed`, and terminal `final_answer` or `handoff_required` events.

## Trace And Evaluation

Trace is separate from Memory. Memory stores operational state for sessions, tasks, tool loops, and safety logs. Trace stores observability events for debugging and evaluation.

`InMemoryTraceManager` supports starting request traces, appending events, reading a trace, listing events, and summarizing event counts. Workflow and tool-service events include supervisor routing, diagnosis, tool routing, tool call started/completed/failed, retrieval trace metadata, fallback decisions, safety review, circuit breaker triggers, handoff creation, and final answer generation.

Trace events are exposed through:

- `GET /api/trace/{trace_id}`
- `GET /api/trace/{trace_id}/events`

Diagnosis SSE responses include `trace_id`, so callers can inspect the trace after a streamed response.

The evaluation layer defines `EvalSample` and computes basic metrics without an LLM:

- tool selection accuracy
- fault code extraction accuracy
- handoff decision accuracy
- source coverage
- safety coverage

`POST /api/eval/run` runs samples through the current workflow and uses workflow outputs plus trace events to produce aggregate metrics and per-sample results.

Real MCP execution, retrieval backends, LLM calls, and durable memory are still intentionally out of scope.

## Document Ingestion And Knowledge Base Indexing

The ingestion layer provides the first runnable manual indexing path:

`Manual Upload -> Parser -> Sensitive Data Guard -> Section Splitter -> Chunk Metadata Builder -> ManualIndexer -> Hybrid Retrieval`

`POST /api/manual/upload` stores uploaded manual bytes in an in-memory manual store and returns a `doc_id`. `POST /api/manual/index` reads that stored manual, parses txt/md content, masks sensitive values, splits the manual into structured chunks, builds standard retrieval metadata, and adds chunks to the shared in-memory Hybrid Retriever.

The parser currently supports txt and Markdown-style text files. `PDFManualParser` is defined as an interface placeholder for a later PDF/OCR pass.

Chunking is explicit Python logic:

- fault code chunks are detected with patterns such as `E03`, `F12`, and `P001`.
- parameter chunks are detected by terms such as temperature, voltage, pressure, threshold, and maintenance cycle.
- safety rule chunks are detected by terms such as disconnect power, high temperature, pressurized work, prohibition, warning, and safety.
- maintenance and general chunks fill the remaining manual sections.

Each chunk carries `chunk_id`, `doc_id`, `device_name`, `device_model`, `section_title`, `page`, `content_type`, `fault_code`, `source_file`, `text`, and metadata. `chunk_id` is generated from stable document and content fields so repeated indexing of the same chunk does not duplicate it in the in-memory retriever.

MCP `manual_hybrid_search`, `fault_code_lookup`, `parameter_lookup`, `safety_rule_search`, and `source_trace` now prefer indexed manual chunks while keeping demo data as a fallback for tests and early development.
