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

Real MCP execution, retrieval backends, LLM calls, and durable memory are still intentionally out of scope.
