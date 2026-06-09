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

Real MCP execution, retrieval backends, LLM calls, and durable memory are still intentionally out of scope.
