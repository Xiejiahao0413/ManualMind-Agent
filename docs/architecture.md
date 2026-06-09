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

Real MCP execution, retrieval backends, LangGraph state transitions, and durable memory are still intentionally out of scope.
