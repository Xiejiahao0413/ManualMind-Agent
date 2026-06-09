# Architecture Notes

This document is a placeholder for the ManualMind-Agent architecture.

The first implementation pass only establishes package boundaries and interfaces:

- FastAPI API layer remains stateless.
- Memory responsibilities are represented by abstract interfaces.
- Tool calls are routed through `ToolRouter` and checked by `ToolCallGuard`.
- Retrieval components are abstract interfaces for BM25, Milvus dense retrieval, hybrid retrieval, and BGE reranking.

Detailed execution flow will be documented when concrete components are implemented.
