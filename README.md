# ManualMind-Agent

ManualMind-Agent is a production-style multi-agent fault diagnosis system for complex equipment manuals.

This repository currently contains the first engineering skeleton only. It defines the FastAPI entrypoint, core schemas, memory interfaces, tool routing guard interfaces, retrieval abstractions, and basic import tests.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
uvicorn app.main:app --reload
```

## Test

```powershell
pytest
```

## Current Scope

- FastAPI app entrypoint.
- Health, manual, and diagnosis API routers.
- SSE-capable diagnosis chat endpoint.
- Typed Pydantic schemas.
- Abstract memory interfaces.
- In-memory Memory Manager for session, task, tool, safety, and case memory.
- Tool router, tool call guard, retry/fallback manager, result validator, and circuit breaker control logic.
- LangGraph diagnosis workflow skeleton with supervisor, diagnosis, retrieval, safety report, circuit breaker, and handoff nodes.
- Retrieval and reranker abstractions.

Real LLM reasoning, MCP execution, Milvus retrieval, BM25 indexing, BGE reranking, and persistent memory will be implemented in later iterations.
