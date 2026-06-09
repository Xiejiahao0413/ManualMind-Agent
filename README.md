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

## Demo

Run the local end-to-end demo without starting the API server:

```powershell
.\.venv\Scripts\python.exe scripts\demo_run.py
```

The script indexes `data/demo_manuals/a100_manual.md`, runs an E03 diagnosis through the workflow, and prints `final_answer`, `source_refs`, and `trace_id`.

To run the same flow through FastAPI:

```powershell
uvicorn app.main:app --reload
```

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

Query trace events with the returned `trace_id`:

```powershell
curl.exe http://127.0.0.1:8000/api/trace/<trace_id>/events
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
- Sensitive Data Guard for regex and dictionary-based masking across input, tool arguments, and SSE streaming output.
- Hybrid Retrieval layer with local BM25, in-memory dense retrieval, candidate merge/dedup, metadata filters, and mock reranker.
- MCP Tool Server foundation with tool schemas, registry, executor, and demo-backed retrieval, lookup, safety, source trace, and handoff tools.
- Workflow-to-MCP integration through Tool Router, Tool Call Guard, MCP Tool Executor, Tool Result Validator, and Memory Manager.
- Trace and Evaluation layer for workflow observability, tool/retrieval/fallback/handoff traces, and basic regression metrics.
- Document ingestion and knowledge base indexing for txt/md manuals, including parsing, sensitive data masking, structured chunking, metadata construction, and in-memory Hybrid Retrieval indexing.
- End-to-end demo data and script for indexing an A100 air compressor manual, diagnosing E03, streaming SSE events, and inspecting trace output.

Real LLM reasoning, remote MCP execution, real Milvus services, production BGE reranking, PDF/OCR parsing, and persistent memory will be implemented in later iterations.
