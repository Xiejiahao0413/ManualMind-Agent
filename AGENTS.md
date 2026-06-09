# AGENTS.md

## Project Name

ManualMind-Agent

## Project Overview

ManualMind-Agent is a production-style multi-agent fault diagnosis system for complex equipment manuals.

The system targets industrial equipment manuals, fault code tables, maintenance instructions, safety rules, and historical fault cases. It is not a simple RAG demo. The goal is to build a deep Agent engineering project with controlled tool calling, hybrid retrieval, MCP tool execution, layered memory, sensitive data protection, retry/fallback handling, circuit breaker control, SSE streaming response, and human handoff.

## Core Architecture

The system should follow this architecture:

FastAPI stateless API layer
→ Sensitive data guard and streaming output filter
→ LangGraph multi-agent workflow
→ Memory Manager
→ Tool Router / Tool Call Guard / Retry & Fallback Manager
→ MCP Client
→ MCP Server tools
→ BM25 + Milvus Dense Retrieval + BGE-rerank
→ Safety review and report generation
→ SSE streaming response

FastAPI must remain stateless. Session state, task state, tool call records, and safety audit records must be managed by Memory Manager.

## Main Agents

Use LangGraph to implement a Supervisor-Worker multi-agent workflow.

Required graph nodes:

1. Supervisor Node
   - Classifies user intent.
   - Routes tasks to diagnosis, retrieval, safety, report, clarification, circuit breaker, or human handoff nodes.

2. Diagnosis Node
   - Extracts device name, device model, fault code, symptoms, query type, and risk level.
   - Writes structured diagnosis information into DiagnosisState.

3. Retrieval Node
   - Calls the tool layer through Tool Router.
   - Uses hybrid retrieval and case memory retrieval.
   - Writes retrieved evidence and source references into DiagnosisState.

4. Safety & Report Node
   - Checks high-risk operations.
   - Adds safety notices.
   - Generates structured diagnosis reports with citations.

5. Circuit Breaker Node
   - Stops repeated invalid reasoning or tool loops.
   - Triggers when retry_count reaches 3 or repeated tool signatures are detected.

6. Handoff Node
   - Creates a sanitized human handoff payload when the system cannot safely answer.

## Tool Calling Rules

Agents must not directly call tools.

All tool calls must go through:

Tool Intent
→ Tool Router
→ Tool Call Guard
→ Retry & Fallback Manager
→ MCP Client
→ MCP Server
→ Tool Result Validator
→ Memory Manager

Tool Call Guard must check:

- tool whitelist
- argument schema
- sensitive data
- duplicate tool call signatures
- retry count
- max tool call budget
- current graph state
- missing required arguments

The system must avoid infinite loops. Use:

- retry_count
- max_retry = 3
- tool_call_history
- args_signature
- called_signatures
- retrieval_status

If the same tool with the same normalized arguments has already been called successfully, reuse the cached result instead of calling it again.

If the task fails after three retries, trigger the circuit breaker and create a sanitized human handoff ticket.

## Retrieval Design

Use hybrid retrieval:

1. BM25 sparse retrieval
2. Milvus dense vector retrieval
3. Candidate merge and chunk_id deduplication
4. BGE-rerank reranking
5. Return Top-N evidence chunks with source metadata

Dense retrieval should use Milvus.

BM25 should be implemented as a local sparse retrieval component first.

Rerank should be abstracted behind a reranker interface, so that a local BGE-rerank model or API implementation can be plugged in later.

Metadata fields should include:

- chunk_id
- doc_id
- device_name
- device_model
- section_title
- page
- content_type
- fault_code
- source_file
- text

## MCP Tools

Implement an MCP Server exposing at least these tools:

1. manual_hybrid_search
2. fault_code_lookup
3. parameter_lookup
4. safety_rule_search
5. case_memory_search
6. source_trace
7. handoff_ticket_create

## Memory Design

Implement a Memory Manager with the following memory types:

1. Session Memory
   - Multi-turn conversation context.

2. Task Memory
   - Persistent diagnosis task state.

3. Tool Memory
   - Tool call history, argument signatures, retry count, fallback status.

4. Device Memory
   - Sanitized device profile and historical device information.

5. Case Memory
   - Historical verified fault cases, searchable by vector retrieval.

6. Safety Memory
   - Sensitive data events, safety review events, human handoff logs.

## Sensitive Data Protection

Implement a sensitive data guard covering:

- user input
- document ingestion
- MCP tool arguments
- retrieval results
- SSE streaming output

Detect and mask:

- phone numbers
- emails
- IP addresses
- device IDs
- work order IDs
- internal URLs
- locations
- contact names when possible

Use regex and dictionary-based detection first. Keep the implementation modular.

## Retry, Fallback, and Human Handoff

Implement Retry & Fallback Manager.

Fallback rules:

- Milvus failure → BM25-only
- BM25 failure → Dense-only
- BGE-rerank failure → fused retrieval score
- LLM generation failure → template-based diagnosis report
- missing device model → clarification response
- high-risk operation without enough safety evidence → human handoff
- evidence conflict → clarification or human handoff
- three failed retries → circuit breaker and human handoff

## API Requirements

Use FastAPI.

Required endpoints:

- POST /api/manual/upload
- POST /api/manual/index
- POST /api/diagnosis/chat
- GET /api/diagnosis/tasks/{task_id}
- POST /api/eval/run
- GET /api/health

The diagnosis chat endpoint must support SSE streaming response.

## Engineering Requirements

- Use Python.
- Keep modules clean and decoupled.
- Use typed Pydantic models for API schemas and tool schemas.
- Use clear interfaces for retrieval, rerank, memory, tools, and agents.
- Add tests for core modules.
- Add README with setup, run, and test instructions.
- Do not hard-code API keys.
- Use environment variables and .env.example.
- Do not implement fake business logic hidden inside prompts.
- Prefer explicit Python logic for routing, guards, retry, fallback, circuit breaker, validation, and handoff.
- Agents must not bypass Tool Router when invoking tools.

## Definition of Done

A task is done only when:

1. Code is implemented.
2. Imports are clean.
3. Basic tests pass.
4. README or docs are updated if necessary.
5. The module can be explained from architecture to execution flow.
