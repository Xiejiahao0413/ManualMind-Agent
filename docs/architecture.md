# Architecture

ManualMind-Agent is a production-style local demo for multi-agent equipment manual diagnosis. It uses synthetic manuals and in-memory or mock components by default, while keeping interfaces open for real LLM, Milvus, MCP transport, and persistent storage.

## Target Flow

```text
FastAPI API
-> Sensitive Data Guard
-> Multi-Agent Workflow
-> Memory Manager
-> Tool Router / Guard / Retry / Fallback / Circuit Breaker
-> MCP-style Tool Executor
-> Hybrid Retrieval
-> Safety Review
-> Report Generation
-> SSE Streaming Output
```

FastAPI remains stateless. Session state, task state, tool records, safety events, and trace events are handled by manager classes.

## Layered Architecture

### API Layer

The API layer is implemented with FastAPI. It exposes health checks, manual upload/indexing, diagnosis chat, task lookup, trace lookup, and evaluation endpoints.

The diagnosis endpoint supports SSE streaming. Output chunks pass through the Streaming Output Guard before being sent to the client.

### Agent Workflow Layer

The workflow follows a Supervisor-Worker design:

- `Supervisor Node`: routes execution according to state, risk, retry count, and handoff flags.
- `Diagnosis Node`: extracts query type, fault code, risk level, symptoms, and device hints with explicit rules.
- `Retrieval Node`: calls tools through the controlled tool chain.
- `Safety Report Node`: checks safety conditions and builds the final report.
- `Circuit Breaker Node`: stops retry loops and routes to handoff.
- `Handoff Node`: creates sanitized handoff payloads.

Agents do not directly call retrievers or tools.

### Tool Control Layer

The controlled call path is:

```text
Tool Intent
-> Tool Router
-> Tool Call Guard
-> Retry & Fallback Manager
-> MCP Tool Executor
-> Tool Result Validator
-> Memory Manager
```

The guard checks whitelist, argument schema, sensitive parameters, duplicate signatures, retry count, and task-level tool call budget. Successful calls can be reused by normalized `args_signature`.

### MCP Tool Server Layer

The project includes an MCP-style local tool server abstraction:

- `manual_hybrid_search`
- `fault_code_lookup`
- `parameter_lookup`
- `safety_rule_search`
- `source_trace`
- `handoff_ticket_create`

The current implementation is local and testable. It prepares the boundary for future real MCP transport without requiring external services in the demo.

### Retrieval Layer

The retrieval pipeline combines sparse precision and semantic-style retrieval:

```text
Query
-> LocalBM25Retriever
-> InMemoryDenseRetriever or optional VectorDenseRetriever
-> Candidate merge and chunk_id deduplication
-> Reranker interface
-> RetrievalResult[]
```

BM25 handles exact signals such as fault codes, model names, parameters, and section titles. Dense retrieval is represented through interfaces and memory/vector backends. Rerank is abstracted so local BGE-rerank or API-based rerank can be plugged in later.

Milvus is optional in v0.5. If Milvus is not configured or unavailable, the system falls back to the local in-memory dense backend.

### Ingestion Layer

Manual ingestion supports:

- Markdown manuals
- TXT manuals
- text-based PDF manuals

Pipeline:

```text
Manual file
-> Parser
-> Sensitive Data Guard
-> Section Splitter
-> Metadata Builder
-> ManualIndexer
-> HybridRetriever
```

Each chunk keeps `chunk_id`, `doc_id`, `device_name`, `device_model`, `section_title`, `page`, `content_type`, `fault_code`, `source_file`, `text`, and metadata.

PDF support is limited to text-based PDFs. Scanned PDFs and OCR are intentionally out of scope for the current local demo.

### Safety Layer

Sensitive Data Guard covers user input, document ingestion, tool arguments, retrieval results, and SSE output.

Streaming Output Guard uses a rolling buffer to prevent sensitive data leakage across SSE chunks. It masks phones, emails, IP addresses, device IDs, work order IDs, and internal URLs without recording raw sensitive values.

High-risk operations, unsafe bypass requests, missing evidence, retry-limit failures, and circuit breaker events can trigger human handoff.

### Reporting Layer

The report generation path uses a service wrapper:

```text
Structured diagnosis context
-> optional LLM client
-> output sanitization
-> validation
-> template fallback
```

By default, the project uses the deterministic template report formatter. If `MANUALMIND_LLM_PROVIDER=openai` and `OPENAI_API_KEY` are configured, the OpenAI client may be used. Failures, empty output, timeout, or unsafe output fall back to the template report.

### Trace & Evaluation Layer

Trace records workflow events, tool calls, fallback decisions, retrieval summaries, circuit breaker triggers, and handoff creation.

Evaluation covers:

- tool selection
- fault code extraction
- source coverage
- safety coverage
- handoff decision
- scenario breakdown for normal, boundary, adversarial, failure, high-risk, and security samples

The current synthetic evaluation set contains 130 samples.

## Version Evolution

| Version | Main capability |
| --- | --- |
| v0.1-local-demo | End-to-end local multi-agent diagnosis demo |
| v0.2-streaming-safety | Rolling-buffer SSE output sanitization |
| v0.3-pdf-ingestion | Text-based PDF parsing and indexing |
| v0.4-real-llm-report | Optional LLM report generation with template fallback |
| v0.5-milvus-retrieval | Optional Milvus/vector backend with memory fallback |

## Fallback Strategy

- **LLM fallback:** no provider, missing key, timeout, exception, empty response, or invalid output all use the template report.
- **Milvus fallback:** missing config, missing dependency, connection failure, or indexing/search failure uses the memory backend.
- **Retrieval fallback:** rerank failure can fall back to fused scores; dense or sparse failures can degrade to the available branch.
- **Streaming safety fallback:** uncertain tail fragments stay in the rolling buffer until safe to release or flush.
- **Workflow fallback:** retry limit, unsafe request, insufficient evidence, or high-risk uncertainty can route to human handoff.

## Why This Is More Than A Simple RAG Demo

A basic RAG demo usually retrieves chunks and asks an LLM to answer. ManualMind-Agent adds explicit engineering controls around the retrieval and generation path:

- agents have separated responsibilities and observable state transitions
- tools are routed, guarded, validated, retried, and recorded
- retrieval combines sparse, dense, metadata filters, dedup, and rerank interfaces
- sensitive data is handled at input, tool, retrieval, and streaming output boundaries
- high-risk cases can stop automatic answering and create handoff payloads
- evaluation checks behavior beyond answer text, including tool choice and safety decisions

The project is still a local synthetic demo, but its structure mirrors the control points needed in a production-style industrial diagnosis system.

## Current Boundaries

The repository does not include real manufacturer manuals, real production data, OCR for scanned PDFs, a running Milvus cluster, production LLM governance, enterprise auth, or persistent trace/memory storage.

Real private manuals should be kept in local ignored folders such as `local_data/` or `private_pdfs/`, not committed to the public repository.
