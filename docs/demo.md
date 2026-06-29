# Demo Guide

All demos use synthetic data. No API key, Milvus service, real LLM, or private manual is required for the default local path.

## Quick Commands

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\demo_run.py
.\.venv\Scripts\python.exe scripts\demo_multi_manuals.py
.\.venv\Scripts\python.exe scripts\demo_pdf_manuals.py
.\.venv\Scripts\python.exe scripts\demo_llm_report.py
.\.venv\Scripts\python.exe scripts\demo_milvus_retrieval.py
.\.venv\Scripts\python.exe scripts\run_eval.py
```

## Web UI Demo

Start the FastAPI backend:

```powershell
uvicorn app.main:app --reload
```

Start the Streamlit UI:

```powershell
streamlit run ui/app.py
```

The UI provides a lightweight product demo for manual upload and diagnosis chat. It uses the existing FastAPI backend and does not change the Agent, retrieval, LLM, or vectorstore logic.

## Demo Matrix

| Command | What it validates | Expected summary |
| --- | --- | --- |
| `pytest` | Import integrity, workflow, security, retrieval, ingestion, LLM fallback, vector fallback, evaluation | `141 passed` in the latest local validation |
| `scripts\demo_run.py` | Single A100 manual ingestion, diagnosis workflow, retrieval, report, trace | prints `indexed_doc_id`, `chunks_count`, `trace_id`, `source_refs`, `final_answer` |
| `scripts\demo_multi_manuals.py` | Multi-device synthetic manuals and cross-manual retrieval | prints one diagnosis result per sample question |
| `scripts\demo_pdf_manuals.py` | text-based PDF parsing, indexing, retrieval, page metadata | prints PDF-based `source_refs` and `trace_id` |
| `scripts\demo_llm_report.py` | optional LLM report service and template fallback | without API key prints `provider=template` and `fallback_used=true` |
| `scripts\demo_milvus_retrieval.py` | optional vector backend with memory fallback | without Milvus prints memory backend or fallback/skipped reason |
| `scripts\run_eval.py` | 130 synthetic evaluation samples across standard, adversarial, and multi-manual sets | prints aggregate metrics and category/scenario breakdown |

## Single Manual Demo

```powershell
.\.venv\Scripts\python.exe scripts\demo_run.py
```

This indexes `data/demo_manuals/a100_manual.md` and asks an E03 fault diagnosis question.

Expected output includes:

- `indexed_doc_id`
- `chunks_count`
- `trace_id`
- `source_refs`
- `final_answer`

The answer should contain sections such as fault identification, possible causes, troubleshooting steps, safety reminders, and cited sources.

## Multi-Manual Demo

```powershell
.\.venv\Scripts\python.exe scripts\demo_multi_manuals.py
```

This loads 10 synthetic Markdown manuals under `data/manuals/` and runs questions such as:

- A100 E03 handling
- B200 hydraulic pressure issue
- C300 conveyor motor overload
- H800 robotic arm emergency stop troubleshooting

It demonstrates metadata-aware retrieval across multiple devices.

## PDF Demo

```powershell
.\.venv\Scripts\python.exe scripts\index_pdf_manuals.py
.\.venv\Scripts\python.exe scripts\demo_pdf_manuals.py
```

This validates text-based PDF ingestion. The parser extracts text by page and preserves page metadata where available.

Scanned PDF and OCR are not part of the current scope.

## Optional LLM Report Demo

Default fallback mode:

```powershell
.\.venv\Scripts\python.exe scripts\demo_llm_report.py
```

Optional OpenAI mode:

```powershell
$env:MANUALMIND_LLM_PROVIDER="openai"
$env:MANUALMIND_LLM_MODEL="gpt-4.1-mini"
$env:OPENAI_API_KEY="your_api_key_here"
.\.venv\Scripts\python.exe scripts\demo_llm_report.py
```

Do not commit `.env`, API keys, or private configuration.

## Optional Milvus Demo

Default fallback mode:

```powershell
.\.venv\Scripts\python.exe scripts\demo_milvus_retrieval.py
```

Optional local Milvus mode:

```powershell
$env:MANUALMIND_VECTOR_BACKEND="milvus"
$env:MILVUS_URI="http://localhost:19530"
.\.venv\Scripts\python.exe scripts\demo_milvus_retrieval.py
```

If Milvus is unavailable, the script should report fallback behavior instead of crashing.

## Evaluation Demo

```powershell
.\.venv\Scripts\python.exe scripts\run_eval.py
```

Default evaluation uses:

- 50 standard samples
- 30 adversarial/boundary samples
- 50 multi-manual samples

The evaluation is designed for local framework validation. It should not be presented as production accuracy on real industrial data.
