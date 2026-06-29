# Demo Walkthrough

This demo uses `data/demo_manuals/a100_manual.md`, an A100 air compressor manual containing fault code `E03`.

The demo path is:

```text
Load demo manual
-> Index manual chunks
-> Run diagnosis workflow
-> Call controlled MCP tools
-> Retrieve evidence
-> Build final diagnosis report
-> Print trace_id and source_refs
```

## Run

```powershell
.\.venv\Scripts\python.exe scripts\demo_run.py
```

## Demo Case

User question:

```text
空压机 A100 报 E03，应该如何排查？
```

The demo manual contains:

- equipment name: 空压机 A100
- fault code: `E03`
- fault meaning: 温度传感器异常
- possible causes
- troubleshooting steps
- safety notices
- technical parameters
- maintenance cycle

## Example Output

```text
indexed_doc_id: demo-a100-manual
chunks_count: 5
trace_id: trace-beabf1dfbffe4855a62b0b316b3f98b3
source_refs: ['a100_manual.md', 'a100_manual.md:1']
final_answer:
故障识别：
E03 表示温度传感器异常。控制器检测到温度传感器信号超出正常范围，可能导致温度读数不稳定、保护停机或过热报警

可能原因：
- 温度传感器接线松动或端子氧化
- 温度传感器探头损坏
- 冷却风扇堵塞导致局部温度异常
- 控制器采样通道异常

排查步骤：
1. 先断电并等待设备冷却
2. 检查温度传感器插头、线束和端子
3. 清理冷却风道和风扇滤网
4. 复位后观察 E03 是否再次出现
5. 如 E03 持续出现，更换温度传感器并记录维修结果

安全提醒：
- 处理 E03 前必须断电、等待冷却，禁止带压拆卸温度传感器

引用来源：
- a100_manual.md
- a100_manual.md:1
```

## Output Fields

- `indexed_doc_id`: the document id used when indexing the demo manual.
- `chunks_count`: number of structured chunks produced by the ingestion pipeline.
- `trace_id`: id for querying workflow and tool execution trace events.
- `source_refs`: manual source references used by retrieval and report generation.
- `final_answer`: structured diagnosis report generated from retrieved evidence.

## Trace Query

When running through FastAPI, the SSE response includes a `trace_id`.

Query trace events:

```powershell
curl.exe http://127.0.0.1:8000/api/trace/<trace_id>/events
```

Useful trace events include:

- `diagnosis_completed`
- `tool_routing_started`
- `tool_call_completed`
- `retrieval_trace`
- `retrieval_completed`
- `safety_review_completed`
- `final_answer_generated`

These events show how the diagnosis moved through the multi-agent workflow and controlled tool layer.

## Evaluation Demo

The project also includes standard, adversarial, and multi-manual evaluation sets:

```text
data/eval_samples/equipment_fault_eval_50.json
data/eval_samples/equipment_fault_adversarial_30.json
data/eval_samples/equipment_fault_multi_manuals_50.json
```

Run all evaluation samples with:

```powershell
.\.venv\Scripts\python.exe scripts\run_eval.py --all
```

The sample set covers:

- fault code questions
- symptom-based questions
- parameter lookup questions
- safety rule questions
- human handoff questions

The output includes tool selection accuracy, fault code accuracy, source coverage, safety coverage, handoff accuracy, and category-level breakdown.

The adversarial and boundary set adds scenario-level coverage for missing device model, fault-code variants, unknown faults, prompt injection, fake tool calls, safety bypass attempts, high-risk repair requests, and sensitive data masking.

## Multi-Manual Demo

`data/manuals/` contains 10 original simulated equipment manuals. Batch-index them with:

```powershell
.\.venv\Scripts\python.exe scripts\index_manuals.py
```

Run multi-manual diagnosis examples:

```powershell
.\.venv\Scripts\python.exe scripts\demo_multi_manuals.py
```

Run only the multi-manual evaluation set:

```powershell
.\.venv\Scripts\python.exe scripts\run_eval.py --multi-manuals
```

The multi-manual set covers fault-code, symptom, parameter, safety, and handoff-style questions across 10 equipment models, and is intended to exercise metadata filtering before a real Milvus backend is connected.

## PDF Manual Demo

`data/pdf_manuals/` contains synthetic text-based PDF manuals. They are generated demo data, not real manufacturer manuals.

Batch-index PDF manuals with:

```powershell
.\.venv\Scripts\python.exe scripts\index_pdf_manuals.py
```

Run PDF diagnosis examples:

```powershell
.\.venv\Scripts\python.exe scripts\demo_pdf_manuals.py
```

The PDF demo exercises text extraction, page metadata, chunk splitting, hybrid retrieval, workflow execution, and source references. Scanned PDF/OCR parsing is intentionally out of scope.

## Optional LLM Report Demo

The report layer uses the deterministic template formatter by default, so no API key is required for local demos.

Run the fallback demo:

```powershell
.\.venv\Scripts\python.exe scripts\demo_llm_report.py
```

To try real LLM report generation, set environment variables locally:

```powershell
$env:MANUALMIND_LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="your_api_key_here"
$env:MANUALMIND_LLM_MODEL="gpt-4o-mini"
.\.venv\Scripts\python.exe scripts\demo_llm_report.py
```

Do not commit `.env`, `.env.*`, or real API keys. If the provider/key is missing, the OpenAI SDK is unavailable, the request times out, or the model output is invalid, the demo falls back to the template report and continues running.
