import asyncio
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.agents import DiagnosisWorkflow
from app.core.dependencies import get_manual_indexer, get_trace_manager
from app.main import app
from app.mcp_server import MCPToolExecutor, create_default_tool_registry
from app.schemas.diagnosis import DiagnosisRequest


DEMO_MANUAL_PATH = Path("data/demo_manuals/a100_manual.md")
DEMO_QUERY = "空压机 A100 报 E03，应该如何排查？"


def index_demo_manual(doc_id: str = "test-demo-a100") -> None:
    get_manual_indexer().index_document(
        content=DEMO_MANUAL_PATH.read_bytes(),
        filename=DEMO_MANUAL_PATH.name,
        doc_id=doc_id,
        device_name="空压机 A100",
        device_model="A100",
    )


def test_demo_manual_can_be_indexed() -> None:
    result = get_manual_indexer().index_document(
        content=DEMO_MANUAL_PATH.read_bytes(),
        filename=DEMO_MANUAL_PATH.name,
        doc_id="test-demo-index",
        device_name="空压机 A100",
        device_model="A100",
    )

    assert result.status == "indexed"
    assert result.chunks_count >= 5
    assert result.content_type_stats["fault_code"] >= 1
    assert result.content_type_stats["safety_rule"] >= 1


def test_demo_index_enables_e03_retrieval() -> None:
    index_demo_manual("test-demo-retrieval")
    executor = MCPToolExecutor(create_default_tool_registry())

    result = asyncio.run(
        executor.execute_tool(
            "manual_hybrid_search",
            {
                "query": "E03 温度传感器异常",
                "device_model": "A100",
                "top_k_bm25": 5,
                "top_k_dense": 5,
                "top_n_rerank": 3,
            },
        )
    )

    assert result.status == "success"
    assert result.data is not None
    assert result.data["results"]
    assert any("E03" in item["text"] for item in result.data["results"])


def test_demo_workflow_outputs_final_answer_and_trace() -> None:
    index_demo_manual("test-demo-workflow")
    workflow = DiagnosisWorkflow()
    state = DiagnosisWorkflow.from_request(
        DiagnosisRequest(session_id="demo-test-session", message=DEMO_QUERY)
    )

    final_state = asyncio.run(workflow.run(state))

    assert final_state.final_answer is not None
    assert "E03" in final_state.final_answer
    assert "温度传感器" in final_state.final_answer
    assert "安全提醒" in final_state.final_answer
    for section in ("故障识别", "可能原因", "排查步骤", "安全提醒", "引用来源"):
        assert section in final_state.final_answer
    assert "Review retrieved manual evidence" not in final_state.final_answer
    assert "Check the fault code or parameter range against the cited source" not in final_state.final_answer
    assert "No high-risk operation detected by current tool results" not in final_state.final_answer
    assert final_state.final_answer.count("E03 表示温度传感器异常") == 1
    assert final_state.source_refs
    assert len(final_state.source_refs) == len(set(final_state.source_refs))
    assert "a100_manual.md" in final_state.final_answer
    assert final_state.trace_id is not None

    events = get_trace_manager().list_events(final_state.trace_id)
    assert events
    assert any(event.event_type == "tool_call_completed" for event in events)


def test_demo_diagnosis_sse_contains_workflow_events_and_trace_api() -> None:
    client = TestClient(app)
    upload_response = client.post(
        "/api/manual/upload",
        files={
            "file": (
                DEMO_MANUAL_PATH.name,
                DEMO_MANUAL_PATH.read_bytes(),
                "text/markdown",
            )
        },
    )
    assert upload_response.status_code == 200
    doc_id = upload_response.json()["doc_id"]

    index_response = client.post(
        "/api/manual/index",
        json={"doc_id": doc_id, "device_name": "空压机 A100", "device_model": "A100"},
    )
    assert index_response.status_code == 200
    assert index_response.json()["chunks_count"] >= 5

    diagnosis_response = client.post(
        "/api/diagnosis/chat",
        json={"session_id": "demo-api-session", "message": DEMO_QUERY},
    )

    assert diagnosis_response.status_code == 200
    body = diagnosis_response.text
    for event_name in (
        "input_sanitized",
        "diagnosis_completed",
        "tool_call_completed",
        "retrieval_completed",
        "final_answer",
    ):
        assert f"event: {event_name}" in body
    assert "trace_id" in body

    trace_id_match = re.search(r'"trace_id":\s*"([^"]+)"', body)
    assert trace_id_match is not None
    trace_id = trace_id_match.group(1)

    trace_response = client.get(f"/api/trace/{trace_id}/events")
    assert trace_response.status_code == 200
    event_types = {event["event_type"] for event in trace_response.json()}
    assert "diagnosis_completed" in event_types
    assert "retrieval_completed" in event_types
    assert "final_answer_generated" in event_types

    final_event = _sse_event_data(body, "final_answer")
    assert final_event is not None
    assert "E03" in str(final_event["final_answer"])
    assert final_event["source_refs"]


def _sse_event_data(body: str, event_name: str) -> dict[str, object] | None:
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if line == f"event: {event_name}" and index + 1 < len(lines):
            data_line = lines[index + 1]
            if data_line.startswith("data: "):
                return json.loads(data_line.removeprefix("data: "))
    return None
