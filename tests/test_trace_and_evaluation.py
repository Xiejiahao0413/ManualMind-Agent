import asyncio
import re

from fastapi.testclient import TestClient

from app.agents import DiagnosisWorkflow
from app.evaluation import EvalSample, EvaluationRunner
from app.main import app
from app.memory import InMemoryMemoryManager
from app.schemas.diagnosis import DiagnosisState
from app.schemas.trace import TraceEvent
from app.tracing import InMemoryTraceManager


def run_workflow(state: DiagnosisState, trace_manager: InMemoryTraceManager | None = None):
    trace_manager = trace_manager or InMemoryTraceManager()
    workflow = DiagnosisWorkflow(memory=InMemoryMemoryManager(), trace_manager=trace_manager)
    return asyncio.run(workflow.run(state)), trace_manager


def test_trace_manager_records_events() -> None:
    manager = InMemoryTraceManager()
    trace = manager.start_trace("req-1", "session-1", "task-1")
    event = TraceEvent(
        trace_id=trace.trace_id,
        request_id="req-1",
        session_id="session-1",
        task_id="task-1",
        event_type="diagnosis_started",
        component="diagnosis_node",
        status="started",
    )

    manager.add_event(trace.trace_id, event)

    assert manager.get_trace(trace.trace_id) is not None
    assert manager.list_events(trace.trace_id) == [event]
    assert manager.summarize_trace(trace.trace_id)["event_count"] == 1


def test_workflow_execution_contains_trace_id() -> None:
    state, manager = run_workflow(
        DiagnosisState(
            task_id="task-trace",
            session_id="session-1",
            user_query="E03 motor overheat",
        )
    )

    assert state.trace_id is not None
    events = manager.list_events(state.trace_id)
    assert any(event.event_type == "diagnosis_completed" for event in events)


def test_tool_call_writes_trace_event() -> None:
    state, manager = run_workflow(
        DiagnosisState(
            task_id="task-tool-trace",
            session_id="session-1",
            user_query="E03 motor overheat",
        )
    )

    events = manager.list_events(state.trace_id or "")
    tool_events = [event for event in events if event.event_type == "tool_call_completed"]
    assert tool_events
    assert any(event.metadata.get("tool_name") == "fault_code_lookup" for event in tool_events)


def test_circuit_breaker_writes_trace_event() -> None:
    state, manager = run_workflow(
        DiagnosisState(
            task_id="task-circuit-trace",
            session_id="session-1",
            user_query="motor fault",
            retry_count=3,
        )
    )

    events = manager.list_events(state.trace_id or "")
    assert any(event.event_type == "circuit_breaker_triggered" for event in events)


def test_handoff_writes_trace_event() -> None:
    state, manager = run_workflow(
        DiagnosisState(
            task_id="task-handoff-trace",
            session_id="session-1",
            user_query="motor fault",
            retry_count=3,
        )
    )

    events = manager.list_events(state.trace_id or "")
    assert any(event.event_type == "handoff_created" for event in events)


def test_trace_api_returns_trace() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/diagnosis/chat",
        json={"session_id": "trace-api-session", "message": "E03 motor overheat"},
    )
    assert response.status_code == 200
    match = re.search(r'"trace_id":\s*"([^"]+)"', response.text)
    assert match is not None

    trace_id = match.group(1)
    trace_response = client.get(f"/api/trace/{trace_id}")
    events_response = client.get(f"/api/trace/{trace_id}/events")

    assert trace_response.status_code == 200
    assert events_response.status_code == 200
    assert trace_response.json()["trace_id"] == trace_id
    assert events_response.json()


def test_evaluation_runner_calculates_tool_selection_accuracy() -> None:
    runner = EvaluationRunner(trace_manager=InMemoryTraceManager())
    response = asyncio.run(
        runner.run_eval(
            [
                EvalSample(
                    sample_id="tool-selection",
                    query="E03 motor overheat",
                    expected_tool_names=["fault_code_lookup"],
                )
            ]
        )
    )

    assert response.total == 1
    assert response.tool_selection_accuracy == 1.0


def test_evaluation_runner_calculates_fault_code_accuracy() -> None:
    runner = EvaluationRunner(trace_manager=InMemoryTraceManager())
    response = asyncio.run(
        runner.run_eval(
            [
                EvalSample(
                    sample_id="fault-code",
                    query="E03 motor overheat",
                    expected_fault_code="E03",
                )
            ]
        )
    )

    assert response.fault_code_accuracy == 1.0


def test_evaluation_runner_detects_workflow_handoff_required() -> None:
    runner = EvaluationRunner(trace_manager=InMemoryTraceManager())
    response = asyncio.run(
        runner.run_eval(
            [
                EvalSample(
                    sample_id="handoff-detected",
                    query="\u8bbe\u5907\u8fd8\u5728\u5e26\u538b\u8fd0\u884c\uff0c\u53ef\u4ee5\u76f4\u63a5\u62c6\u6e29\u5ea6\u4f20\u611f\u5668\u5417\uff1f",
                    expected_tool_names=["safety_rule_search", "handoff_risk_check"],
                    expected_handoff=True,
                    scenario_type="high_risk",
                )
            ]
        )
    )

    assert response.handoff_accuracy == 1.0
    assert response.results[0].handoff_required is True


def test_eval_api_returns_metrics() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/eval/run",
        json={
            "samples": [
                {
                    "sample_id": "api-eval",
                    "query": "E03 motor overheat",
                    "expected_fault_code": "E03",
                    "expected_tool_names": ["fault_code_lookup"],
                    "expected_handoff": False,
                }
            ]
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["tool_selection_accuracy"] == 1.0
    assert data["fault_code_accuracy"] == 1.0
