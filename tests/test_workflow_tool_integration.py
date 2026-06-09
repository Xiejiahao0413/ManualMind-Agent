import asyncio

from fastapi.testclient import TestClient

from app.agents import DiagnosisWorkflow
from app.agents.services import WorkflowToolService
from app.main import app
from app.mcp_server import MCPTool, MCPToolExecutor, ToolRegistry, create_default_tool_registry
from app.mcp_server.schemas import (
    ManualHybridSearchRequest,
    ManualHybridSearchResponse,
    SafetyRuleSearchRequest,
    SafetyRuleSearchResponse,
)
from app.memory import InMemoryMemoryManager
from app.schemas.diagnosis import DiagnosisState
from app.schemas.retrieval import RetrievalResult
from app.tools import DEFAULT_TOOL_WHITELIST, ToolCallGuard, ToolRouter


def run_workflow(state: DiagnosisState, workflow: DiagnosisWorkflow | None = None) -> DiagnosisState:
    workflow = workflow or DiagnosisWorkflow(memory=InMemoryMemoryManager())
    return asyncio.run(workflow.run(state))


def test_retrieval_node_calls_manual_hybrid_search_through_mcp_executor() -> None:
    memory = InMemoryMemoryManager()
    result = run_workflow(
        DiagnosisState(
            task_id="task-manual",
            session_id="session-1",
            user_query="motor overheat fault",
        ),
        DiagnosisWorkflow(memory=memory),
    )

    tool_names = [record.tool_name for record in result.tool_call_history]
    assert "manual_hybrid_search" in tool_names
    assert result.retrieval_status == "completed"
    assert result.retrieved_chunks


def test_e03_question_calls_fault_code_lookup() -> None:
    result = run_workflow(
        DiagnosisState(
            task_id="task-e03-tool",
            session_id="session-1",
            user_query="E03 motor overheat",
        )
    )

    tool_names = [record.tool_name for record in result.tool_call_history]
    assert "fault_code_lookup" in tool_names
    assert result.fault_info is not None
    assert result.fault_info["status"] == "found"


def test_tool_memory_records_workflow_tool_calls() -> None:
    memory = InMemoryMemoryManager()
    state = DiagnosisState(
        task_id="task-memory-tool",
        session_id="session-1",
        user_query="E03 motor overheat",
    )

    result = run_workflow(state, DiagnosisWorkflow(memory=memory))
    records = asyncio.run(memory.list_tool_calls(result.task_id))

    assert records
    assert any(record.tool_name == "fault_code_lookup" for record in records)
    assert all(record.args_signature for record in records)


def test_same_args_signature_reuses_cached_tool_result() -> None:
    calls = {"count": 0}

    async def counted_manual_search(arguments: dict) -> ManualHybridSearchResponse:
        calls["count"] += 1
        return ManualHybridSearchResponse(
            results=[
                RetrievalResult(
                    chunk_id="cached-1",
                    doc_id="manual",
                    text="cached result",
                    score=1.0,
                    source_refs=["manual.pdf:1"],
                    source_file="manual.pdf",
                )
            ],
            source_refs=["manual.pdf:1"],
        )

    registry = ToolRegistry()
    registry.register(
        MCPTool(
            name="manual_hybrid_search",
            description="counted",
            input_schema=ManualHybridSearchRequest,
            execute=counted_manual_search,
        )
    )
    memory = InMemoryMemoryManager()
    guard = ToolCallGuard(tool_whitelist=DEFAULT_TOOL_WHITELIST)
    service = WorkflowToolService(
        router=ToolRouter(guard),
        guard=guard,
        memory=memory,
        executor=MCPToolExecutor(registry),
    )
    state = DiagnosisState(
        task_id="task-cache",
        session_id="session-1",
        user_query="same query",
        sanitized_query="same query",
        query_type="general_fault_symptom",
    )

    asyncio.run(service.run_retrieval_tools(state))
    state.retrieval_status = "not_started"
    asyncio.run(service.run_retrieval_tools(state))

    assert calls["count"] == 1
    assert any(record.result_summary == "reuse_cached" for record in state.tool_call_history)


def test_retry_limit_still_triggers_circuit_breaker() -> None:
    result = run_workflow(
        DiagnosisState(
            task_id="task-circuit-tool",
            session_id="session-1",
            user_query="motor fault",
            retry_count=3,
        )
    )

    assert result.handoff_required is True
    assert result.handoff_reason == "circuit_breaker_retry_limit_reached"
    assert result.handoff_payload is not None


def test_high_risk_without_safety_evidence_triggers_handoff() -> None:
    async def empty_safety(arguments: dict) -> SafetyRuleSearchResponse:
        return SafetyRuleSearchResponse(safety_rules=[], source_refs=[], risk_level="high")

    registry = create_default_tool_registry()
    registry.register(
        MCPTool(
            name="safety_rule_search",
            description="empty safety",
            input_schema=SafetyRuleSearchRequest,
            execute=empty_safety,
        )
    )
    memory = InMemoryMemoryManager()
    workflow = DiagnosisWorkflow(memory=memory)
    workflow.tool_service = WorkflowToolService(
        router=workflow.router,
        guard=workflow.guard,
        memory=memory,
        executor=MCPToolExecutor(registry),
    )

    result = run_workflow(
        DiagnosisState(
            task_id="task-high-risk-empty",
            session_id="session-1",
            user_query="高压带电拆卸",
        ),
        workflow,
    )

    assert result.handoff_required is True
    assert result.handoff_reason == "high_risk_without_evidence"
    assert result.handoff_payload is not None


def test_diagnosis_chat_returns_workflow_stage_events() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/diagnosis/chat",
        json={"session_id": "session-sse", "message": "E03 motor overheat"},
    )

    assert response.status_code == 200
    body = response.text
    assert "event: input_sanitized" in body
    assert "event: diagnosis_completed" in body
    assert "event: tool_routing_started" in body
    assert "event: tool_call_completed" in body
    assert "event: retrieval_completed" in body
    assert "event: safety_review_completed" in body
    assert "event: final_answer" in body
