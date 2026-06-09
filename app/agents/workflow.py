import re
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from app.core.dependencies import get_memory_manager
from app.memory import InMemoryMemoryManager
from app.security import sanitize_text
from app.schemas.diagnosis import DiagnosisRequest, DiagnosisState
from app.schemas.retrieval import RetrievalResult
from app.schemas.tools import ToolCallRecord
from app.tools import CircuitBreaker, DEFAULT_TOOL_WHITELIST, ToolCallGuard, ToolIntent, ToolRouter

class WorkflowState(TypedDict, total=False):
    task_id: str
    session_id: str
    user_query: str
    raw_query: str | None
    sanitized_query: str | None
    sanitized_fields: list[str]
    security_events: list[dict[str, Any]]
    device_name: str | None
    device_model: str | None
    fault_code: str | None
    symptoms: list[str]
    query_type: str | None
    risk_level: str
    retrieval_status: str
    retrieved_chunks: list[dict[str, Any]]
    source_refs: list[str]
    retrieved_evidence: list[dict[str, Any]]
    tool_call_history: list[dict[str, Any]]
    retry_count: int
    max_retry: int
    handoff_required: bool
    handoff_reason: str | None
    handoff_payload: dict[str, Any] | None
    final_answer: str | None


FAULT_CODE_PATTERN = re.compile(r"(?<![A-Z0-9])([EFP]\d{2,3})(?![A-Z0-9])", re.IGNORECASE)
HIGH_RISK_KEYWORDS = ("高压", "带电", "带压", "拆卸", "更换", "过热")
PARAMETER_KEYWORDS = ("温度", "电压", "压力", "阈值", "维护周期")


def _state_from_mapping(state: WorkflowState) -> DiagnosisState:
    return DiagnosisState.model_validate(state)


def _state_to_mapping(state: DiagnosisState) -> WorkflowState:
    return state.model_dump()


class DiagnosisWorkflow:
    def __init__(self, memory: InMemoryMemoryManager | None = None) -> None:
        self.memory = memory or get_memory_manager()
        self.guard = ToolCallGuard(tool_whitelist=DEFAULT_TOOL_WHITELIST)
        self.router = ToolRouter(self.guard)
        self.circuit_breaker = CircuitBreaker()
        self.graph = create_diagnosis_graph(self)

    async def run(self, state: DiagnosisState) -> DiagnosisState:
        result = await self.graph.ainvoke(_state_to_mapping(state))
        final_state = _state_from_mapping(result)
        await self.memory.save_task(final_state)
        return final_state

    @classmethod
    def from_request(cls, request: DiagnosisRequest) -> DiagnosisState:
        sanitized = sanitize_text(request.message)
        sanitized_fields = sorted({span.replacement.strip("[]") for span in sanitized.spans})
        security_events = []
        if sanitized.has_sensitive_data:
            security_events.append(
                {
                    "event_type": "sensitive_data_detected",
                    "sanitized_fields": sanitized_fields,
                    "action_taken": "masked",
                }
            )
        return DiagnosisState(
            task_id=request.task_id or f"task-{uuid4().hex}",
            session_id=request.session_id,
            raw_query=request.message,
            user_query=sanitized.sanitized_text,
            sanitized_query=sanitized.sanitized_text,
            sanitized_fields=sanitized_fields,
            security_events=security_events,
        )

    def supervisor_node(self, state: WorkflowState) -> WorkflowState:
        return state

    def supervisor_route(self, state: WorkflowState) -> str:
        diagnosis_state = _state_from_mapping(state)
        if diagnosis_state.retry_count >= diagnosis_state.max_retry:
            return "circuit_breaker"
        if diagnosis_state.handoff_required:
            return "handoff"
        if not diagnosis_state.query_type:
            return "diagnosis"
        if not diagnosis_state.retrieved_chunks:
            return "retrieval"
        return "safety_report"

    async def diagnosis_node(self, state: WorkflowState) -> WorkflowState:
        diagnosis_state = _state_from_mapping(state)
        query = diagnosis_state.sanitized_query or diagnosis_state.user_query
        normalized_query = query.upper()
        fault_match = FAULT_CODE_PATTERN.search(normalized_query)

        if fault_match:
            diagnosis_state.fault_code = fault_match.group(1).upper()

        detected_symptoms = [keyword for keyword in HIGH_RISK_KEYWORDS if keyword in query]
        if detected_symptoms:
            diagnosis_state.symptoms = sorted(set(diagnosis_state.symptoms + detected_symptoms))

        if any(keyword in query for keyword in HIGH_RISK_KEYWORDS):
            diagnosis_state.risk_level = "high"
            diagnosis_state.query_type = "high_risk_operation"
        elif any(keyword in query for keyword in PARAMETER_KEYWORDS):
            diagnosis_state.risk_level = "low"
            diagnosis_state.query_type = "parameter"
        elif diagnosis_state.fault_code:
            diagnosis_state.risk_level = "medium"
            diagnosis_state.query_type = "fault_code"
        else:
            diagnosis_state.risk_level = "medium"
            diagnosis_state.query_type = "general_fault_symptom"

        diagnosis_state.sanitized_query = query
        await self.memory.save_task(diagnosis_state)
        return _state_to_mapping(diagnosis_state)

    async def retrieval_node(self, state: WorkflowState) -> WorkflowState:
        diagnosis_state = _state_from_mapping(state)
        query_type = diagnosis_state.query_type or "general_fault_symptom"
        tool_names = self.router.route_query_type(query_type)
        chunks: list[dict[str, Any]] = []
        source_refs: list[str] = []
        evidence: list[RetrievalResult] = []

        for tool_name in tool_names:
            intent = ToolIntent(
                tool_name=tool_name,
                args={
                    "query": diagnosis_state.sanitized_query or diagnosis_state.user_query,
                    "fault_code": diagnosis_state.fault_code,
                },
                required_args={"query"},
            )
            decision = self.router.validate_intent(
                intent,
                retry_count=diagnosis_state.retry_count,
                task_tool_call_count=len(diagnosis_state.tool_call_history),
            )
            if not decision.args_signature:
                diagnosis_state.handoff_required = (
                    decision.status == "circuit_breaker_required"
                    or decision.reason == "sanitized_required"
                )
                diagnosis_state.handoff_reason = decision.reason if diagnosis_state.handoff_required else None
                continue

            record = ToolCallRecord(
                task_id=diagnosis_state.task_id,
                tool_name=tool_name,
                args=intent.args,
                args_signature=decision.args_signature,
                status="success" if decision.allowed else "skipped",
                retry_count=diagnosis_state.retry_count,
                result_summary=decision.reason,
            )
            diagnosis_state.tool_call_history.append(record)
            await self.memory.append_tool_call(record)
            if decision.allowed:
                self.guard.remember_success(record)

            if not decision.allowed:
                continue

            chunk = self._mock_chunk_for_tool(tool_name, diagnosis_state)
            chunks.append(chunk)
            source_refs.append(str(chunk["source_ref"]))
            evidence.append(
                RetrievalResult(
                    chunk_id=str(chunk["chunk_id"]),
                    doc_id=str(chunk["doc_id"]),
                    text=str(chunk["text"]),
                    score=float(chunk["score"]),
                    source_file=str(chunk["source_ref"]),
                    fault_code=diagnosis_state.fault_code,
                    content_type=str(chunk["content_type"]),
                )
            )

        diagnosis_state.retrieved_chunks = chunks
        diagnosis_state.source_refs = source_refs
        diagnosis_state.retrieved_evidence = evidence
        diagnosis_state.retrieval_status = "completed" if chunks else "empty"
        await self.memory.save_task(diagnosis_state)
        return _state_to_mapping(diagnosis_state)

    async def safety_report_node(self, state: WorkflowState) -> WorkflowState:
        diagnosis_state = _state_from_mapping(state)
        has_safety_evidence = any(
            bool(chunk.get("safety_evidence")) for chunk in diagnosis_state.retrieved_chunks
        )
        if diagnosis_state.risk_level == "high" and not has_safety_evidence:
            diagnosis_state.handoff_required = True
            diagnosis_state.handoff_reason = "high_risk_without_evidence"
            await self.memory.save_task(diagnosis_state)
            return _state_to_mapping(diagnosis_state)

        citations = ", ".join(diagnosis_state.source_refs) or "no_source_refs"
        safety_notice = "High-risk operation detected. Follow safety procedures before action." if (
            diagnosis_state.risk_level == "high"
        ) else "No high-risk operation detected in this mock workflow."
        diagnosis_state.final_answer = (
            f"Diagnosis summary: query_type={diagnosis_state.query_type}; "
            f"fault_code={diagnosis_state.fault_code or 'not_detected'}; "
            f"risk_level={diagnosis_state.risk_level}. "
            f"{safety_notice} Sources: {citations}."
        )
        await self.memory.save_task(diagnosis_state)
        return _state_to_mapping(diagnosis_state)

    async def circuit_breaker_node(self, state: WorkflowState) -> WorkflowState:
        diagnosis_state = _state_from_mapping(state)
        diagnosis_state.handoff_required = True
        diagnosis_state.handoff_reason = "circuit_breaker_retry_limit_reached"
        await self.memory.save_task(diagnosis_state)
        return _state_to_mapping(diagnosis_state)

    async def handoff_node(self, state: WorkflowState) -> WorkflowState:
        diagnosis_state = _state_from_mapping(state)
        reason = diagnosis_state.handoff_reason or "workflow_handoff_required"
        payload = self.circuit_breaker.create_handoff_payload(
            state=diagnosis_state,
            handoff_reason=reason,
            tool_trace=diagnosis_state.tool_call_history,
        )
        diagnosis_state.handoff_payload = payload.model_dump()
        diagnosis_state.final_answer = f"Human handoff required: {reason}."
        await self.memory.append_event(
            diagnosis_state.task_id,
            {
                "event": "handoff_required",
                "reason": reason,
                "risk_level": diagnosis_state.risk_level,
            },
        )
        await self.memory.save_task(diagnosis_state)
        return _state_to_mapping(diagnosis_state)

    def _mock_chunk_for_tool(
        self,
        tool_name: str,
        state: DiagnosisState,
    ) -> dict[str, str | float | bool | None]:
        safety_evidence = tool_name == "safety_rule_search"
        return {
            "chunk_id": f"mock-{tool_name}-{state.fault_code or 'general'}",
            "doc_id": "mock-manual",
            "text": f"Mock evidence from {tool_name} for {state.user_query}",
            "score": 0.85,
            "source_ref": f"mock_manual::{tool_name}",
            "content_type": "safety_rule" if safety_evidence else "manual_chunk",
            "safety_evidence": safety_evidence,
        }


def create_diagnosis_graph(workflow: DiagnosisWorkflow | None = None):
    owner = workflow
    if owner is None:
        owner = DiagnosisWorkflow.__new__(DiagnosisWorkflow)
        owner.memory = get_memory_manager()
        owner.guard = ToolCallGuard(tool_whitelist=DEFAULT_TOOL_WHITELIST)
        owner.router = ToolRouter(owner.guard)
        owner.circuit_breaker = CircuitBreaker()

    graph = StateGraph(WorkflowState)
    graph.add_node("supervisor_node", owner.supervisor_node)
    graph.add_node("diagnosis_node", owner.diagnosis_node)
    graph.add_node("retrieval_node", owner.retrieval_node)
    graph.add_node("safety_report_node", owner.safety_report_node)
    graph.add_node("circuit_breaker_node", owner.circuit_breaker_node)
    graph.add_node("handoff_node", owner.handoff_node)

    graph.set_entry_point("supervisor_node")
    graph.add_conditional_edges(
        "supervisor_node",
        owner.supervisor_route,
        {
            "diagnosis": "diagnosis_node",
            "retrieval": "retrieval_node",
            "safety_report": "safety_report_node",
            "circuit_breaker": "circuit_breaker_node",
            "handoff": "handoff_node",
        },
    )
    graph.add_edge("diagnosis_node", "supervisor_node")
    graph.add_edge("retrieval_node", "supervisor_node")
    graph.add_conditional_edges(
        "safety_report_node",
        lambda state: "handoff" if state.get("handoff_required") else "done",
        {
            "handoff": "handoff_node",
            "done": END,
        },
    )
    graph.add_edge("circuit_breaker_node", "handoff_node")
    graph.add_edge("handoff_node", END)
    return graph.compile()
