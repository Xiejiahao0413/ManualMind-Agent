import re
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from app.agents.services import WorkflowToolService
from app.core.dependencies import get_memory_manager
from app.memory import InMemoryMemoryManager
from app.security import sanitize_text
from app.schemas.diagnosis import DiagnosisRequest, DiagnosisState
from app.tools import CircuitBreaker, DEFAULT_TOOL_WHITELIST, ToolCallGuard, ToolRouter

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
    retrieval_mode: str | None
    retrieved_chunks: list[dict[str, Any]]
    source_refs: list[str]
    retrieved_evidence: list[dict[str, Any]]
    fault_info: dict[str, Any] | None
    parameter_info: dict[str, Any] | None
    safety_rules: list[str]
    fallback_decision: str | None
    workflow_events: list[dict[str, Any]]
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


HIGH_RISK_KEYWORDS = HIGH_RISK_KEYWORDS + ("高压", "带电", "带压", "拆卸", "更换", "过热")
PARAMETER_KEYWORDS = PARAMETER_KEYWORDS + ("温度", "电压", "压力", "阈值", "维护周期")
PARAMETER_KEYWORDS = PARAMETER_KEYWORDS + ("temperature", "voltage", "pressure", "threshold", "maintenance")


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
        self.tool_service = WorkflowToolService(
            router=self.router,
            guard=self.guard,
            memory=self.memory,
        )
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
        if diagnosis_state.retrieval_status == "not_started":
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
        diagnosis_state.workflow_events.append(
            {
                "event": "diagnosis_completed",
                "query_type": diagnosis_state.query_type,
                "fault_code": diagnosis_state.fault_code,
                "risk_level": diagnosis_state.risk_level,
            }
        )
        await self.memory.save_task(diagnosis_state)
        return _state_to_mapping(diagnosis_state)

    async def retrieval_node(self, state: WorkflowState) -> WorkflowState:
        diagnosis_state = _state_from_mapping(state)
        diagnosis_state.workflow_events.append({"event": "tool_routing_started"})
        diagnosis_state = await self.tool_service.run_retrieval_tools(diagnosis_state)
        diagnosis_state.workflow_events.append(
            {
                "event": "retrieval_completed",
                "retrieval_status": diagnosis_state.retrieval_status,
            }
        )
        await self.memory.save_task(diagnosis_state)
        return _state_to_mapping(diagnosis_state)

    async def safety_report_node(self, state: WorkflowState) -> WorkflowState:
        diagnosis_state = _state_from_mapping(state)
        if diagnosis_state.risk_level == "high" and (
            not diagnosis_state.safety_rules or not diagnosis_state.source_refs
        ):
            diagnosis_state.handoff_required = True
            diagnosis_state.handoff_reason = "high_risk_without_evidence"
            await self.memory.save_task(diagnosis_state)
            return _state_to_mapping(diagnosis_state)

        citations = ", ".join(diagnosis_state.source_refs) or "no_source_refs"
        fault_summary = (
            diagnosis_state.fault_info.get("description")
            if diagnosis_state.fault_info
            else f"fault_code={diagnosis_state.fault_code or 'not_detected'}"
        )
        possible_cause = (
            fault_summary
            if diagnosis_state.fault_info
            else "Manual evidence suggests checking device-specific operating conditions."
        )
        troubleshooting_steps = (
            "Review retrieved manual evidence. "
            "Check the fault code or parameter range against the cited source. "
            "Record observations before corrective action."
        )
        safety_notice = (
            " ".join(diagnosis_state.safety_rules)
            if diagnosis_state.safety_rules
            else "No high-risk operation detected by current tool results."
        )
        diagnosis_state.final_answer = (
            f"故障识别: {fault_summary}. "
            f"可能原因: {possible_cause}. "
            f"排查步骤: {troubleshooting_steps} "
            f"安全提醒: {safety_notice}. "
            f"参数信息: {diagnosis_state.parameter_info or 'not_applicable'}. "
            f"引用来源: {citations}."
        )
        diagnosis_state.workflow_events.append({"event": "safety_review_completed"})
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

def create_diagnosis_graph(workflow: DiagnosisWorkflow | None = None):
    owner = workflow
    if owner is None:
        owner = DiagnosisWorkflow.__new__(DiagnosisWorkflow)
        owner.memory = get_memory_manager()
        owner.guard = ToolCallGuard(tool_whitelist=DEFAULT_TOOL_WHITELIST)
        owner.router = ToolRouter(owner.guard)
        owner.circuit_breaker = CircuitBreaker()
        owner.tool_service = WorkflowToolService(
            router=owner.router,
            guard=owner.guard,
            memory=owner.memory,
        )

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
