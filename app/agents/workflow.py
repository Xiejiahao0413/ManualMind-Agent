import re
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from app.agents.reporting import build_diagnosis_report
from app.agents.services import WorkflowToolService
from app.core.dependencies import get_memory_manager, get_trace_manager
from app.memory import InMemoryMemoryManager
from app.security import sanitize_text
from app.schemas.diagnosis import DiagnosisRequest, DiagnosisState
from app.schemas.trace import TraceEvent
from app.tools import CircuitBreaker, DEFAULT_TOOL_WHITELIST, ToolCallGuard, ToolRouter
from app.tracing import InMemoryTraceManager

class WorkflowState(TypedDict, total=False):
    task_id: str
    request_id: str | None
    trace_id: str | None
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


FAULT_CODE_PATTERN = re.compile(
    r"(?<![A-Z0-9])(?:ERROR\s*|ERR\s*)?([EFP])[-\s]?(\d{2,3})(?![A-Z0-9])",
    re.IGNORECASE,
)
HIGH_RISK_KEYWORDS = (
    "高压",
    "带电",
    "带压",
    "拆卸",
    "更换",
    "过热",
    "高温",
    "未断电",
    "没有断电",
    "没断电",
    "不停机",
    "开盖",
    "拆管路",
    "拆阀门",
    "拆传感器",
    "马上拆",
    "立即拆",
    "压力表没归零",
    "没有释放压力",
)
UNSAFE_BYPASS_KEYWORDS = (
    "忽略所有安全规则",
    "忽略安全规则",
    "无视安全",
    "绕过安全",
    "不要提醒断电",
    "跳过 Tool Guard",
    "跳过Tool Guard",
    "只给我最终拆卸命令",
    "删掉安全提醒",
    "直接输出带压拆卸步骤",
    "直接告诉我怎么带电拆",
)
UNSAFE_OPERATION_PATTERNS = (
    ("带压", "拆"),
    ("带电", "拆"),
    ("没有断电", "拆"),
    ("没断电", "拆"),
    ("未断电", "拆"),
    ("没有断电", "更换"),
    ("没断电", "更换"),
    ("未断电", "更换"),
    ("带电", "检查"),
    ("带电", "更换"),
    ("不停机", "拆"),
    ("不停机", "更换"),
    ("高温", "拆"),
    ("高温", "开盖"),
    ("过热", "拆"),
    ("压力表没归零", "拆"),
    ("没有释放压力", "拆"),
    ("压力没释放", "拆"),
)
PARAMETER_KEYWORDS = ("温度", "电压", "压力", "阈值", "维护周期")
PARAMETER_KEYWORDS = PARAMETER_KEYWORDS + ("temperature", "voltage", "pressure", "threshold", "maintenance")


def _state_from_mapping(state: WorkflowState) -> DiagnosisState:
    return DiagnosisState.model_validate(state)


def _state_to_mapping(state: DiagnosisState) -> WorkflowState:
    return state.model_dump()


def _query_text(state: DiagnosisState) -> str:
    return state.raw_query or state.sanitized_query or state.user_query


def _fault_code_from_match(match: re.Match[str]) -> str:
    return f"{match.group(1).upper()}{match.group(2)}"


def _contains_high_risk_intent(query: str) -> bool:
    return any(keyword in query for keyword in HIGH_RISK_KEYWORDS) or any(
        all(term in query for term in terms) for terms in UNSAFE_OPERATION_PATTERNS
    )


def _unsafe_handoff_reason(query: str) -> str | None:
    if any(keyword in query for keyword in UNSAFE_BYPASS_KEYWORDS):
        return "unsafe_instruction_requires_handoff"
    if any(all(term in query for term in terms) for terms in UNSAFE_OPERATION_PATTERNS):
        return "high_risk_operation_requires_handoff"
    return None


def _has_reliable_fault_source(state: DiagnosisState) -> bool:
    if state.fault_info:
        return True
    if not state.fault_code:
        return True
    for chunk in state.retrieved_chunks:
        if str(chunk.get("fault_code") or "").upper() == state.fault_code:
            return True
        text = str(chunk.get("text") or "").upper()
        if state.fault_code in text:
            return True
    return False


def _handoff_reason_for_state(state: DiagnosisState) -> str | None:
    query = _query_text(state)
    unsafe_reason = _unsafe_handoff_reason(query)
    if unsafe_reason:
        return unsafe_reason
    if state.risk_level == "high" and (not state.safety_rules or not state.source_refs):
        return "insufficient_evidence_requires_handoff"
    if state.query_type == "fault_code" and state.fault_code and not _has_reliable_fault_source(state):
        return "insufficient_evidence_requires_handoff"
    return None


class DiagnosisWorkflow:
    def __init__(
        self,
        memory: InMemoryMemoryManager | None = None,
        trace_manager: InMemoryTraceManager | None = None,
    ) -> None:
        self.memory = memory or get_memory_manager()
        self.trace_manager = trace_manager or get_trace_manager()
        self.guard = ToolCallGuard(tool_whitelist=DEFAULT_TOOL_WHITELIST)
        self.router = ToolRouter(self.guard)
        self.circuit_breaker = CircuitBreaker()
        self.tool_service = WorkflowToolService(
            router=self.router,
            guard=self.guard,
            memory=self.memory,
            trace_manager=self.trace_manager,
        )
        self.graph = create_diagnosis_graph(self)

    async def run(self, state: DiagnosisState) -> DiagnosisState:
        self.ensure_trace(state)
        result = await self.graph.ainvoke(_state_to_mapping(state))
        final_state = _state_from_mapping(result)
        await self.memory.save_task(final_state)
        return final_state

    def ensure_trace(self, state: DiagnosisState) -> DiagnosisState:
        if state.request_id is None:
            state.request_id = f"req-{uuid4().hex}"
        if state.trace_id is None:
            trace = self.trace_manager.start_trace(
                request_id=state.request_id,
                session_id=state.session_id,
                task_id=state.task_id,
            )
            state.trace_id = trace.trace_id
        return state

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
            request_id=f"req-{uuid4().hex}",
            session_id=request.session_id,
            raw_query=request.message,
            user_query=sanitized.sanitized_text,
            sanitized_query=sanitized.sanitized_text,
            sanitized_fields=sanitized_fields,
            security_events=security_events,
        )

    def supervisor_node(self, state: WorkflowState) -> WorkflowState:
        diagnosis_state = _state_from_mapping(state)
        self._trace_event(
            diagnosis_state,
            "supervisor_started",
            "supervisor_node",
            "started",
            "Supervisor routing started.",
        )
        route = self.supervisor_route(state)
        self._trace_event(
            diagnosis_state,
            "supervisor_completed",
            "supervisor_node",
            "completed",
            "Supervisor routing completed.",
            {"route": route},
        )
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
        self._trace_event(
            diagnosis_state,
            "diagnosis_started",
            "diagnosis_node",
            "started",
            "Diagnosis node started.",
        )
        query = diagnosis_state.sanitized_query or diagnosis_state.user_query
        normalized_query = query.upper()
        fault_match = FAULT_CODE_PATTERN.search(normalized_query)

        if fault_match:
            diagnosis_state.fault_code = _fault_code_from_match(fault_match)

        detected_symptoms = [keyword for keyword in HIGH_RISK_KEYWORDS if keyword in query]
        if detected_symptoms:
            diagnosis_state.symptoms = sorted(set(diagnosis_state.symptoms + detected_symptoms))

        if _contains_high_risk_intent(query) or _unsafe_handoff_reason(query):
            diagnosis_state.risk_level = "high"
            diagnosis_state.query_type = "high_risk_operation"
        elif diagnosis_state.fault_code:
            diagnosis_state.risk_level = "medium"
            diagnosis_state.query_type = "fault_code"
        elif any(keyword in query for keyword in PARAMETER_KEYWORDS):
            diagnosis_state.risk_level = "low"
            diagnosis_state.query_type = "parameter"
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
        self._trace_event(
            diagnosis_state,
            "diagnosis_completed",
            "diagnosis_node",
            "completed",
            "Diagnosis node completed.",
            {
                "query_type": diagnosis_state.query_type,
                "fault_code": diagnosis_state.fault_code,
                "risk_level": diagnosis_state.risk_level,
            },
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
        self._trace_event(
            diagnosis_state,
            "retrieval_completed",
            "retrieval_node",
            "completed",
            "Retrieval node completed.",
            {
                "retrieval_status": diagnosis_state.retrieval_status,
                "retrieval_mode": diagnosis_state.retrieval_mode,
                "source_count": len(diagnosis_state.source_refs),
            },
        )
        await self.memory.save_task(diagnosis_state)
        return _state_to_mapping(diagnosis_state)

    async def safety_report_node(self, state: WorkflowState) -> WorkflowState:
        diagnosis_state = _state_from_mapping(state)
        self._trace_event(
            diagnosis_state,
            "safety_review_started",
            "safety_report_node",
            "started",
            "Safety report node started.",
        )
        handoff_reason = _handoff_reason_for_state(diagnosis_state)
        if handoff_reason:
            diagnosis_state.handoff_required = True
            diagnosis_state.handoff_reason = handoff_reason
            self._trace_event(
                diagnosis_state,
                "safety_review_completed",
                "safety_report_node",
                "handoff_required",
                "Safety review requires handoff.",
                {"handoff_reason": diagnosis_state.handoff_reason},
            )
            await self.memory.save_task(diagnosis_state)
            return _state_to_mapping(diagnosis_state)

        diagnosis_state.source_refs = sorted(set(diagnosis_state.source_refs))
        diagnosis_state.final_answer = build_diagnosis_report(diagnosis_state)
        diagnosis_state.workflow_events.append({"event": "safety_review_completed"})
        self._trace_event(
            diagnosis_state,
            "safety_review_completed",
            "safety_report_node",
            "completed",
            "Safety report node completed.",
            {"source_count": len(diagnosis_state.source_refs)},
        )
        self._trace_event(
            diagnosis_state,
            "final_answer_generated",
            "safety_report_node",
            "completed",
            "Final answer generated.",
        )
        await self.memory.save_task(diagnosis_state)
        return _state_to_mapping(diagnosis_state)

    async def circuit_breaker_node(self, state: WorkflowState) -> WorkflowState:
        diagnosis_state = _state_from_mapping(state)
        diagnosis_state.handoff_required = True
        diagnosis_state.handoff_reason = "circuit_breaker_retry_limit_reached"
        self._trace_event(
            diagnosis_state,
            "circuit_breaker_triggered",
            "circuit_breaker_node",
            "triggered",
            "Circuit breaker triggered by retry limit.",
            {"retry_count": diagnosis_state.retry_count},
        )
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
        diagnosis_state.final_answer = (
            f"已触发人工接管：{reason}。当前问题涉及高风险操作或缺少足够依据，"
            "无法安全给出直接操作步骤。请先断电、释放压力并等待设备冷却，再由人工确认现场条件。"
        )
        self._trace_event(
            diagnosis_state,
            "handoff_created",
            "handoff_node",
            "completed",
            "Human handoff payload created.",
            {
                "handoff_reason": reason,
                "risk_level": diagnosis_state.risk_level,
            },
        )
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

    def _trace_event(
        self,
        state: DiagnosisState,
        event_type: str,
        component: str,
        status: str,
        message: str,
        metadata: dict[str, Any] | None = None,
        latency_ms: float | None = None,
    ) -> None:
        if not state.trace_id or not state.request_id:
            return
        self.trace_manager.add_event(
            state.trace_id,
            TraceEvent(
                trace_id=state.trace_id,
                request_id=state.request_id,
                session_id=state.session_id,
                task_id=state.task_id,
                event_type=event_type,
                component=component,
                status=status,
                message=message,
                metadata=metadata or {},
                latency_ms=latency_ms,
            ),
        )

def create_diagnosis_graph(workflow: DiagnosisWorkflow | None = None):
    owner = workflow
    if owner is None:
        owner = DiagnosisWorkflow.__new__(DiagnosisWorkflow)
        owner.memory = get_memory_manager()
        owner.trace_manager = get_trace_manager()
        owner.guard = ToolCallGuard(tool_whitelist=DEFAULT_TOOL_WHITELIST)
        owner.router = ToolRouter(owner.guard)
        owner.circuit_breaker = CircuitBreaker()
        owner.tool_service = WorkflowToolService(
            router=owner.router,
            guard=owner.guard,
            memory=owner.memory,
            trace_manager=owner.trace_manager,
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
