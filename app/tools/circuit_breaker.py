from dataclasses import dataclass

from app.schemas.diagnosis import DiagnosisState
from app.schemas.handoff import HandoffPayload
from app.schemas.tools import ToolCallRecord


@dataclass(frozen=True)
class CircuitBreakerDecision:
    triggered: bool
    reason: str = "not_triggered"


class CircuitBreaker:
    def __init__(self, max_retry: int = 3, max_tool_calls_per_task: int = 8) -> None:
        self.max_retry = max_retry
        self.max_tool_calls_per_task = max_tool_calls_per_task

    def evaluate(
        self,
        retry_count: int,
        called_signatures: set[str],
        tool_call_count: int,
    ) -> CircuitBreakerDecision:
        if retry_count >= self.max_retry:
            return CircuitBreakerDecision(True, "max_retry_reached")

        if tool_call_count >= self.max_tool_calls_per_task:
            return CircuitBreakerDecision(True, "max_tool_calls_per_task_reached")

        if len(called_signatures) < tool_call_count:
            return CircuitBreakerDecision(True, "repeated_tool_signature_detected")

        return CircuitBreakerDecision(False)

    def create_handoff_payload(
        self,
        state: DiagnosisState,
        handoff_reason: str,
        tool_trace: list[ToolCallRecord],
    ) -> HandoffPayload:
        return HandoffPayload(
            task_id=state.task_id,
            session_id=state.session_id,
            reason=handoff_reason,
            handoff_reason=handoff_reason,
            sanitized_summary=state.user_query,
            risk_level=state.risk_level,
            device_name=state.device_name,
            fault_code=state.fault_code,
            symptoms=list(state.symptoms),
            tool_trace=[
                {
                    "tool_name": record.tool_name,
                    "args_signature": record.args_signature,
                    "status": record.status,
                    "retry_count": record.retry_count,
                    "fallback_used": record.fallback_used,
                    "error_type": record.error_type,
                    "latency_ms": record.latency_ms,
                }
                for record in tool_trace
            ],
            retrieved_sources=[
                evidence.source_file or evidence.doc_id for evidence in state.retrieved_evidence
            ],
        )
