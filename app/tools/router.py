from dataclasses import dataclass, field
from typing import Any

from app.tools.guard import ToolCallGuard, ToolGuardDecision

DEFAULT_TOOL_WHITELIST = {
    "manual_hybrid_search",
    "fault_code_lookup",
    "parameter_lookup",
    "safety_rule_search",
    "case_memory_search",
    "source_trace",
    "handoff_ticket_create",
    "handoff_risk_check",
}

QUERY_TYPE_TOOL_RULES: dict[str, list[str]] = {
    "fault_code": ["fault_code_lookup", "manual_hybrid_search"],
    "parameter": ["parameter_lookup"],
    "safety": ["safety_rule_search"],
    "general_fault_symptom": ["manual_hybrid_search"],
    "high_risk_operation": ["safety_rule_search", "handoff_risk_check"],
}


@dataclass(frozen=True)
class ToolIntent:
    tool_name: str
    args: dict[str, Any]
    required_args: set[str] = field(default_factory=set)


class ToolRouter:
    def __init__(self, guard: ToolCallGuard) -> None:
        self.guard = guard

    def validate_intent(
        self,
        intent: ToolIntent,
        retry_count: int = 0,
        task_tool_call_count: int = 0,
    ) -> ToolGuardDecision:
        return self.guard.check(
            tool_name=intent.tool_name,
            args=intent.args,
            retry_count=retry_count,
            required_args=intent.required_args,
            task_tool_call_count=task_tool_call_count,
        )

    def route_query_type(self, query_type: str) -> list[str]:
        return list(QUERY_TYPE_TOOL_RULES.get(query_type, ["manual_hybrid_search"]))
