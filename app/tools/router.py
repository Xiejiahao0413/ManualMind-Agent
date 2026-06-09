from dataclasses import dataclass, field
from typing import Any

from app.tools.guard import ToolCallGuard, ToolGuardDecision


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
    ) -> ToolGuardDecision:
        return self.guard.check(
            tool_name=intent.tool_name,
            args=intent.args,
            retry_count=retry_count,
            required_args=intent.required_args,
        )
