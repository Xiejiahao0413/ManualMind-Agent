import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from app.schemas.tools import ToolCallRecord

ToolGuardStatus = Literal["allowed", "reuse_cached", "blocked", "circuit_breaker_required"]


@dataclass(frozen=True)
class ToolGuardDecision:
    allowed: bool
    reason: str
    args_signature: str | None = None
    status: ToolGuardStatus = "blocked"
    cached_record: ToolCallRecord | None = None


@dataclass
class ToolCallGuard:
    tool_whitelist: set[str]
    max_retry: int = 3
    max_tool_calls_per_task: int = 8
    called_signatures: set[str] = field(default_factory=set)
    successful_records: dict[str, ToolCallRecord] = field(default_factory=dict)

    def build_args_signature(self, tool_name: str, args: dict[str, Any]) -> str:
        normalized = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
        raw_signature = f"{tool_name}:{normalized}"
        return hashlib.sha256(raw_signature.encode("utf-8")).hexdigest()

    def check(
        self,
        tool_name: str,
        args: dict[str, Any],
        retry_count: int = 0,
        required_args: set[str] | None = None,
        task_tool_call_count: int = 0,
    ) -> ToolGuardDecision:
        if tool_name not in self.tool_whitelist:
            return ToolGuardDecision(False, "tool_not_whitelisted")

        missing_args = sorted((required_args or set()) - set(args.keys()))
        if missing_args:
            return ToolGuardDecision(False, f"missing_required_args:{','.join(missing_args)}")

        if task_tool_call_count >= self.max_tool_calls_per_task:
            return ToolGuardDecision(
                False,
                "max_tool_calls_per_task_reached",
                status="circuit_breaker_required",
            )

        if retry_count >= self.max_retry:
            return ToolGuardDecision(False, "max_retry_reached", status="circuit_breaker_required")

        args_signature = self.build_args_signature(tool_name, args)
        cached_record = self.successful_records.get(args_signature)
        if cached_record is not None:
            return ToolGuardDecision(
                True,
                "duplicate_successful_call_reuse_cached",
                args_signature,
                status="reuse_cached",
                cached_record=cached_record,
            )

        if args_signature in self.called_signatures:
            return ToolGuardDecision(
                False,
                "duplicate_tool_call_signature",
                args_signature,
                status="blocked",
            )

        return ToolGuardDecision(True, "allowed", args_signature, status="allowed")

    def remember_signature(self, args_signature: str) -> None:
        self.called_signatures.add(args_signature)

    def remember_success(self, record: ToolCallRecord) -> None:
        self.called_signatures.add(record.args_signature)
        self.successful_records[record.args_signature] = record
