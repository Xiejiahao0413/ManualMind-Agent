import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolGuardDecision:
    allowed: bool
    reason: str
    args_signature: str | None = None


@dataclass
class ToolCallGuard:
    tool_whitelist: set[str]
    max_retry: int = 3
    called_signatures: set[str] = field(default_factory=set)

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
    ) -> ToolGuardDecision:
        if tool_name not in self.tool_whitelist:
            return ToolGuardDecision(False, "tool_not_whitelisted")

        missing_args = sorted((required_args or set()) - set(args.keys()))
        if missing_args:
            return ToolGuardDecision(False, f"missing_required_args:{','.join(missing_args)}")

        if retry_count >= self.max_retry:
            return ToolGuardDecision(False, "max_retry_reached")

        args_signature = self.build_args_signature(tool_name, args)
        if args_signature in self.called_signatures:
            return ToolGuardDecision(False, "duplicate_tool_call_signature", args_signature)

        return ToolGuardDecision(True, "allowed", args_signature)

    def remember_signature(self, args_signature: str) -> None:
        self.called_signatures.add(args_signature)
