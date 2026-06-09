from dataclasses import dataclass
from typing import Any

from app.tools.retry_fallback import ToolErrorType


@dataclass(frozen=True)
class ToolResultValidation:
    valid: bool
    error_type: ToolErrorType | None = None
    reason: str = "ok"


class ToolResultValidator:
    def __init__(self, min_score: float = 0.2) -> None:
        self.min_score = min_score

    def validate(
        self,
        result: dict[str, Any] | list[dict[str, Any]] | None,
        risk_level: str = "unknown",
    ) -> ToolResultValidation:
        if result is None or result == {} or result == []:
            return ToolResultValidation(False, ToolErrorType.INSUFFICIENT_EVIDENCE, "empty_result")

        records = result if isinstance(result, list) else [result]
        has_source_refs = any(bool(record.get("source_refs")) for record in records)
        has_safety_evidence = any(bool(record.get("safety_evidence")) for record in records)
        scores = [float(record["score"]) for record in records if record.get("score") is not None]

        if not has_source_refs:
            return ToolResultValidation(
                False,
                ToolErrorType.INSUFFICIENT_EVIDENCE,
                "missing_source_refs",
            )

        if scores and max(scores) < self.min_score:
            return ToolResultValidation(
                False,
                ToolErrorType.INSUFFICIENT_EVIDENCE,
                "retrieval_score_below_threshold",
            )

        if risk_level == "high" and not has_safety_evidence:
            return ToolResultValidation(
                False,
                ToolErrorType.HIGH_RISK_WITHOUT_EVIDENCE,
                "high_risk_without_safety_evidence",
            )

        return ToolResultValidation(True)
