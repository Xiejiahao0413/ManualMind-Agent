from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMReportContext:
    query: str
    query_type: str | None = None
    device_model: str | None = None
    fault_code: str | None = None
    risk_level: str = "unknown"
    evidence_snippets: list[str] = field(default_factory=list)
    evidence_items: list[dict[str, str]] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    safety_warnings: list[str] = field(default_factory=list)
    handoff_required: bool = False


@dataclass(frozen=True)
class LLMReportResult:
    success: bool
    text: str = ""
    provider: str = "template"
    model: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    fallback_required: bool = False


class BaseLLMClient(ABC):
    provider: str = "unknown"

    @abstractmethod
    async def generate_report(self, context: LLMReportContext) -> LLMReportResult:
        raise NotImplementedError
