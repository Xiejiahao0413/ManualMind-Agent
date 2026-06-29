import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.agents.reporting import build_diagnosis_report
from app.llm import BaseLLMClient, LLMReportContext, LLMReportResult, OpenAIReportClient
from app.schemas.diagnosis import DiagnosisState
from app.security import sanitize_text


REQUIRED_REPORT_SECTIONS = ("故障识别", "可能原因", "排查步骤", "安全提醒", "引用来源")


@dataclass(frozen=True)
class ReportGenerationResult:
    final_answer: str
    llm_enabled: bool
    llm_provider: str
    fallback_used: bool
    error_type: str | None = None
    sanitized_fields: list[str] | None = None


class ReportGenerationService:
    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.env = os.environ if env is None else env
        self.llm_client = llm_client

    async def generate(self, state: DiagnosisState) -> ReportGenerationResult:
        template_report = build_diagnosis_report(state)
        if state.handoff_required:
            return ReportGenerationResult(
                final_answer=template_report,
                llm_enabled=False,
                llm_provider="template",
                fallback_used=True,
                error_type="handoff_required",
            )

        client = self.llm_client or self._client_from_env()
        if client is None:
            return ReportGenerationResult(
                final_answer=template_report,
                llm_enabled=False,
                llm_provider="template",
                fallback_used=True,
                error_type="llm_disabled",
            )

        context = self._build_context(state)
        try:
            llm_result = await client.generate_report(context)
        except Exception:
            return self._fallback(template_report, getattr(client, "provider", "unknown"), "llm_exception")

        if not llm_result.success or llm_result.fallback_required:
            return self._fallback(template_report, llm_result.provider, llm_result.error_type or "llm_failed")

        sanitized = sanitize_text(llm_result.text)
        final_answer = self._ensure_source_refs(sanitized.sanitized_text.strip(), state.source_refs)
        if not self._is_valid_report(final_answer, state.source_refs):
            return self._fallback(template_report, llm_result.provider, "invalid_report_format")

        return ReportGenerationResult(
            final_answer=final_answer,
            llm_enabled=True,
            llm_provider=llm_result.provider,
            fallback_used=False,
            sanitized_fields=sorted({span.replacement.strip("[]") for span in sanitized.spans}),
        )

    def _client_from_env(self) -> BaseLLMClient | None:
        provider = self.env.get("MANUALMIND_LLM_PROVIDER", "").lower().strip()
        api_key = self.env.get("OPENAI_API_KEY", "").strip()
        if provider != "openai" or not api_key:
            return None
        model = self.env.get("MANUALMIND_LLM_MODEL", "gpt-4o-mini")
        timeout_seconds = float(self.env.get("MANUALMIND_LLM_TIMEOUT_SECONDS", "20"))
        max_tokens = int(self.env.get("MANUALMIND_LLM_MAX_TOKENS", "900"))
        temperature = float(self.env.get("MANUALMIND_LLM_TEMPERATURE", "0.2"))
        return OpenAIReportClient(
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _build_context(self, state: DiagnosisState) -> LLMReportContext:
        query = state.sanitized_query or state.user_query
        evidence_snippets = []
        for chunk in state.retrieved_chunks[:5]:
            text = sanitize_text(str(chunk.get("text") or "")).sanitized_text
            if text:
                evidence_snippets.append(text[:500])
        safety_warnings = [sanitize_text(str(item)).sanitized_text for item in state.safety_rules if item]
        return LLMReportContext(
            query=sanitize_text(query).sanitized_text,
            device_model=state.device_model,
            fault_code=state.fault_code,
            risk_level=state.risk_level,
            evidence_snippets=evidence_snippets,
            source_refs=sorted(set(state.source_refs)),
            safety_warnings=safety_warnings,
            handoff_required=state.handoff_required,
        )

    def _is_valid_report(self, report: str, source_refs: list[str]) -> bool:
        if not report.strip():
            return False
        if not all(section in report for section in REQUIRED_REPORT_SECTIONS):
            return False
        return all(source in report for source in source_refs)

    def _ensure_source_refs(self, report: str, source_refs: list[str]) -> str:
        missing_refs = [source for source in sorted(set(source_refs)) if source and source not in report]
        if not missing_refs:
            return report
        refs_block = "\n".join(f"- {source}" for source in missing_refs)
        if "引用来源" in report:
            return f"{report.rstrip()}\n{refs_block}"
        return f"{report.rstrip()}\n\n引用来源：\n{refs_block}"

    def _fallback(self, template_report: str, provider: str, error_type: str) -> ReportGenerationResult:
        return ReportGenerationResult(
            final_answer=template_report,
            llm_enabled=provider not in {"", "template"},
            llm_provider=provider or "template",
            fallback_used=True,
            error_type=error_type,
        )


def report_result_metadata(result: ReportGenerationResult) -> dict[str, Any]:
    return {
        "llm_enabled": result.llm_enabled,
        "llm_provider": result.llm_provider,
        "fallback_used": result.fallback_used,
        "error_type": result.error_type,
    }
