import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.agents.reporting import build_diagnosis_report, build_manual_qa_answer
from app.llm import (
    BaseLLMClient,
    DeepSeekReportClient,
    LLMReportContext,
    LLMReportResult,
    OpenAIReportClient,
    QwenReportClient,
)
from app.schemas.diagnosis import DiagnosisState
from app.security import sanitize_text


REQUIRED_REPORT_SECTIONS = ("故障识别", "可能原因", "排查步骤", "安全提醒", "引用来源")


@dataclass(frozen=True)
class ReportGenerationResult:
    final_answer: str
    llm_enabled: bool
    llm_provider: str
    llm_model: str | None = None
    llm_used: bool = False
    fallback_used: bool = True
    fallback_reason: str | None = None
    error_type: str | None = None
    error_message_preview: str | None = None
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
        env_debug = self._env_debug()
        template_report = (
            build_manual_qa_answer(state)
            if state.query_type == "manual_qa"
            else build_diagnosis_report(state)
        )
        if state.handoff_required:
            return ReportGenerationResult(
                final_answer=template_report,
                llm_enabled=False,
                llm_provider="template",
                llm_model=env_debug["llm_model"],
                llm_used=False,
                fallback_used=True,
                fallback_reason="handoff_required",
                error_type="handoff_required",
            )

        client = self.llm_client or self._client_from_env()
        if client is None:
            fallback_reason = self._no_client_reason(env_debug)
            if env_debug["llm_enabled"] and env_debug["llm_provider"] == "qwen":
                return ReportGenerationResult(
                    final_answer=template_report,
                    llm_enabled=True,
                    llm_provider="qwen",
                    llm_model=env_debug["llm_model"] or "qwen-plus",
                    llm_used=False,
                    fallback_used=True,
                    fallback_reason=fallback_reason,
                    error_type=fallback_reason,
                )
            return ReportGenerationResult(
                final_answer=template_report,
                llm_enabled=False,
                llm_provider="template",
                llm_model=env_debug["llm_model"],
                llm_used=False,
                fallback_used=True,
                fallback_reason=fallback_reason,
                error_type="llm_disabled",
            )

        context = self._build_context(state)
        provider = getattr(client, "provider", "unknown")
        model = getattr(client, "model", None) or env_debug["llm_model"]
        try:
            llm_result = await client.generate_report(context)
        except Exception as exc:
            return self._fallback(
                template_report,
                provider,
                "llm_exception",
                model=model,
                error_message=str(exc),
            )

        if not llm_result.success or llm_result.fallback_required:
            return self._fallback(
                template_report,
                llm_result.provider,
                llm_result.error_type or "llm_failed",
                model=llm_result.model or model,
                error_message=llm_result.error_message,
            )

        sanitized = sanitize_text(llm_result.text)
        final_answer = self._ensure_source_refs(
            self._remove_untrusted_source_refs(sanitized.sanitized_text.strip(), state.source_refs),
            state.source_refs,
        )
        if not self._is_valid_report(final_answer, state.source_refs, state.query_type):
            return self._fallback(
                template_report,
                llm_result.provider,
                "invalid_report_format",
                model=llm_result.model or model,
                fallback_reason="invalid_llm_output",
            )

        return ReportGenerationResult(
            final_answer=final_answer,
            llm_enabled=True,
            llm_provider=llm_result.provider,
            llm_model=llm_result.model or model,
            llm_used=True,
            fallback_used=False,
            fallback_reason=None,
            sanitized_fields=sorted({span.replacement.strip("[]") for span in sanitized.spans}),
        )

    def _env_debug(self) -> dict[str, Any]:
        llm_enabled = self.env.get("LLM_ENABLED", "").lower().strip() in {"true", "1", "yes", "on"}
        provider = (self.env.get("LLM_PROVIDER") or self.env.get("MANUALMIND_LLM_PROVIDER") or "").lower().strip()
        if not provider:
            provider = "template"
        return {
            "llm_enabled": llm_enabled or self.llm_client is not None,
            "llm_provider": getattr(self.llm_client, "provider", provider) if self.llm_client else provider,
            "llm_model": getattr(self.llm_client, "model", None)
            or self.env.get("LLM_MODEL")
            or self.env.get("MANUALMIND_LLM_MODEL"),
        }

    def _no_client_reason(self, env_debug: dict[str, Any]) -> str:
        if not env_debug["llm_enabled"]:
            return "disabled"
        provider = str(env_debug["llm_provider"] or "").lower()
        if provider == "deepseek" and not self.env.get("DEEPSEEK_API_KEY", "").strip():
            return "missing_key"
        if provider == "qwen" and not self.env.get("DASHSCOPE_API_KEY", "").strip():
            return "missing_key"
        if provider == "openai" and not self.env.get("OPENAI_API_KEY", "").strip():
            return "missing_key"
        if provider not in {"deepseek", "qwen", "openai", "mock"}:
            return "unsupported_provider"
        return "no_client"

    def _client_from_env(self) -> BaseLLMClient | None:
        llm_enabled = self.env.get("LLM_ENABLED", "").lower().strip()
        if llm_enabled not in {"true", "1", "yes", "on"}:
            return None
        provider = (self.env.get("LLM_PROVIDER") or self.env.get("MANUALMIND_LLM_PROVIDER") or "").lower().strip()
        timeout_seconds = float(self.env.get("LLM_TIMEOUT_SECONDS") or self.env.get("MANUALMIND_LLM_TIMEOUT_SECONDS", "20"))
        max_tokens = int(self.env.get("LLM_MAX_TOKENS") or self.env.get("MANUALMIND_LLM_MAX_TOKENS", "900"))
        temperature = float(self.env.get("LLM_TEMPERATURE") or self.env.get("MANUALMIND_LLM_TEMPERATURE", "0.2"))

        if provider == "deepseek":
            api_key = self.env.get("DEEPSEEK_API_KEY", "").strip()
            if not api_key:
                return None
            return DeepSeekReportClient(
                api_key=api_key,
                model=self.env.get("LLM_MODEL") or self.env.get("MANUALMIND_LLM_MODEL", "deepseek-v4-flash"),
                base_url=self.env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                timeout_seconds=timeout_seconds,
                max_tokens=max_tokens,
                temperature=temperature,
                max_retries=int(self.env.get("LLM_MAX_RETRIES", "2")),
            )

        if provider == "qwen":
            api_key = self.env.get("DASHSCOPE_API_KEY", "").strip()
            if not api_key:
                return None
            return QwenReportClient(
                api_key=api_key,
                model=self.env.get("LLM_MODEL") or self.env.get("MANUALMIND_LLM_MODEL", "qwen-plus"),
                base_url=self.env.get(
                    "DASHSCOPE_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                timeout_seconds=timeout_seconds,
                max_tokens=max_tokens,
                temperature=temperature,
                max_retries=int(self.env.get("LLM_MAX_RETRIES", "2")),
            )

        if provider != "openai":
            return None
        api_key = self.env.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        return OpenAIReportClient(
            api_key=api_key,
            model=self.env.get("LLM_MODEL") or self.env.get("MANUALMIND_LLM_MODEL", "gpt-4o-mini"),
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _build_context(self, state: DiagnosisState) -> LLMReportContext:
        query = state.sanitized_query or state.user_query
        evidence_snippets = []
        evidence_items: list[dict[str, str]] = []
        for chunk in state.retrieved_chunks[:5]:
            text = sanitize_text(str(chunk.get("text") or "")).sanitized_text
            if text:
                evidence_snippets.append(text[:500])
                evidence_items.append(
                    {
                        "text": text[:800],
                        "source_ref": self._chunk_source_ref(chunk),
                        "section_title": sanitize_text(str(chunk.get("section_title") or "")).sanitized_text,
                        "page": str(chunk.get("page") or ""),
                    }
                )
        safety_warnings = [sanitize_text(str(item)).sanitized_text for item in state.safety_rules if item]
        return LLMReportContext(
            query=sanitize_text(query).sanitized_text,
            query_type=state.query_type,
            device_model=state.device_model,
            fault_code=state.fault_code,
            risk_level=state.risk_level,
            evidence_snippets=evidence_snippets,
            evidence_items=evidence_items,
            source_refs=sorted(set(state.source_refs)),
            safety_warnings=safety_warnings,
            handoff_required=state.handoff_required,
        )

    def _is_valid_report(self, report: str, source_refs: list[str], query_type: str | None = None) -> bool:
        if not report.strip():
            return False
        if query_type == "manual_qa":
            if len(report.strip()) < 20:
                return False
            if source_refs and not all(source in report for source in source_refs):
                return False
            return "引用来源" in report or "寮曠敤鏉ユ簮" in report
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

    def _remove_untrusted_source_refs(self, report: str, source_refs: list[str]) -> str:
        trusted = {source for source in source_refs if source}
        if not trusted:
            return report
        cleaned_lines: list[str] = []
        for line in report.splitlines():
            stripped = line.strip()
            if stripped.startswith(("- ", "* ")):
                candidate = stripped[2:].strip()
                looks_like_source_ref = any(token in candidate for token in (".pdf", ".md", ".txt", ":"))
                if looks_like_source_ref and candidate not in trusted:
                    continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _chunk_source_ref(self, chunk: dict[str, Any]) -> str:
        source_file = chunk.get("source_file")
        page = chunk.get("page")
        section_title = chunk.get("section_title")
        if source_file and page is not None:
            return f"{source_file}:{page}"
        if source_file and section_title:
            return f"{source_file} / {section_title}"
        return str(source_file or "")

    def _fallback(
        self,
        template_report: str,
        provider: str,
        error_type: str,
        *,
        model: str | None = None,
        fallback_reason: str | None = None,
        error_message: str | None = None,
    ) -> ReportGenerationResult:
        return ReportGenerationResult(
            final_answer=template_report,
            llm_enabled=provider not in {"", "template"},
            llm_provider=provider or "template",
            llm_model=model,
            llm_used=False,
            fallback_used=True,
            fallback_reason=fallback_reason or error_type,
            error_type=error_type,
            error_message_preview=error_message[:300] if error_message else None,
        )


def report_result_metadata(result: ReportGenerationResult) -> dict[str, Any]:
    return {
        "llm_enabled": result.llm_enabled,
        "llm_provider": result.llm_provider,
        "llm_model": result.llm_model,
        "llm_used": result.llm_used,
        "fallback_used": result.fallback_used,
        "fallback_reason": result.fallback_reason,
        "llm_error_type": result.error_type,
        "llm_error_message_preview": result.error_message_preview,
    }
