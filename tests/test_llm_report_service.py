import asyncio

from app.llm import MockLLMClient
from app.reporting import ReportGenerationService
from app.schemas.diagnosis import DiagnosisState


def run_service(state: DiagnosisState, service: ReportGenerationService):
    return asyncio.run(service.generate(state))


def test_default_without_env_uses_template_fallback() -> None:
    state = _sample_state()
    result = run_service(state, ReportGenerationService(env={}))

    assert result.llm_enabled is False
    assert result.llm_provider == "template"
    assert result.fallback_used is True
    assert "故障识别" in result.final_answer
    assert "manual.md:1" in result.final_answer


def test_openai_provider_without_api_key_falls_back() -> None:
    state = _sample_state()
    result = run_service(
        state,
        ReportGenerationService(env={"MANUALMIND_LLM_PROVIDER": "openai"}),
    )

    assert result.llm_enabled is False
    assert result.llm_provider == "template"
    assert result.fallback_used is True
    assert result.error_type == "llm_disabled"


def test_mock_llm_success_uses_llm_output() -> None:
    state = _sample_state()
    llm_text = _valid_llm_report("manual.md:1")
    result = run_service(state, ReportGenerationService(llm_client=MockLLMClient(llm_text), env={}))

    assert result.llm_enabled is True
    assert result.llm_provider == "mock"
    assert result.fallback_used is False
    assert "LLM summary" in result.final_answer


def test_mock_llm_empty_output_falls_back() -> None:
    state = _sample_state()
    result = run_service(state, ReportGenerationService(llm_client=MockLLMClient(""), env={}))

    assert result.fallback_used is True
    assert result.error_type == "empty_output"
    assert "LLM summary" not in result.final_answer
    assert "manual.md:1" in result.final_answer


def test_mock_llm_exception_falls_back() -> None:
    state = _sample_state()
    result = run_service(
        state,
        ReportGenerationService(llm_client=MockLLMClient(raise_error=True), env={}),
    )

    assert result.fallback_used is True
    assert result.error_type == "llm_exception"
    assert "manual.md:1" in result.final_answer


def test_llm_output_sensitive_data_is_sanitized() -> None:
    state = _sample_state()
    llm_text = _valid_llm_report("manual.md:1") + "\nContact 13800138000 or ops@example.com."
    result = run_service(state, ReportGenerationService(llm_client=MockLLMClient(llm_text), env={}))

    assert result.fallback_used is False
    assert "13800138000" not in result.final_answer
    assert "ops@example.com" not in result.final_answer
    assert "[PHONE]" in result.final_answer
    assert "[EMAIL]" in result.final_answer
    assert result.sanitized_fields == ["EMAIL", "PHONE"]


def test_source_refs_are_not_lost_from_llm_output() -> None:
    state = _sample_state()
    llm_text = _valid_llm_report("")
    result = run_service(state, ReportGenerationService(llm_client=MockLLMClient(llm_text), env={}))

    assert result.fallback_used is False
    assert "manual.md:1" in result.final_answer


def test_invalid_llm_format_falls_back() -> None:
    state = _sample_state()
    result = run_service(
        state,
        ReportGenerationService(llm_client=MockLLMClient("not a structured report"), env={}),
    )

    assert result.fallback_used is True
    assert result.error_type == "invalid_report_format"
    assert "manual.md:1" in result.final_answer


def test_handoff_required_cannot_be_overridden_by_llm() -> None:
    state = _sample_state()
    state.handoff_required = True
    llm_text = _valid_llm_report("manual.md:1")
    result = run_service(state, ReportGenerationService(llm_client=MockLLMClient(llm_text), env={}))

    assert result.llm_enabled is False
    assert result.fallback_used is True
    assert result.error_type == "handoff_required"
    assert "LLM summary" not in result.final_answer


def _sample_state() -> DiagnosisState:
    return DiagnosisState(
        task_id="task-llm-report",
        session_id="session-llm-report",
        user_query="A100 E03 fault",
        sanitized_query="A100 E03 fault",
        fault_code="E03",
        query_type="fault_code",
        risk_level="medium",
        fault_info={
            "status": "found",
            "fault_code": "E03",
            "description": "E03 temperature sensor abnormal.",
            "source_refs": ["manual.md:1"],
        },
        retrieved_chunks=[
            {
                "text": "E03 temperature sensor abnormal. Disconnect power before inspection.",
                "content_type": "fault_code",
                "fault_code": "E03",
                "source_file": "manual.md",
                "page": 1,
            }
        ],
        safety_rules=["Disconnect power before inspection."],
        source_refs=["manual.md:1"],
    )


def _valid_llm_report(source_ref: str) -> str:
    return "\n\n".join(
        [
            "故障识别：\nLLM summary for E03.",
            "可能原因：\n- sensor wiring issue",
            "排查步骤：\n1. inspect wiring",
            "安全提醒：\n- disconnect power",
            f"引用来源：\n- {source_ref}".rstrip(),
        ]
    )
