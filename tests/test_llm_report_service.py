import asyncio

from app.llm import DeepSeekReportClient, MockLLMClient
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


def test_deepseek_without_api_key_falls_back() -> None:
    state = _sample_state()
    result = run_service(
        state,
        ReportGenerationService(
            env={
                "LLM_ENABLED": "true",
                "LLM_PROVIDER": "deepseek",
            }
        ),
    )

    assert result.llm_enabled is False
    assert result.llm_provider == "template"
    assert result.fallback_used is True
    assert result.error_type == "llm_disabled"


def test_llm_disabled_false_uses_template_fallback() -> None:
    state = _sample_state()
    result = run_service(
        state,
        ReportGenerationService(
            env={
                "LLM_ENABLED": "false",
                "LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "fake-key",
            }
        ),
    )

    assert result.llm_enabled is False
    assert result.llm_provider == "template"
    assert result.fallback_used is True
    assert result.error_type == "llm_disabled"


def test_deepseek_client_is_created_from_env_without_network_call() -> None:
    service = ReportGenerationService(
        env={
            "LLM_ENABLED": "true",
            "LLM_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "fake-key",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            "LLM_MODEL": "deepseek-v4-flash",
            "LLM_TIMEOUT_SECONDS": "7",
            "LLM_MAX_RETRIES": "3",
        }
    )

    client = service._client_from_env()

    assert isinstance(client, DeepSeekReportClient)
    assert client.provider == "deepseek"
    assert client.model == "deepseek-v4-flash"
    assert client.base_url == "https://api.deepseek.com"
    assert client.timeout_seconds == 7
    assert client.max_retries == 3


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


def test_manual_qa_llm_success_does_not_require_diagnosis_sections() -> None:
    state = _manual_qa_state()
    llm_text = "\n\n".join(
        [
            "根据上传手册，操作步骤如下：",
            "1. 点击“增加指令”。",
            "2. 选择“动作指令”。",
            "引用来源：\n- robot.pdf:160",
        ]
    )
    result = run_service(
        state,
        ReportGenerationService(llm_client=_MockDeepSeekClient(llm_text), env={}),
    )

    assert result.llm_enabled is True
    assert result.llm_provider == "deepseek"
    assert result.fallback_used is False
    assert "故障识别" not in result.final_answer
    assert "可能原因" not in result.final_answer
    assert "robot.pdf:160" in result.final_answer


def test_llm_fake_source_refs_are_removed_and_system_refs_are_kept() -> None:
    state = _manual_qa_state()
    llm_text = "\n\n".join(
        [
            "根据上传手册，操作步骤如下：",
            "1. 点击“增加指令”。",
            "引用来源：",
            "- fake.pdf:999",
        ]
    )
    result = run_service(
        state,
        ReportGenerationService(llm_client=_MockDeepSeekClient(llm_text), env={}),
    )

    assert result.fallback_used is False
    assert "fake.pdf:999" not in result.final_answer
    assert "robot.pdf:160" in result.final_answer


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


def _manual_qa_state() -> DiagnosisState:
    return DiagnosisState(
        task_id="task-manual-qa-llm",
        session_id="session-manual-qa-llm",
        user_query="如何增加动作指令",
        sanitized_query="如何增加动作指令",
        query_type="manual_qa",
        risk_level="low",
        retrieved_chunks=[
            {
                "text": "1. 点击“增加指令”。\n2. 选择“动作指令”。",
                "content_type": "general",
                "source_file": "robot.pdf",
                "page": 160,
                "section_title": "4.1.5 创建动作指令",
            }
        ],
        source_refs=["robot.pdf:160"],
    )


class _MockDeepSeekClient(MockLLMClient):
    provider = "deepseek"


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
