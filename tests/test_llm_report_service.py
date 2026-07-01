import asyncio

from app.llm import DeepSeekReportClient, MockLLMClient, QwenReportClient
from app.reporting import ReportGenerationService, report_result_metadata
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


def test_qwen_without_dashscope_api_key_falls_back() -> None:
    state = _sample_state()
    result = run_service(
        state,
        ReportGenerationService(
            env={
                "LLM_ENABLED": "true",
                "LLM_PROVIDER": "qwen",
                "LLM_MODEL": "qwen-plus",
            }
        ),
    )
    metadata = report_result_metadata(result)

    assert result.llm_enabled is True
    assert result.llm_provider == "qwen"
    assert result.llm_model == "qwen-plus"
    assert result.llm_used is False
    assert result.fallback_used is True
    assert result.fallback_reason == "missing_key"
    assert result.error_type == "missing_key"
    assert "manual.md:1" in result.final_answer
    assert metadata["llm_provider"] == "qwen"
    assert metadata["llm_model"] == "qwen-plus"
    assert metadata["llm_error_type"] == "missing_key"


def test_qwen_client_is_created_from_env_without_network_call() -> None:
    service = ReportGenerationService(
        env={
            "LLM_ENABLED": "true",
            "LLM_PROVIDER": "qwen",
            "DASHSCOPE_API_KEY": "fake-dashscope-key",
            "DASHSCOPE_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "LLM_MODEL": "qwen-turbo",
            "LLM_TIMEOUT_SECONDS": "9",
            "LLM_MAX_RETRIES": "4",
        }
    )

    client = service._client_from_env()

    assert isinstance(client, QwenReportClient)
    assert client.provider == "qwen"
    assert client.model == "qwen-turbo"
    assert client.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert client.timeout_seconds == 9
    assert client.max_retries == 4


def test_mock_qwen_success_uses_llm_output_and_debug_metadata() -> None:
    state = _sample_state()
    result = run_service(
        state,
        ReportGenerationService(llm_client=_MockQwenClient(_valid_llm_report("manual.md:1")), env={}),
    )
    metadata = report_result_metadata(result)

    assert result.llm_enabled is True
    assert result.llm_provider == "qwen"
    assert result.llm_model == "qwen-plus"
    assert result.llm_used is True
    assert result.fallback_used is False
    assert "LLM summary" in result.final_answer
    assert metadata["llm_provider"] == "qwen"
    assert metadata["llm_model"] == "qwen-plus"
    assert metadata["llm_used"] is True
    assert metadata["fallback_reason"] is None


def test_mock_qwen_exception_falls_back_with_qwen_debug_metadata() -> None:
    state = _sample_state()
    result = run_service(
        state,
        ReportGenerationService(llm_client=_MockQwenClient("", raise_error=True, error_type="dashscope boom"), env={}),
    )
    metadata = report_result_metadata(result)

    assert result.llm_enabled is True
    assert result.llm_provider == "qwen"
    assert result.llm_used is False
    assert result.fallback_used is True
    assert result.fallback_reason == "llm_exception"
    assert result.error_type == "llm_exception"
    assert result.error_message_preview == "dashscope boom"
    assert "manual.md:1" in result.final_answer
    assert metadata["llm_provider"] == "qwen"
    assert metadata["llm_error_message_preview"] == "dashscope boom"


def test_qwen_fake_source_refs_are_removed_and_system_refs_are_kept() -> None:
    state = _manual_qa_state()
    llm_text = "\n\n".join(
        [
            "鏍规嵁涓婁紶鎵嬪唽锛屾搷浣滄楠ゅ涓嬶細",
            "1. 鐐瑰嚮鈥滃鍔犳寚浠も€濄€?",
            "寮曠敤鏉ユ簮锛?",
            "- fake-qwen.pdf:999",
        ]
    )
    result = run_service(
        state,
        ReportGenerationService(llm_client=_MockQwenClient(llm_text), env={}),
    )

    assert result.llm_provider == "qwen"
    assert result.fallback_used is False
    assert "fake-qwen.pdf:999" not in result.final_answer
    assert "robot.pdf:160" in result.final_answer


def test_mock_llm_success_uses_llm_output() -> None:
    state = _sample_state()
    llm_text = _valid_llm_report("manual.md:1")
    result = run_service(state, ReportGenerationService(llm_client=MockLLMClient(llm_text), env={}))

    assert result.llm_enabled is True
    assert result.llm_provider == "mock"
    assert result.fallback_used is False
    assert result.llm_used is True
    assert "LLM summary" in result.final_answer


def test_llm_debug_metadata_reports_success() -> None:
    state = _sample_state()
    result = run_service(
        state,
        ReportGenerationService(llm_client=MockLLMClient(_valid_llm_report("manual.md:1")), env={}),
    )
    metadata = report_result_metadata(result)

    assert metadata["llm_enabled"] is True
    assert metadata["llm_provider"] == "mock"
    assert metadata["llm_used"] is True
    assert metadata["fallback_used"] is False
    assert metadata["fallback_reason"] is None


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
    assert result.fallback_reason == "llm_exception"
    assert result.error_message_preview == "mock_error"
    assert "manual.md:1" in result.final_answer


def test_llm_debug_metadata_reports_fallback_error_preview() -> None:
    state = _sample_state()
    result = run_service(
        state,
        ReportGenerationService(llm_client=MockLLMClient(raise_error=True), env={}),
    )
    metadata = report_result_metadata(result)

    assert metadata["llm_used"] is False
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"] == "llm_exception"
    assert metadata["llm_error_type"] == "llm_exception"
    assert metadata["llm_error_message_preview"] == "mock_error"
    assert len(metadata["llm_error_message_preview"]) <= 300


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


class _MockQwenClient(MockLLMClient):
    provider = "qwen"

    def __init__(
        self,
        response_text: str = "",
        *,
        model: str = "qwen-plus",
        raise_error: bool = False,
        error_type: str = "qwen_mock_error",
    ) -> None:
        super().__init__(response_text, raise_error=raise_error, error_type=error_type)
        self.model = model


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
