import asyncio

from app.memory import InMemoryMemoryManager
from app.schemas.diagnosis import DiagnosisState
from app.schemas.tools import ToolCallRecord
from app.tools import (
    CircuitBreaker,
    FallbackAction,
    RetryFallbackManager,
    ToolCallGuard,
    ToolErrorType,
    ToolIntent,
    ToolResultValidator,
    ToolRouter,
)


def test_successful_args_signature_can_reuse_cached_record() -> None:
    guard = ToolCallGuard(tool_whitelist={"manual_hybrid_search"})
    router = ToolRouter(guard)
    intent = ToolIntent(
        tool_name="manual_hybrid_search",
        args={"query": "E01", "top_k": 5},
        required_args={"query"},
    )

    first = router.validate_intent(intent)
    assert first.allowed is True
    assert first.status == "allowed"
    assert first.args_signature is not None

    record = ToolCallRecord(
        task_id="task-1",
        tool_name="manual_hybrid_search",
        args=intent.args,
        args_signature=first.args_signature,
        status="success",
        result_summary="found evidence",
    )
    guard.remember_success(record)

    second = router.validate_intent(intent)
    assert second.allowed is True
    assert second.status == "reuse_cached"
    assert second.cached_record == record


def test_retry_count_triggers_circuit_breaker() -> None:
    guard = ToolCallGuard(tool_whitelist={"manual_hybrid_search"}, max_retry=3)
    router = ToolRouter(guard)
    intent = ToolIntent(
        tool_name="manual_hybrid_search",
        args={"query": "E01"},
        required_args={"query"},
    )

    decision = router.validate_intent(intent, retry_count=3)
    assert decision.allowed is False
    assert decision.status == "circuit_breaker_required"

    breaker = CircuitBreaker(max_retry=3)
    breaker_decision = breaker.evaluate(
        retry_count=3,
        called_signatures=set(),
        tool_call_count=1,
    )
    assert breaker_decision.triggered is True
    assert breaker_decision.reason == "max_retry_reached"


def test_retry_fallback_rules() -> None:
    manager = RetryFallbackManager(max_retry=3)

    assert manager.decide(ToolErrorType.MILVUS_TIMEOUT) == FallbackAction.BM25_ONLY
    assert manager.decide(ToolErrorType.RERANK_ERROR) == FallbackAction.FUSED_SCORE
    assert (
        manager.decide(ToolErrorType.HIGH_RISK_WITHOUT_EVIDENCE)
        == FallbackAction.HUMAN_HANDOFF
    )
    assert manager.decide(ToolErrorType.LLM_ERROR, retry_count=3) == (
        FallbackAction.CIRCUIT_BREAKER_HANDOFF
    )


def test_high_risk_without_safety_evidence_is_invalid() -> None:
    validator = ToolResultValidator(min_score=0.2)

    result = {
        "text": "Lockout required before opening panel.",
        "score": 0.9,
        "source_refs": ["manual:page-9"],
    }
    validation = validator.validate(result, risk_level="high")

    assert validation.valid is False
    assert validation.error_type == ToolErrorType.HIGH_RISK_WITHOUT_EVIDENCE


def test_tool_memory_saves_and_queries_by_signature() -> None:
    asyncio.run(_assert_tool_memory_saves_and_queries_by_signature())


async def _assert_tool_memory_saves_and_queries_by_signature() -> None:
    memory = InMemoryMemoryManager()
    record = ToolCallRecord(
        task_id="task-1",
        tool_name="fault_code_lookup",
        args={"fault_code": "E01"},
        args_signature="signature-1",
        status="success",
        retry_count=1,
        latency_ms=12.5,
        result_summary="fault code found",
    )

    await memory.append_tool_call(record)

    records = await memory.list_tool_calls("task-1")
    found = await memory.get_tool_call_by_signature("task-1", "signature-1")

    assert records == [record]
    assert found == record


def test_circuit_breaker_creates_handoff_payload() -> None:
    breaker = CircuitBreaker()
    state = DiagnosisState(
        task_id="task-1",
        session_id="session-1",
        user_query="Motor overheats during restart.",
        device_name="Compressor",
        fault_code="E99",
        symptoms=["overheat", "restart failure"],
        risk_level="high",
    )
    tool_record = ToolCallRecord(
        task_id="task-1",
        tool_name="manual_hybrid_search",
        args={"query": "E99"},
        args_signature="signature-1",
        status="failed",
        error_type=ToolErrorType.HIGH_RISK_WITHOUT_EVIDENCE,
    )

    payload = breaker.create_handoff_payload(
        state=state,
        handoff_reason="high_risk_without_evidence",
        tool_trace=[tool_record],
    )

    assert payload.task_id == "task-1"
    assert payload.handoff_reason == "high_risk_without_evidence"
    assert payload.device_name == "Compressor"
    assert payload.fault_code == "E99"
    assert payload.symptoms == ["overheat", "restart failure"]
    assert payload.tool_trace[0]["tool_name"] == "manual_hybrid_search"
