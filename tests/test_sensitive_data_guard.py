import asyncio
import json
import re

from fastapi.testclient import TestClient

import app.core.sse as sse_core
from app.core.dependencies import get_memory_manager
from app.core.dependencies import get_trace_manager
from app.main import app
from app.security import StreamingOutputGuard, sanitize_text
from app.schemas.diagnosis import DiagnosisRequest, DiagnosisState
from app.tools import ToolCallGuard


def test_phone_number_is_sanitized() -> None:
    result = sanitize_text("call 13800138000 for support")

    assert result.sanitized_text == "call [PHONE] for support"
    assert result.spans[0].type == "phone_number"


def test_email_is_sanitized() -> None:
    result = sanitize_text("email ops@example.com now")

    assert result.sanitized_text == "email [EMAIL] now"
    assert result.spans[0].type == "email"


def test_ip_address_is_sanitized() -> None:
    result = sanitize_text("connect to 192.168.1.8")

    assert result.sanitized_text == "connect to [IP_ADDRESS]"
    assert result.spans[0].type == "ip_address"


def test_device_id_is_sanitized() -> None:
    result = sanitize_text("device DEV-ABC123 failed")

    assert result.sanitized_text == "device [DEVICE_ID] failed"
    assert result.spans[0].type == "device_id"


def test_work_order_id_is_sanitized() -> None:
    result = sanitize_text("check WO-123456 today")

    assert result.sanitized_text == "check [WORK_ORDER_ID] today"
    assert result.spans[0].type == "work_order_id"


def test_streaming_output_guard_handles_split_sensitive_text() -> None:
    guard = StreamingOutputGuard(buffer_size=16)

    output = (
        guard.feed("call 138")
        + guard.feed("0013")
        + guard.feed("8000 now")
        + guard.flush()
    )

    assert output == "call [PHONE] now"


def test_streaming_output_guard_masks_phone_split_across_chunks() -> None:
    guard = StreamingOutputGuard(buffer_size=16)

    output = guard.feed("phone 138") + guard.feed("1234") + guard.feed("5678 ok") + guard.flush()

    assert output == "phone [PHONE] ok"
    events = guard.pop_events()
    assert events == [
        {
            "event_type": "streaming_sensitive_data_masked",
            "sensitive_type": "phone_number",
            "source": "sse_output",
        }
    ]
    assert "13812345678" not in json.dumps(events)


def test_streaming_output_guard_masks_email_split_across_chunks() -> None:
    guard = StreamingOutputGuard(buffer_size=18)

    output = (
        guard.feed("email xiejiahao")
        + guard.feed("@example")
        + guard.feed(".com done")
        + guard.flush()
    )

    assert output == "email [EMAIL] done"


def test_streaming_output_guard_masks_ip_split_across_chunks() -> None:
    guard = StreamingOutputGuard(buffer_size=12)

    output = guard.feed("IP 192.168") + guard.feed(".1.10 ready") + guard.flush()

    assert output == "IP [IP_ADDRESS] ready"


def test_streaming_output_guard_masks_work_order_split_across_chunks() -> None:
    guard = StreamingOutputGuard(buffer_size=10)

    output = guard.feed("ticket WO-") + guard.feed("2026-001 ready") + guard.flush()

    assert output == "ticket [WORK_ORDER_ID] ready"


def test_streaming_output_guard_masks_device_id_split_across_chunks() -> None:
    guard = StreamingOutputGuard(buffer_size=12)

    output = guard.feed("device SN-") + guard.feed("A100-8899 alarm") + guard.flush()

    assert output == "device [DEVICE_ID] alarm"


def test_streaming_output_guard_preserves_normal_text_and_codes() -> None:
    guard = StreamingOutputGuard(buffer_size=8)
    text = "\u666e\u901a\u53e5\u5b50\n- A100 E03 source_refs: a100_manual.md:3\nB200 ok"

    output = guard.feed(text[:12]) + guard.feed(text[12:28]) + guard.feed(text[28:]) + guard.flush()

    assert output == text
    assert "E03" in output
    assert "A100" in output
    assert "B200" in output
    assert "a100_manual.md:3" in output
    assert guard.pop_events() == []


def test_streaming_output_guard_flush_emits_remaining_safe_text() -> None:
    guard = StreamingOutputGuard(buffer_size=64)

    output = guard.feed("safe tail text") + guard.flush()

    assert output == "safe tail text"


def test_diagnosis_chat_uses_sanitized_query() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/diagnosis/chat",
        json={
            "session_id": "security-session",
            "message": "device DEV-ABC123 phone 13800138000 reports E03",
        },
    )

    assert response.status_code == 200
    body = response.text
    assert "13800138000" not in body
    assert "DEV-ABC123" not in body
    assert "[PHONE]" in body
    assert "PHONE" in body
    assert "DEVICE_ID" in body

    task_id_match = re.search(r'"task_id":\s*"([^"]+)"', body)
    assert task_id_match is not None
    task_id = task_id_match.group(1)
    state = asyncio.run(get_memory_manager().get_task(task_id))

    assert state is not None
    assert state.raw_query is not None
    assert "13800138000" in state.raw_query
    assert "[PHONE]" in state.sanitized_query
    assert "PHONE" in state.sanitized_fields
    assert "DEVICE_ID" in state.sanitized_fields

    events = asyncio.run(get_memory_manager().list_events(task_id))
    sensitive_events = [
        event for event in events if event.get("event_type") == "sensitive_data_detected"
    ]
    assert sensitive_events
    assert sensitive_events[0]["action_taken"] == "masked"


def test_diagnosis_chat_streaming_output_masks_sensitive_values_and_records_event(monkeypatch) -> None:
    task_id = "task-streaming-output-mask"

    class FakeWorkflow:
        def __init__(self) -> None:
            self.memory = get_memory_manager()
            self.trace_manager = get_trace_manager()

        @classmethod
        def from_request(cls, request: DiagnosisRequest) -> DiagnosisState:
            return DiagnosisState(
                task_id=task_id,
                request_id="req-streaming-output-mask",
                session_id=request.session_id,
                user_query=request.message,
                sanitized_query=request.message,
            )

        def ensure_trace(self, state: DiagnosisState) -> DiagnosisState:
            if state.trace_id is None:
                trace = self.trace_manager.start_trace(
                    request_id=state.request_id or "req-streaming-output-mask",
                    session_id=state.session_id,
                    task_id=state.task_id,
                )
                state.trace_id = trace.trace_id
            return state

        async def run(self, state: DiagnosisState) -> DiagnosisState:
            state.workflow_events.append({"event": "diagnosis_completed"})
            state.final_answer = (
                "contact 13812345678, email xiejiahao@example.com, IP 192.168.1.10, "
                "ticket WO-2026-001, device SN-A100-8899"
            )
            await self.memory.save_task(state)
            return state

    monkeypatch.setattr(sse_core, "DiagnosisWorkflow", FakeWorkflow)

    response = TestClient(app).post(
        "/api/diagnosis/chat",
        json={"session_id": "session-streaming-output-mask", "message": "safe question"},
    )

    assert response.status_code == 200
    body = response.text
    assert "13812345678" not in body
    assert "xiejiahao@example.com" not in body
    assert "192.168.1.10" not in body
    assert "WO-2026-001" not in body
    assert "SN-A100-8899" not in body
    assert "[PHONE]" in body
    assert "[EMAIL]" in body
    assert "[IP_ADDRESS]" in body
    assert "[WORK_ORDER_ID]" in body
    assert "[DEVICE_ID]" in body

    events = asyncio.run(get_memory_manager().list_events(task_id))
    streaming_events = [
        event for event in events if event.get("event_type") == "streaming_sensitive_data_masked"
    ]
    assert streaming_events
    serialized_events = json.dumps(streaming_events)
    assert "13812345678" not in serialized_events
    assert "xiejiahao@example.com" not in serialized_events
    assert "192.168.1.10" not in serialized_events
    assert "WO-2026-001" not in serialized_events
    assert "SN-A100-8899" not in serialized_events


def test_tool_call_guard_blocks_unsanitized_sensitive_args() -> None:
    guard = ToolCallGuard(tool_whitelist={"manual_hybrid_search"})

    decision = guard.check(
        tool_name="manual_hybrid_search",
        args={"query": "call 13800138000"},
        required_args={"query"},
    )

    assert decision.allowed is False
    assert decision.reason == "sanitized_required"


def test_tool_call_guard_allows_sanitized_sensitive_args() -> None:
    guard = ToolCallGuard(tool_whitelist={"manual_hybrid_search"})

    decision = guard.check(
        tool_name="manual_hybrid_search",
        args={"query": "call [PHONE]"},
        required_args={"query"},
    )

    assert decision.allowed is True


def test_sse_payload_is_valid_json_after_sanitizing() -> None:
    payload = sanitize_text(json.dumps({"message": "ops@example.com"})).sanitized_text

    assert "[EMAIL]" in payload
