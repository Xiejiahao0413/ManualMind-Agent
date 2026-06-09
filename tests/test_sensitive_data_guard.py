import asyncio
import json
import re

from fastapi.testclient import TestClient

from app.core.dependencies import get_memory_manager
from app.main import app
from app.security import StreamingOutputGuard, sanitize_text
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
