import json
from collections.abc import AsyncIterator

from app.agents import DiagnosisWorkflow
from app.schemas.diagnosis import DiagnosisRequest
from app.security import StreamingOutputGuard


def format_sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def diagnosis_event_stream(request: DiagnosisRequest) -> AsyncIterator[str]:
    workflow = DiagnosisWorkflow()
    state = DiagnosisWorkflow.from_request(request)
    stream_guard = StreamingOutputGuard()

    async def emit(event: str, data: dict[str, object]) -> AsyncIterator[str]:
        guarded = stream_guard.feed(format_sse(event, data))
        if guarded:
            yield guarded

    if state.security_events:
        await workflow.memory.append_event(
            state.task_id,
            {
                "task_id": state.task_id,
                "event_type": "sensitive_data_detected",
                "sanitized_fields": state.sanitized_fields,
                "action_taken": "masked",
            },
        )

    async for chunk in emit(
        "diagnosis_started",
        {
            "session_id": request.session_id,
            "task_id": state.task_id,
            "message": "Diagnosis request accepted.",
            "sanitized_query": state.sanitized_query,
            "sensitive_detected": bool(state.sanitized_fields),
            "sanitized_fields": state.sanitized_fields,
        },
    ):
        yield chunk

    async for chunk in emit(
        "retrieval_started",
        {
            "task_id": state.task_id,
            "message": "Tool-routed mock retrieval stage started.",
            "sensitive_detected": bool(state.sanitized_fields),
            "sanitized_fields": state.sanitized_fields,
        },
    ):
        yield chunk

    async for chunk in emit(
        "safety_review_started",
        {
            "task_id": state.task_id,
            "message": "Safety review and report stage started.",
            "sensitive_detected": bool(state.sanitized_fields),
            "sanitized_fields": state.sanitized_fields,
        },
    ):
        yield chunk

    final_state = await workflow.run(state)
    if final_state.handoff_required:
        async for chunk in emit(
            "handoff_required",
            {
                "task_id": final_state.task_id,
                "handoff_reason": final_state.handoff_reason,
                "handoff_payload": final_state.handoff_payload,
                "sensitive_detected": bool(final_state.sanitized_fields),
                "sanitized_fields": final_state.sanitized_fields,
            },
        ):
            yield chunk
        tail = stream_guard.flush()
        if tail:
            yield tail
        return

    async for chunk in emit(
        "final_answer",
        {
            "task_id": final_state.task_id,
            "final_answer": final_state.final_answer,
            "source_refs": final_state.source_refs,
            "sensitive_detected": bool(final_state.sanitized_fields),
            "sanitized_fields": final_state.sanitized_fields,
        },
    ):
        yield chunk

    tail = stream_guard.flush()
    if tail:
        yield tail
