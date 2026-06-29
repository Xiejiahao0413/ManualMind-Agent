import json
from collections.abc import AsyncIterator

from app.agents import DiagnosisWorkflow
from app.schemas.diagnosis import DiagnosisRequest
from app.schemas.trace import TraceEvent
from app.security import StreamingOutputGuard


def format_sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def retrieval_debug(chunks: list[dict[str, object]]) -> dict[str, object]:
    top_chunk = chunks[0] if chunks else {}
    return {
        "retrieved_chunk_count": len(chunks),
        "top_chunk_section_title": top_chunk.get("section_title"),
        "top_chunk_page": top_chunk.get("page"),
        "top_chunk_text_preview": str(top_chunk.get("text") or "")[:300],
    }


async def diagnosis_event_stream(request: DiagnosisRequest) -> AsyncIterator[str]:
    workflow = DiagnosisWorkflow()
    state = DiagnosisWorkflow.from_request(request)
    workflow.ensure_trace(state)
    stream_guard = StreamingOutputGuard()

    async def record_streaming_security_events() -> None:
        for event in stream_guard.pop_events():
            await workflow.memory.append_event(state.task_id, event)
            if state.trace_id and state.request_id:
                workflow.trace_manager.add_event(
                    state.trace_id,
                    TraceEvent(
                        trace_id=state.trace_id,
                        request_id=state.request_id,
                        session_id=state.session_id,
                        task_id=state.task_id,
                        event_type=event["event_type"],
                        component="streaming_output_guard",
                        status="masked",
                        message="Sensitive data was masked from SSE output.",
                        metadata={
                            "sensitive_type": event["sensitive_type"],
                            "source": event["source"],
                        },
                    ),
                )

    async def emit(event: str, data: dict[str, object]) -> AsyncIterator[str]:
        guarded = stream_guard.feed(format_sse(event, data))
        await record_streaming_security_events()
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
        "input_sanitized",
        {
            "session_id": request.session_id,
            "task_id": state.task_id,
            "trace_id": state.trace_id,
            "sanitized_query": state.sanitized_query,
            "sensitive_detected": bool(state.sanitized_fields),
            "sanitized_fields": state.sanitized_fields,
        },
    ):
        yield chunk

    final_state = await workflow.run(state)
    for workflow_event in final_state.workflow_events:
        event_name = str(workflow_event.get("event", "workflow_event"))
        async for chunk in emit(
            event_name,
            {
                "task_id": final_state.task_id,
                "trace_id": final_state.trace_id,
                **workflow_event,
                "sensitive_detected": bool(final_state.sanitized_fields),
                "sanitized_fields": final_state.sanitized_fields,
            },
        ):
            yield chunk

    if final_state.handoff_required:
        async for chunk in emit(
            "handoff_required",
            {
                "task_id": final_state.task_id,
                "trace_id": final_state.trace_id,
                "handoff_reason": final_state.handoff_reason,
                "handoff_payload": final_state.handoff_payload,
                "sensitive_detected": bool(final_state.sanitized_fields),
                "sanitized_fields": final_state.sanitized_fields,
            },
        ):
            yield chunk
        tail = stream_guard.flush()
        await record_streaming_security_events()
        if tail:
            yield tail
        return

    async for chunk in emit(
        "final_answer",
        {
            "task_id": final_state.task_id,
            "trace_id": final_state.trace_id,
            "final_answer": final_state.final_answer,
            "source_refs": final_state.source_refs,
            "retrieval_debug": retrieval_debug(final_state.retrieved_chunks),
            "sensitive_detected": bool(final_state.sanitized_fields),
            "sanitized_fields": final_state.sanitized_fields,
        },
    ):
        yield chunk

    tail = stream_guard.flush()
    await record_streaming_security_events()
    if tail:
        yield tail
