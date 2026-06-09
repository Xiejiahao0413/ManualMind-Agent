import json
from collections.abc import AsyncIterator

from app.agents import DiagnosisWorkflow
from app.schemas.diagnosis import DiagnosisRequest


def format_sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def diagnosis_event_stream(request: DiagnosisRequest) -> AsyncIterator[str]:
    workflow = DiagnosisWorkflow()
    state = DiagnosisWorkflow.from_request(request)

    yield format_sse(
        "diagnosis_started",
        {
            "session_id": request.session_id,
            "task_id": state.task_id,
            "message": "Diagnosis request accepted.",
        },
    )
    yield format_sse(
        "retrieval_started",
        {
            "task_id": state.task_id,
            "message": "Tool-routed mock retrieval stage started.",
        },
    )
    yield format_sse(
        "safety_review_started",
        {
            "task_id": state.task_id,
            "message": "Safety review and report stage started.",
        },
    )

    final_state = await workflow.run(state)
    if final_state.handoff_required:
        yield format_sse(
            "handoff_required",
            {
                "task_id": final_state.task_id,
                "handoff_reason": final_state.handoff_reason,
                "handoff_payload": final_state.handoff_payload,
            },
        )
        return

    yield format_sse(
        "final_answer",
        {
            "task_id": final_state.task_id,
            "final_answer": final_state.final_answer,
            "source_refs": final_state.source_refs,
        },
    )
