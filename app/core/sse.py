import json
from collections.abc import AsyncIterator

from app.schemas.diagnosis import DiagnosisRequest


def format_sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def diagnosis_event_stream(request: DiagnosisRequest) -> AsyncIterator[str]:
    yield format_sse(
        "diagnosis.started",
        {
            "session_id": request.session_id,
            "message": "Diagnosis request accepted.",
        },
    )
    yield format_sse(
        "diagnosis.placeholder",
        {
            "message": "Agent workflow is not implemented yet.",
        },
    )
    yield format_sse(
        "diagnosis.completed",
        {
            "status": "pending_implementation",
        },
    )
