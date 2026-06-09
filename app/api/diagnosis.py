from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.sse import diagnosis_event_stream
from app.schemas.diagnosis import DiagnosisRequest, DiagnosisResponse, DiagnosisState

router = APIRouter(tags=["diagnosis"])


@router.post("/diagnosis/chat")
async def diagnosis_chat(request: DiagnosisRequest) -> StreamingResponse:
    return StreamingResponse(
        diagnosis_event_stream(request),
        media_type="text/event-stream",
    )


@router.get("/diagnosis/tasks/{task_id}", response_model=DiagnosisResponse)
async def get_diagnosis_task(task_id: str) -> DiagnosisResponse:
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")

    state = DiagnosisState(task_id=task_id, session_id="", user_query="")
    return DiagnosisResponse(
        task_id=task_id,
        status="not_found",
        message="Task memory backend is not configured yet.",
        state=state,
    )
