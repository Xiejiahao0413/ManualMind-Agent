from fastapi import APIRouter, HTTPException

from app.core.dependencies import get_trace_manager
from app.schemas.trace import RequestTrace, TraceEvent

router = APIRouter(tags=["trace"])


@router.get("/trace/{trace_id}", response_model=RequestTrace)
async def get_trace(trace_id: str) -> RequestTrace:
    trace = get_trace_manager().get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace_not_found")
    return trace


@router.get("/trace/{trace_id}/events", response_model=list[TraceEvent])
async def get_trace_events(trace_id: str) -> list[TraceEvent]:
    trace = get_trace_manager().get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace_not_found")
    return get_trace_manager().list_events(trace_id)
