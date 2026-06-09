from uuid import uuid4

from app.schemas.trace import RequestTrace, TraceEvent


class InMemoryTraceManager:
    def __init__(self) -> None:
        self._traces: dict[str, RequestTrace] = {}

    def start_trace(self, request_id: str, session_id: str, task_id: str) -> RequestTrace:
        trace = RequestTrace(
            trace_id=f"trace-{uuid4().hex}",
            request_id=request_id,
            session_id=session_id,
            task_id=task_id,
        )
        self._traces[trace.trace_id] = trace
        return trace

    def add_event(self, trace_id: str, event: TraceEvent) -> None:
        trace = self._traces.get(trace_id)
        if trace is None:
            trace = RequestTrace(
                trace_id=trace_id,
                request_id=event.request_id,
                session_id=event.session_id,
                task_id=event.task_id,
            )
            self._traces[trace_id] = trace
        trace.events.append(event)

    def get_trace(self, trace_id: str) -> RequestTrace | None:
        return self._traces.get(trace_id)

    def list_events(self, trace_id: str) -> list[TraceEvent]:
        trace = self._traces.get(trace_id)
        return list(trace.events) if trace else []

    def summarize_trace(self, trace_id: str) -> dict[str, object]:
        trace = self._traces.get(trace_id)
        if trace is None:
            return {"trace_id": trace_id, "status": "not_found", "event_count": 0}

        event_counts: dict[str, int] = {}
        for event in trace.events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
        return {
            "trace_id": trace.trace_id,
            "request_id": trace.request_id,
            "session_id": trace.session_id,
            "task_id": trace.task_id,
            "status": "found",
            "event_count": len(trace.events),
            "event_counts": event_counts,
        }
