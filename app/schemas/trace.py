from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    trace_id: str
    request_id: str
    session_id: str
    task_id: str
    event_type: str
    component: str
    status: str = "ok"
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentNodeTrace(BaseModel):
    node_name: str
    status: str
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallTrace(BaseModel):
    tool_name: str
    args_signature: str
    status: str
    retry_count: int = 0
    fallback_used: bool = False
    error_type: str | None = None
    latency_ms: float | None = None
    result_summary: str | None = None


class RetrievalTrace(BaseModel):
    query: str
    top_k_bm25: int | None = None
    top_k_dense: int | None = None
    top_n_rerank: int | None = None
    candidate_count: int = 0
    final_count: int = 0
    metadata_filter: dict[str, Any] = Field(default_factory=dict)
    max_score: float | None = None
    max_rerank_score: float | None = None
    retrieval_mode: str | None = None


class FallbackTrace(BaseModel):
    error_type: str
    fallback_decision: str
    retry_count: int = 0


class HandoffTrace(BaseModel):
    handoff_reason: str
    risk_level: str
    tool_trace_count: int = 0
    source_count: int = 0


class RequestTrace(BaseModel):
    trace_id: str
    request_id: str
    session_id: str
    task_id: str
    events: list[TraceEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
