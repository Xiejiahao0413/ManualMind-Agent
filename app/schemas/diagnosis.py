from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.retrieval import RetrievalResult
from app.schemas.tools import ToolCallRecord


class DiagnosisRequest(BaseModel):
    session_id: str = Field(..., description="Conversation session id.")
    message: str = Field(..., description="User diagnosis question.")
    task_id: str | None = Field(default=None, description="Existing diagnosis task id.")
    stream: bool = Field(default=True, description="Whether to stream response events.")


class DiagnosisState(BaseModel):
    task_id: str
    session_id: str
    user_query: str
    raw_query: str | None = None
    sanitized_query: str | None = None
    sanitized_fields: list[str] = Field(default_factory=list)
    security_events: list[dict[str, Any]] = Field(default_factory=list)
    device_name: str | None = None
    device_model: str | None = None
    fault_code: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    query_type: str | None = None
    risk_level: Literal["low", "medium", "high", "unknown"] = "unknown"
    retrieval_status: str = "not_started"
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    retrieved_evidence: list[RetrievalResult] = Field(default_factory=list)
    tool_call_history: list[ToolCallRecord] = Field(default_factory=list)
    retry_count: int = 0
    max_retry: int = 3
    handoff_required: bool = False
    handoff_reason: str | None = None
    handoff_payload: dict[str, Any] | None = None
    final_answer: str | None = None


class DiagnosisResponse(BaseModel):
    task_id: str
    status: str
    message: str
    state: DiagnosisState | None = None
