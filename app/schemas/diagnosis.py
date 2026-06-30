from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.retrieval import RetrievalResult
from app.schemas.tools import ToolCallRecord


class DiagnosisRequest(BaseModel):
    session_id: str = Field(..., description="Conversation session id.")
    message: str = Field(..., description="User diagnosis question.")
    task_id: str | None = Field(default=None, description="Existing diagnosis task id.")
    stream: bool = Field(default=True, description="Whether to stream response events.")
    doc_id: str | None = Field(default=None, description="Current uploaded manual document id.")
    doc_ids: list[str] = Field(default_factory=list, description="Uploaded manual document scope.")


class DiagnosisState(BaseModel):
    task_id: str
    request_id: str | None = None
    trace_id: str | None = None
    session_id: str
    user_query: str
    raw_query: str | None = None
    sanitized_query: str | None = None
    sanitized_fields: list[str] = Field(default_factory=list)
    security_events: list[dict[str, Any]] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)
    knowledge_scope: Literal["demo", "uploaded_docs"] = "demo"
    device_name: str | None = None
    device_model: str | None = None
    fault_code: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    query_type: str | None = None
    risk_level: Literal["low", "medium", "high", "unknown"] = "unknown"
    retrieval_status: str = "not_started"
    retrieval_mode: str | None = None
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    retrieved_evidence: list[RetrievalResult] = Field(default_factory=list)
    fault_info: dict[str, Any] | None = None
    parameter_info: dict[str, Any] | None = None
    safety_rules: list[str] = Field(default_factory=list)
    fallback_decision: str | None = None
    workflow_events: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_history: list[ToolCallRecord] = Field(default_factory=list)
    retry_count: int = 0
    max_retry: int = 3
    handoff_required: bool = False
    handoff_reason: str | None = None
    handoff_payload: dict[str, Any] | None = None
    final_answer: str | None = None
    llm_enabled: bool = False
    llm_provider: str = "template"
    llm_model: str | None = None
    llm_used: bool = False
    fallback_used: bool = True
    fallback_reason: str | None = None
    llm_error_type: str | None = None
    llm_error_message_preview: str | None = None


class DiagnosisResponse(BaseModel):
    task_id: str
    status: str
    message: str
    state: DiagnosisState | None = None
