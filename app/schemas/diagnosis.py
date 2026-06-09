from typing import Literal

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
    device_name: str | None = None
    device_model: str | None = None
    fault_code: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    query_type: str | None = None
    risk_level: Literal["low", "medium", "high", "unknown"] = "unknown"
    retrieval_status: str = "not_started"
    retrieved_evidence: list[RetrievalResult] = Field(default_factory=list)
    tool_call_history: list[ToolCallRecord] = Field(default_factory=list)
    retry_count: int = 0
    max_retry: int = 3


class DiagnosisResponse(BaseModel):
    task_id: str
    status: str
    message: str
    state: DiagnosisState | None = None
