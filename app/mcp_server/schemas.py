from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.handoff import HandoffPayload
from app.schemas.retrieval import RetrievalResult


class ManualHybridSearchRequest(BaseModel):
    query: str
    device_name: str | None = None
    device_model: str | None = None
    doc_ids: list[str] = Field(default_factory=list)
    content_types: list[str] = Field(default_factory=list)
    top_k_bm25: int = 5
    top_k_dense: int = 5
    top_n_rerank: int = 5


class ManualHybridSearchResponse(BaseModel):
    results: list[RetrievalResult] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    retrieval_mode: str = "hybrid"


class FaultCodeLookupRequest(BaseModel):
    fault_code: str
    device_model: str | None = None
    doc_ids: list[str] = Field(default_factory=list)


class FaultCodeLookupResponse(BaseModel):
    status: Literal["found", "not_found"]
    fault_code: str
    device_model: str | None = None
    description: str | None = None
    source_refs: list[str] = Field(default_factory=list)


class ParameterLookupRequest(BaseModel):
    parameter_name: str
    device_model: str | None = None
    observed_value: float | None = None
    doc_ids: list[str] = Field(default_factory=list)


class ParameterLookupResponse(BaseModel):
    status: Literal["found", "not_found"]
    parameter_name: str
    standard_range: str | None = None
    observed_value: float | None = None
    is_abnormal: bool | None = None
    source_refs: list[str] = Field(default_factory=list)


class SafetyRuleSearchRequest(BaseModel):
    operation: str
    risk_level: str = "unknown"
    device_model: str | None = None
    doc_ids: list[str] = Field(default_factory=list)


class SafetyRuleSearchResponse(BaseModel):
    safety_rules: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    risk_level: str = "unknown"


class SourceTraceRequest(BaseModel):
    chunk_ids: list[str]
    doc_ids: list[str] = Field(default_factory=list)


class SourceTraceItem(BaseModel):
    chunk_id: str
    source_file: str | None = None
    page: int | None = None
    section_title: str | None = None
    content_type: str | None = None


class SourceTraceResponse(BaseModel):
    sources: list[SourceTraceItem] = Field(default_factory=list)


class HandoffTicketCreateRequest(BaseModel):
    payload: HandoffPayload


class HandoffTicketCreateResponse(BaseModel):
    handoff_id: str
    status: str
    handoff_reason: str | None = None
    risk_level: str = "unknown"


class ToolExecutionResult(BaseModel):
    tool_name: str
    status: Literal["success", "error"]
    data: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    latency_ms: float
