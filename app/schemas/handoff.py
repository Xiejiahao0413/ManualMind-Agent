from pydantic import BaseModel, Field


class HandoffPayload(BaseModel):
    task_id: str
    session_id: str
    reason: str | None = None
    handoff_reason: str | None = None
    sanitized_summary: str = ""
    risk_level: str = "unknown"
    evidence_ids: list[str] = Field(default_factory=list)
    device_name: str | None = None
    fault_code: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    tool_trace: list[dict[str, str | int | float | bool | None]] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    created_by: str = "system"
