from pydantic import BaseModel, Field


class HandoffPayload(BaseModel):
    task_id: str
    session_id: str
    reason: str
    sanitized_summary: str
    risk_level: str = "unknown"
    evidence_ids: list[str] = Field(default_factory=list)
    created_by: str = "system"
