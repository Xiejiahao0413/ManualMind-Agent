from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallRecord(BaseModel):
    task_id: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    args_signature: str
    status: Literal["pending", "success", "failed", "skipped"] = "pending"
    retry_count: int = 0
    fallback_used: bool = False
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
