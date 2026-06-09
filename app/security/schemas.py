from pydantic import BaseModel, Field


class SensitiveSpan(BaseModel):
    type: str
    value: str
    start: int
    end: int
    replacement: str


class SanitizedText(BaseModel):
    original_text: str
    sanitized_text: str
    spans: list[SensitiveSpan] = Field(default_factory=list)
    has_sensitive_data: bool = False
