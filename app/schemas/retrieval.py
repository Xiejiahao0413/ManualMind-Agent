from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float = 0.0
    device_name: str | None = None
    device_model: str | None = None
    section_title: str | None = None
    page: int | None = None
    content_type: str | None = None
    fault_code: str | None = None
    source_file: str | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
