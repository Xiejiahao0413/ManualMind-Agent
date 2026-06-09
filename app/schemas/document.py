from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.retrieval import DocumentChunk as RetrievalDocumentChunk


DocumentChunk = RetrievalDocumentChunk


class ManualDocument(BaseModel):
    doc_id: str
    filename: str
    text: str
    device_name: str | None = None
    device_model: str | None = None
    source_file: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    sanitized_fields: list[str] = Field(default_factory=list)


class ManualUploadResult(BaseModel):
    doc_id: str
    filename: str
    content_type: str | None = None
    size: int
    status: str = "received"


class DocumentIndexResult(BaseModel):
    doc_id: str
    status: str
    chunks_count: int = 0
    content_type_stats: dict[str, int] = Field(default_factory=dict)
    sanitized_fields: list[str] = Field(default_factory=list)


def normalize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, str | int | float | bool | None]:
    normalized: dict[str, str | int | float | bool | None] = {}
    for key, value in (metadata or {}).items():
        if value is None or isinstance(value, str | int | float | bool):
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized
