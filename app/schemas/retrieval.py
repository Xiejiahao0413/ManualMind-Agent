from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    device_name: str | None = None
    device_model: str | None = None
    section_title: str | None = None
    page: int | None = None
    content_type: str | None = None
    fault_code: str | None = None
    source_file: str | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float = 0.0
    bm25_score: float | None = None
    dense_score: float | None = None
    rerank_score: float | None = None
    source: str = "unknown"
    source_refs: list[str] = Field(default_factory=list)
    device_name: str | None = None
    device_model: str | None = None
    section_title: str | None = None
    page: int | None = None
    content_type: str | None = None
    fault_code: str | None = None
    source_file: str | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @classmethod
    def from_chunk(
        cls,
        chunk: DocumentChunk,
        score: float = 0.0,
        source: str = "unknown",
        bm25_score: float | None = None,
        dense_score: float | None = None,
        rerank_score: float | None = None,
    ) -> "RetrievalResult":
        return cls(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            score=score,
            bm25_score=bm25_score,
            dense_score=dense_score,
            rerank_score=rerank_score,
            source=source,
            source_refs=[chunk.source_file] if chunk.source_file else [],
            device_name=chunk.device_name,
            device_model=chunk.device_model,
            section_title=chunk.section_title,
            page=chunk.page,
            content_type=chunk.content_type,
            fault_code=chunk.fault_code,
            source_file=chunk.source_file,
            metadata=chunk.metadata,
        )
