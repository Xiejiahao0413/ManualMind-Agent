from dataclasses import dataclass
from uuid import uuid4

from app.schemas.document import ManualUploadResult


@dataclass(frozen=True)
class StoredManual:
    doc_id: str
    filename: str
    content_type: str | None
    content: bytes


class InMemoryManualStore:
    def __init__(self) -> None:
        self._documents: dict[str, StoredManual] = {}

    def save(
        self,
        filename: str,
        content: bytes,
        content_type: str | None = None,
        doc_id: str | None = None,
    ) -> ManualUploadResult:
        document_id = doc_id or f"manual-{uuid4().hex[:12]}"
        self._documents[document_id] = StoredManual(
            doc_id=document_id,
            filename=filename,
            content_type=content_type,
            content=content,
        )
        return ManualUploadResult(
            doc_id=document_id,
            filename=filename,
            content_type=content_type,
            size=len(content),
        )

    def get(self, doc_id: str) -> StoredManual | None:
        return self._documents.get(doc_id)
