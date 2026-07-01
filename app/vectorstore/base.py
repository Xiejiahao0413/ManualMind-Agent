from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.schemas.retrieval import DocumentChunk, RetrievalResult


class VectorStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class VectorStoreStatus:
    backend: str
    available: bool
    fallback_used: bool = False
    skipped_reason: str | None = None


class VectorStore(ABC):
    backend: str = "unknown"

    @abstractmethod
    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        raise NotImplementedError

    def delete_doc(self, doc_id: str) -> None:
        raise NotImplementedError

    def status(self) -> VectorStoreStatus:
        return VectorStoreStatus(backend=self.backend, available=True)
