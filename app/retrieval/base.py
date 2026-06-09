from abc import ABC, abstractmethod
from typing import Any

from app.schemas.retrieval import DocumentChunk, RetrievalResult


class BM25Retriever(ABC):
    @abstractmethod
    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        raise NotImplementedError


class MilvusDenseRetriever(ABC):
    @abstractmethod
    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        raise NotImplementedError

    @abstractmethod
    def build_filter_expr(self, metadata_filter: dict[str, Any] | None = None) -> str:
        raise NotImplementedError


class BGEReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_n: int = 5,
    ) -> list[RetrievalResult]:
        raise NotImplementedError


class HybridRetriever(ABC):
    @abstractmethod
    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        raise NotImplementedError
