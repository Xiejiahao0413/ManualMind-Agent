from abc import ABC, abstractmethod

from app.schemas.retrieval import RetrievalResult


class BM25Retriever(ABC):
    @abstractmethod
    async def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        raise NotImplementedError


class MilvusDenseRetriever(ABC):
    @abstractmethod
    async def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        raise NotImplementedError


class BGEReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        raise NotImplementedError


class HybridRetriever(ABC):
    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        raise NotImplementedError
