from abc import ABC, abstractmethod

from app.retrieval.bm25 import tokenize_text
from app.schemas.retrieval import RetrievalResult


class Reranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_n: int = 5,
    ) -> list[RetrievalResult]:
        raise NotImplementedError


class MockReranker(Reranker):
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_n: int = 5,
    ) -> list[RetrievalResult]:
        if self.should_fail:
            raise RuntimeError("mock rerank failure")

        query_tokens = set(tokenize_text(query))
        reranked: list[RetrievalResult] = []
        for candidate in candidates:
            candidate_tokens = set(tokenize_text(candidate.text))
            overlap = len(query_tokens & candidate_tokens)
            rerank_score = float(overlap) + candidate.score
            reranked.append(candidate.model_copy(update={"rerank_score": rerank_score}))
        return sorted(reranked, key=lambda result: result.rerank_score or 0.0, reverse=True)[:top_n]
