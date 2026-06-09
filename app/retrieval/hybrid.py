from typing import Any

from app.retrieval.bm25 import LocalBM25Retriever
from app.retrieval.dense import InMemoryDenseRetriever
from app.retrieval.merge import merge_retrieval_results
from app.retrieval.reranker import MockReranker, Reranker
from app.schemas.retrieval import DocumentChunk, RetrievalResult


class HybridRetrieverImpl:
    def __init__(
        self,
        bm25_retriever: LocalBM25Retriever | None = None,
        dense_retriever: InMemoryDenseRetriever | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.bm25_retriever = bm25_retriever or LocalBM25Retriever()
        self.dense_retriever = dense_retriever or InMemoryDenseRetriever()
        self.reranker = reranker or MockReranker()

    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        self.bm25_retriever.add_documents(chunks)
        self.dense_retriever.add_documents(chunks)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        top_k_bm25: int = 10,
        top_k_dense: int = 10,
        top_n_rerank: int | None = None,
    ) -> list[RetrievalResult]:
        rerank_limit = top_n_rerank or top_k
        bm25_results = await self.bm25_retriever.search(
            query,
            top_k=top_k_bm25,
            metadata_filter=metadata_filter,
        )
        dense_results = await self.dense_retriever.search(
            query,
            top_k=top_k_dense,
            metadata_filter=metadata_filter,
        )
        candidates = merge_retrieval_results(bm25_results, dense_results)
        try:
            return await self.reranker.rerank(query, candidates, top_n=rerank_limit)
        except Exception:
            return sorted(candidates, key=lambda result: result.score, reverse=True)[:rerank_limit]
