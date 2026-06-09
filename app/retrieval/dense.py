from typing import Any

from app.retrieval.embeddings import EmbeddingProvider, MockEmbeddingProvider, cosine_similarity
from app.retrieval.filters import matches_metadata_filter
from app.schemas.retrieval import DocumentChunk, RetrievalResult


class InMemoryDenseRetriever:
    def __init__(self, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.embedding_provider = embedding_provider or MockEmbeddingProvider()
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[list[float]] = []

    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        self._chunks.extend(chunks)
        self._vectors.extend(self.embedding_provider.embed_documents([chunk.text for chunk in chunks]))

    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        query_vector = self.embedding_provider.embed_query(query)
        scored: list[RetrievalResult] = []
        for chunk, vector in zip(self._chunks, self._vectors, strict=True):
            if not matches_metadata_filter(chunk, metadata_filter):
                continue
            score = cosine_similarity(query_vector, vector)
            if score <= 0:
                continue
            scored.append(
                RetrievalResult.from_chunk(
                    chunk,
                    score=score,
                    source="dense",
                    dense_score=score,
                )
            )
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]

    def build_filter_expr(self, metadata_filter: dict[str, Any] | None = None) -> str:
        if not metadata_filter:
            return ""
        parts: list[str] = []
        for key, value in metadata_filter.items():
            if isinstance(value, list | tuple | set):
                values = ", ".join(f'"{item}"' for item in value)
                parts.append(f"{key} in [{values}]")
            else:
                parts.append(f'{key} == "{value}"')
        return " and ".join(parts)


MockMilvusDenseRetriever = InMemoryDenseRetriever
