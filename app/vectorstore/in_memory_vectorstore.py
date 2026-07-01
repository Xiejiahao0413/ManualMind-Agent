from math import sqrt
from typing import Any

from app.retrieval.filters import matches_metadata_filter
from app.schemas.retrieval import DocumentChunk, RetrievalResult
from app.vectorstore.base import VectorStore


class InMemoryVectorStore(VectorStore):
    backend = "memory"

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._embeddings: list[list[float]] = []

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        existing_ids = {chunk.chunk_id: index for index, chunk in enumerate(self._chunks)}
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            existing_index = existing_ids.get(chunk.chunk_id)
            if existing_index is None:
                self._chunks.append(chunk)
                self._embeddings.append(embedding)
            else:
                self._chunks[existing_index] = chunk
                self._embeddings[existing_index] = embedding

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        scored: list[RetrievalResult] = []
        for chunk, embedding in zip(self._chunks, self._embeddings, strict=True):
            if not matches_metadata_filter(chunk, filters):
                continue
            score = cosine_similarity(query_embedding, embedding)
            if score <= 0:
                continue
            scored.append(
                RetrievalResult.from_chunk(
                    chunk,
                    score=score,
                    source="vector",
                    dense_score=score,
                )
            )
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]

    def delete_doc(self, doc_id: str) -> None:
        kept_chunks: list[DocumentChunk] = []
        kept_embeddings: list[list[float]] = []
        for chunk, embedding in zip(self._chunks, self._embeddings, strict=True):
            if chunk.doc_id == doc_id:
                continue
            kept_chunks.append(chunk)
            kept_embeddings.append(embedding)
        self._chunks = kept_chunks
        self._embeddings = kept_embeddings


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    return dot / (left_norm * right_norm)
