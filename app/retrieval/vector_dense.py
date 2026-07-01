from typing import Any

from app.embeddings import EmbeddingClient, MockEmbeddingClient
from app.schemas.retrieval import DocumentChunk, RetrievalResult
from app.vectorstore.base import VectorStore, VectorStoreError
from app.vectorstore.in_memory_vectorstore import InMemoryVectorStore


class VectorDenseRetriever:
    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        vector_store: VectorStore | None = None,
        fallback_store: InMemoryVectorStore | None = None,
    ) -> None:
        self.embedding_client = embedding_client or MockEmbeddingClient()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.fallback_store = fallback_store or InMemoryVectorStore()
        self._mirror_fallback = self.vector_store.backend != self.fallback_store.backend
        self._using_fallback = False

    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        embeddings = self.embedding_client.embed_texts([chunk.text for chunk in chunks])
        if self._mirror_fallback or self._using_fallback:
            self.fallback_store.upsert_chunks(chunks, embeddings)
        if self._using_fallback:
            return
        try:
            self.vector_store.upsert_chunks(chunks, embeddings)
        except VectorStoreError:
            self._using_fallback = True
            if not self._mirror_fallback:
                self.fallback_store.upsert_chunks(chunks, embeddings)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        query_embedding = self.embedding_client.embed_text(query)
        if self._using_fallback:
            return self.fallback_store.search(query_embedding, top_k=top_k, filters=metadata_filter)
        try:
            return self.vector_store.search(query_embedding, top_k=top_k, filters=metadata_filter)
        except VectorStoreError:
            self._using_fallback = True
            return self.fallback_store.search(query_embedding, top_k=top_k, filters=metadata_filter)

    def delete_doc(self, doc_id: str) -> None:
        self.fallback_store.delete_doc(doc_id)
        if self._using_fallback:
            return
        try:
            self.vector_store.delete_doc(doc_id)
        except VectorStoreError:
            self._using_fallback = True

    def build_filter_expr(self, metadata_filter: dict[str, Any] | None = None) -> str:
        if not metadata_filter:
            return ""
        return " and ".join(f"{key}={value}" for key, value in metadata_filter.items())
