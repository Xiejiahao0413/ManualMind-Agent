from typing import Any

from app.embeddings import EmbeddingClient, MockEmbeddingClient
from app.schemas.retrieval import DocumentChunk, RetrievalResult
from app.vectorstore import InMemoryVectorStore, VectorStore


class VectorDenseRetriever:
    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.embedding_client = embedding_client or MockEmbeddingClient()
        self.vector_store = vector_store or InMemoryVectorStore()

    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        embeddings = self.embedding_client.embed_texts([chunk.text for chunk in chunks])
        self.vector_store.upsert_chunks(chunks, embeddings)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        query_embedding = self.embedding_client.embed_text(query)
        return self.vector_store.search(query_embedding, top_k=top_k, filters=metadata_filter)

    def build_filter_expr(self, metadata_filter: dict[str, Any] | None = None) -> str:
        if not metadata_filter:
            return ""
        return " and ".join(f"{key}={value}" for key, value in metadata_filter.items())
