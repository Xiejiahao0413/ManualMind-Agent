import os
from collections.abc import Mapping
from dataclasses import dataclass

from app.embeddings import EmbeddingClient, MockEmbeddingClient, OpenAIEmbeddingClient
from app.vectorstore import InMemoryVectorStore, MilvusVectorStore, VectorStore, VectorStoreStatus


@dataclass(frozen=True)
class VectorRuntime:
    vector_store: VectorStore
    embedding_client: EmbeddingClient
    status: VectorStoreStatus
    fallback_used: bool = False


def create_embedding_client(env: Mapping[str, str] | None = None) -> tuple[EmbeddingClient, bool, str | None]:
    active_env = os.environ if env is None else env
    provider = active_env.get("MANUALMIND_EMBEDDING_PROVIDER", "mock").lower().strip()
    dimension = int(active_env.get("MANUALMIND_EMBEDDING_DIMENSION", "64"))
    if provider == "openai":
        api_key = active_env.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return MockEmbeddingClient(dimension=dimension), True, "openai_api_key_missing"
        model = active_env.get("MANUALMIND_EMBEDDING_MODEL", "text-embedding-3-small")
        timeout = float(active_env.get("MANUALMIND_EMBEDDING_TIMEOUT_SECONDS", "20"))
        openai_dimension = int(active_env.get("MANUALMIND_EMBEDDING_DIMENSION", "1536"))
        return (
            OpenAIEmbeddingClient(
                api_key=api_key,
                model=model,
                dimension=openai_dimension,
                timeout_seconds=timeout,
            ),
            False,
            None,
        )
    return MockEmbeddingClient(dimension=dimension), provider not in {"", "mock"}, None


def create_vector_runtime(env: Mapping[str, str] | None = None) -> VectorRuntime:
    active_env = os.environ if env is None else env
    embedding_client, embedding_fallback, embedding_reason = create_embedding_client(active_env)
    backend = active_env.get("MANUALMIND_VECTOR_BACKEND", "memory").lower().strip()
    if backend == "milvus":
        milvus_store = MilvusVectorStore(
            uri=active_env.get("MILVUS_URI"),
            token=active_env.get("MILVUS_TOKEN"),
            collection_name=active_env.get("MANUALMIND_MILVUS_COLLECTION", "manualmind_chunks"),
            dimension=embedding_client.dimension,
        )
        status = milvus_store.status()
        if status.available and not embedding_fallback:
            return VectorRuntime(
                vector_store=milvus_store,
                embedding_client=embedding_client,
                status=status,
                fallback_used=False,
            )
        skipped_reason = status.skipped_reason or embedding_reason or "milvus_unavailable"
        return VectorRuntime(
            vector_store=InMemoryVectorStore(),
            embedding_client=MockEmbeddingClient(
                dimension=int(active_env.get("MANUALMIND_EMBEDDING_DIMENSION", "64"))
            ),
            status=VectorStoreStatus(
                backend="memory",
                available=True,
                fallback_used=True,
                skipped_reason=skipped_reason,
            ),
            fallback_used=True,
        )

    return VectorRuntime(
        vector_store=InMemoryVectorStore(),
        embedding_client=embedding_client,
        status=VectorStoreStatus(
            backend="memory",
            available=True,
            fallback_used=True,
            skipped_reason=embedding_reason or "milvus_not_enabled",
        ),
        fallback_used=True,
    )
