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
    provider = _env_value(active_env, "EMBEDDING_PROVIDER", "MANUALMIND_EMBEDDING_PROVIDER", default="mock")
    provider = provider.lower().strip()
    dimension = _env_int(active_env, "EMBEDDING_DIM", "MANUALMIND_EMBEDDING_DIMENSION", "MILVUS_DIM", default=64)
    if provider == "openai":
        api_key = active_env.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return MockEmbeddingClient(dimension=dimension), True, "openai_api_key_missing"
        model = _env_value(
            active_env,
            "EMBEDDING_MODEL",
            "MANUALMIND_EMBEDDING_MODEL",
            default="text-embedding-3-small",
        )
        timeout = float(active_env.get("MANUALMIND_EMBEDDING_TIMEOUT_SECONDS", "20"))
        openai_dimension = _env_int(
            active_env,
            "EMBEDDING_DIM",
            "MANUALMIND_EMBEDDING_DIMENSION",
            "MILVUS_DIM",
            default=1536,
        )
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
    backend = _env_value(active_env, "VECTORSTORE_BACKEND", "MANUALMIND_VECTOR_BACKEND", default="local")
    backend = _normalize_backend(backend)
    if backend == "milvus":
        milvus_store = MilvusVectorStore(
            uri=active_env.get("MILVUS_URI"),
            token=active_env.get("MILVUS_TOKEN"),
            collection_name=_env_value(
                active_env,
                "MILVUS_COLLECTION",
                "MANUALMIND_MILVUS_COLLECTION",
                default="manualmind_chunks",
            ),
            dimension=_env_int(
                active_env,
                "MILVUS_DIM",
                "EMBEDDING_DIM",
                "MANUALMIND_EMBEDDING_DIMENSION",
                default=embedding_client.dimension,
            ),
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
                dimension=_env_int(
                    active_env,
                    "EMBEDDING_DIM",
                    "MANUALMIND_EMBEDDING_DIMENSION",
                    "MILVUS_DIM",
                    default=64,
                )
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


def _env_value(active_env: Mapping[str, str], *names: str, default: str) -> str:
    for name in names:
        value = active_env.get(name)
        if value is not None and value.strip():
            return value
    return default


def _env_int(active_env: Mapping[str, str], *names: str, default: int) -> int:
    raw = _env_value(active_env, *names, default=str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def _normalize_backend(backend: str) -> str:
    normalized = backend.lower().strip()
    if normalized in {"local", "memory", "in_memory", "in-memory"}:
        return "memory"
    return normalized
