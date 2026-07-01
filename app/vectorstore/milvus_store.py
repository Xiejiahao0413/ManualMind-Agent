from typing import Any

from app.schemas.retrieval import DocumentChunk, RetrievalResult
from app.vectorstore.base import VectorStore, VectorStoreError, VectorStoreStatus


class MilvusVectorStore(VectorStore):
    backend = "milvus"

    def __init__(
        self,
        uri: str | None,
        token: str | None = None,
        collection_name: str = "manualmind_chunks",
        dimension: int = 1536,
        client: Any | None = None,
    ) -> None:
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.dimension = dimension
        self._client = client

    def status(self) -> VectorStoreStatus:
        if not self.uri:
            return VectorStoreStatus(
                backend=self.backend,
                available=False,
                skipped_reason="milvus_uri_not_configured",
            )
        try:
            client = self._ensure_client()
            client.list_collections()
        except VectorStoreError as exc:
            return VectorStoreStatus(
                backend=self.backend,
                available=False,
                skipped_reason=str(exc),
            )
        except Exception:
            return VectorStoreStatus(
                backend=self.backend,
                available=False,
                skipped_reason="milvus_connection_failed",
            )
        return VectorStoreStatus(backend=self.backend, available=True)

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        client = self._ensure_client()
        self._ensure_collection(client)
        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "source_file": chunk.source_file or "",
                "device_name": chunk.device_name or "",
                "device_model": chunk.device_model or "",
                "section_title": chunk.section_title or "",
                "page": int(chunk.page or 0),
                "content_type": chunk.content_type or "",
                "fault_code": chunk.fault_code or "",
                "embedding": embedding,
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        try:
            client.upsert(collection_name=self.collection_name, data=rows)
        except Exception as exc:
            raise VectorStoreError("milvus_upsert_failed") from exc

    def delete_doc(self, doc_id: str) -> None:
        if not doc_id:
            return
        client = self._ensure_client()
        self._ensure_collection(client)
        try:
            client.delete(
                collection_name=self.collection_name,
                filter=f'doc_id == "{_escape_filter_value(doc_id)}"',
            )
        except Exception as exc:
            raise VectorStoreError("milvus_delete_failed") from exc

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        client = self._ensure_client()
        self._ensure_collection(client)
        try:
            hits = client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                filter=_build_filter_expr(filters),
                output_fields=[
                    "chunk_id",
                    "doc_id",
                    "text",
                    "source_file",
                    "device_name",
                    "device_model",
                    "section_title",
                    "page",
                    "content_type",
                    "fault_code",
                ],
            )
        except Exception as exc:
            raise VectorStoreError("milvus_search_failed") from exc

        results: list[RetrievalResult] = []
        for hit in hits[0] if hits else []:
            entity = _hit_entity(hit)
            source_file = entity.get("source_file") or None
            page = _coerce_page(entity.get("page"))
            score = _hit_score(hit)
            results.append(
                RetrievalResult(
                    chunk_id=str(entity.get("chunk_id") or _hit_value(hit, "id", "") or ""),
                    doc_id=str(entity.get("doc_id") or ""),
                    text=str(entity.get("text") or ""),
                    score=score,
                    dense_score=score,
                    source="milvus",
                    source_refs=_source_refs(source_file, page),
                    device_name=entity.get("device_name") or None,
                    device_model=entity.get("device_model") or None,
                    section_title=entity.get("section_title") or None,
                    page=page,
                    content_type=entity.get("content_type") or None,
                    fault_code=entity.get("fault_code") or None,
                    source_file=source_file,
                )
            )
        return results

    def _ensure_client(self):
        if not self.uri:
            raise VectorStoreError("milvus_uri_not_configured")
        if self._client is not None:
            return self._client
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise VectorStoreError("pymilvus_not_installed") from exc
        try:
            kwargs = {"uri": self.uri}
            if self.token:
                kwargs["token"] = self.token
            self._client = MilvusClient(**kwargs)
            return self._client
        except Exception as exc:
            raise VectorStoreError("milvus_connection_failed") from exc

    def _ensure_collection(self, client) -> None:
        try:
            if client.has_collection(self.collection_name):
                return
            self._create_collection(client)
        except Exception as exc:
            raise VectorStoreError("milvus_collection_init_failed") from exc

    def _create_collection(self, client) -> None:
        try:
            from pymilvus import DataType

            schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=256)
            schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
            schema.add_field("text", DataType.VARCHAR, max_length=8192)
            schema.add_field("source_file", DataType.VARCHAR, max_length=512)
            schema.add_field("device_name", DataType.VARCHAR, max_length=256)
            schema.add_field("device_model", DataType.VARCHAR, max_length=256)
            schema.add_field("section_title", DataType.VARCHAR, max_length=512)
            schema.add_field("page", DataType.INT64)
            schema.add_field("content_type", DataType.VARCHAR, max_length=128)
            schema.add_field("fault_code", DataType.VARCHAR, max_length=128)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dimension)
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            return
        except (ImportError, AttributeError, TypeError):
            pass

        try:
            client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                primary_field_name="chunk_id",
                vector_field_name="embedding",
                metric_type="COSINE",
                auto_id=False,
                enable_dynamic_field=True,
            )
        except TypeError:
            client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                primary_field_name="chunk_id",
                vector_field_name="embedding",
                metric_type="COSINE",
                auto_id=False,
            )


def _build_filter_expr(filters: dict[str, Any] | None = None) -> str:
    if not filters:
        return ""
    supported = {"device_name", "device_model", "fault_code", "content_type", "source_file", "doc_id"}
    parts: list[str] = []
    for key, value in filters.items():
        if key not in supported or value is None:
            continue
        if isinstance(value, list | tuple | set):
            values = ", ".join(f'"{_escape_filter_value(item)}"' for item in value if item is not None)
            if not values:
                continue
            parts.append(f"{key} in [{values}]")
        else:
            parts.append(f'{key} == "{_escape_filter_value(value)}"')
    return " and ".join(parts)


def _escape_filter_value(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _hit_entity(hit: Any) -> dict[str, Any]:
    if isinstance(hit, dict):
        entity = hit.get("entity") or hit.get("fields") or {}
        return dict(entity)
    entity = getattr(hit, "entity", None) or getattr(hit, "fields", None) or {}
    return dict(entity)


def _hit_value(hit: Any, key: str, default: Any = None) -> Any:
    if isinstance(hit, dict):
        return hit.get(key, default)
    return getattr(hit, key, default)


def _hit_score(hit: Any) -> float:
    value = _hit_value(hit, "distance")
    if value is None:
        value = _hit_value(hit, "score", 0.0)
    return float(value or 0.0)


def _coerce_page(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def _source_refs(source_file: str | None, page: int | None) -> list[str]:
    if source_file and page is not None:
        return [f"{source_file}:{page}"]
    if source_file:
        return [source_file]
    return []
