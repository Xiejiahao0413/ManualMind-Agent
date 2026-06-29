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
    ) -> None:
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.dimension = dimension
        self._client = None

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
            entity = hit.get("entity", {})
            source_file = entity.get("source_file") or None
            results.append(
                RetrievalResult(
                    chunk_id=str(entity.get("chunk_id") or hit.get("id") or ""),
                    doc_id=str(entity.get("doc_id") or ""),
                    text=str(entity.get("text") or ""),
                    score=float(hit.get("distance") or hit.get("score") or 0.0),
                    dense_score=float(hit.get("distance") or hit.get("score") or 0.0),
                    source="milvus",
                    source_refs=[source_file] if source_file else [],
                    device_name=entity.get("device_name") or None,
                    device_model=entity.get("device_model") or None,
                    section_title=entity.get("section_title") or None,
                    page=int(entity["page"]) if entity.get("page") else None,
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
            client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                primary_field_name="chunk_id",
                vector_field_name="embedding",
                metric_type="COSINE",
                auto_id=False,
            )
        except Exception as exc:
            raise VectorStoreError("milvus_collection_init_failed") from exc


def _build_filter_expr(filters: dict[str, Any] | None = None) -> str:
    if not filters:
        return ""
    supported = {"device_model", "fault_code", "content_type", "source_file"}
    parts: list[str] = []
    for key, value in filters.items():
        if key not in supported or value is None:
            continue
        if isinstance(value, list | tuple | set):
            values = ", ".join(f'"{item}"' for item in value)
            parts.append(f"{key} in [{values}]")
        else:
            parts.append(f'{key} == "{value}"')
    return " and ".join(parts)
