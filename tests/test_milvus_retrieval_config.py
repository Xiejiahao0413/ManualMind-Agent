import importlib
import json
import subprocess
import sys

from app.vectorstore import InMemoryVectorStore, MilvusVectorStore
from app.vectorstore.factory import create_vector_runtime
from tests.test_vectorstore import sample_chunks


def test_milvus_store_import_does_not_require_connection() -> None:
    module = importlib.import_module("app.vectorstore.milvus_store")

    assert hasattr(module, "MilvusVectorStore")


def test_unconfigured_milvus_runtime_falls_back_to_memory() -> None:
    runtime = create_vector_runtime({"MANUALMIND_VECTOR_BACKEND": "milvus"})

    assert isinstance(runtime.vector_store, InMemoryVectorStore)
    assert runtime.fallback_used is True
    assert runtime.status.skipped_reason in {"milvus_uri_not_configured", "pymilvus_not_installed"}


def test_unconfigured_new_milvus_runtime_falls_back_to_memory() -> None:
    runtime = create_vector_runtime({"VECTORSTORE_BACKEND": "milvus"})

    assert isinstance(runtime.vector_store, InMemoryVectorStore)
    assert runtime.fallback_used is True
    assert runtime.status.skipped_reason in {"milvus_uri_not_configured", "pymilvus_not_installed"}


def test_milvus_status_reports_clear_error_without_uri() -> None:
    store = MilvusVectorStore(uri=None)
    status = store.status()

    assert status.available is False
    assert status.skipped_reason == "milvus_uri_not_configured"


def test_default_runtime_uses_memory_backend() -> None:
    runtime = create_vector_runtime({})

    assert isinstance(runtime.vector_store, InMemoryVectorStore)
    assert runtime.vector_store.backend == "memory"
    assert runtime.embedding_client.provider == "mock"


def test_local_runtime_uses_memory_backend() -> None:
    runtime = create_vector_runtime({"VECTORSTORE_BACKEND": "local", "EMBEDDING_PROVIDER": "mock"})

    assert isinstance(runtime.vector_store, InMemoryVectorStore)
    assert runtime.vector_store.backend == "memory"
    assert runtime.embedding_client.provider == "mock"


def test_mock_milvus_upsert_search_and_doc_id_filter() -> None:
    client = FakeMilvusClient()
    store = MilvusVectorStore(uri="mock://milvus", collection_name="manualmind_test", dimension=4, client=client)
    chunks = sample_chunks()

    store.upsert_chunks(chunks, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=5, filters={"doc_id": ["a100"]})

    assert client.created_collection == "manualmind_test"
    assert len(client.rows) == 2
    assert results
    assert [result.doc_id for result in results] == ["a100"]
    assert results[0].source_refs == ["a100_manual.md:3"]
    assert results[0].source_file == "a100_manual.md"
    assert results[0].page == 3
    assert results[0].section_title is None
    assert 'doc_id in ["a100"]' in client.last_filter


def test_mock_milvus_delete_doc_removes_rows() -> None:
    client = FakeMilvusClient()
    store = MilvusVectorStore(uri="mock://milvus", collection_name="manualmind_test", dimension=4, client=client)
    chunks = sample_chunks()
    store.upsert_chunks(chunks, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    store.delete_doc("a100")
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=5)

    assert [row["doc_id"] for row in client.rows] == ["b200"]
    assert all(result.doc_id != "a100" for result in results)


def test_index_milvus_manuals_script_runs_without_milvus() -> None:
    output = _run_script("scripts/index_milvus_manuals.py")

    assert output["vector_backend"] == "memory"
    assert output["embedding_provider"] == "mock"
    assert output["indexed_chunks_count"] > 0
    assert "source_files" in output


def test_demo_milvus_retrieval_script_runs_without_milvus() -> None:
    output = _run_script("scripts/demo_milvus_retrieval.py")

    assert output["vector_backend"] == "memory"
    assert output["embedding_provider"] == "mock"
    assert output["top_k_results"]
    first = output["top_k_results"][0]
    assert first["source_refs"]
    assert first["device_model"]
    assert first["page"] is not None


def _run_script(script_path: str) -> dict:
    result = subprocess.run(
        [sys.executable, script_path],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return json.loads(result.stdout)


class FakeMilvusClient:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.created_collection: str | None = None
        self.last_filter = ""

    def list_collections(self) -> list[str]:
        return [self.created_collection] if self.created_collection else []

    def has_collection(self, collection_name: str) -> bool:
        return self.created_collection == collection_name

    def create_collection(self, collection_name: str, **kwargs) -> None:
        self.created_collection = collection_name

    def upsert(self, collection_name: str, data: list[dict]) -> None:
        existing = {row["chunk_id"]: row for row in self.rows}
        for row in data:
            existing[row["chunk_id"]] = row
        self.rows = list(existing.values())

    def search(
        self,
        collection_name: str,
        data: list[list[float]],
        limit: int,
        filter: str,
        output_fields: list[str],
    ) -> list[list[dict]]:
        self.last_filter = filter
        rows = [row for row in self.rows if self._matches_filter(row, filter)]
        hits = [
            {
                "id": row["chunk_id"],
                "distance": 1.0 / (index + 1),
                "entity": {field: row.get(field) for field in output_fields},
            }
            for index, row in enumerate(rows[:limit])
        ]
        return [hits]

    def delete(self, collection_name: str, filter: str) -> None:
        if 'doc_id == "' not in filter:
            return
        doc_id = filter.split('doc_id == "', 1)[1].split('"', 1)[0]
        self.rows = [row for row in self.rows if row["doc_id"] != doc_id]

    def _matches_filter(self, row: dict, filter: str) -> bool:
        if not filter:
            return True
        if 'doc_id in ["a100"]' in filter:
            return row["doc_id"] == "a100"
        if 'doc_id in ["b200"]' in filter:
            return row["doc_id"] == "b200"
        return True
