import importlib
import json
import subprocess
import sys

from app.vectorstore import InMemoryVectorStore, MilvusVectorStore
from app.vectorstore.factory import create_vector_runtime


def test_milvus_store_import_does_not_require_connection() -> None:
    module = importlib.import_module("app.vectorstore.milvus_store")

    assert hasattr(module, "MilvusVectorStore")


def test_unconfigured_milvus_runtime_falls_back_to_memory() -> None:
    runtime = create_vector_runtime({"MANUALMIND_VECTOR_BACKEND": "milvus"})

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
