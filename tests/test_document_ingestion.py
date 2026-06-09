import asyncio

from fastapi.testclient import TestClient

from app.core.dependencies import get_manual_indexer
from app.ingestion import ChunkMetadataBuilder, ManualIndexer, SectionSplitter, TextManualParser
from app.main import app
from app.mcp_server import MCPToolExecutor, create_default_tool_registry
from app.schemas.document import ManualDocument


MANUAL_TEXT = """# Fault Codes
E03 Motor overheat. Check cooling fan and temperature sensor.

# Safety Rules
Warning: disconnect power and release pressure before maintenance. Contact 13800138000 or ops@example.com. PLC IP 192.168.1.8.

# Parameters
Temperature threshold is 0-80 C.
"""


def test_txt_document_can_be_parsed_and_sanitized() -> None:
    parser = TextManualParser()

    document = parser.parse(
        MANUAL_TEXT.encode("utf-8"),
        filename="mx100.txt",
        doc_id="manual-test",
        device_name="Compressor",
        device_model="MX100",
    )

    assert document.doc_id == "manual-test"
    assert "[PHONE]" in document.text
    assert "[EMAIL]" in document.text
    assert "[IP_ADDRESS]" in document.text
    assert "PHONE" in document.sanitized_fields


def test_md_document_can_be_split_into_typed_chunks() -> None:
    parser = TextManualParser()
    splitter = SectionSplitter()
    builder = ChunkMetadataBuilder(splitter)
    document = parser.parse(MANUAL_TEXT.encode("utf-8"), "mx100.md", "manual-md")

    sections = splitter.split(document)
    chunks = builder.build_chunks(document, sections)

    fault_chunks = [chunk for chunk in chunks if chunk.content_type == "fault_code"]
    safety_chunks = [chunk for chunk in chunks if chunk.content_type == "safety_rule"]

    assert fault_chunks
    assert fault_chunks[0].fault_code == "E03"
    assert safety_chunks


def test_chunk_id_is_stable() -> None:
    builder = ChunkMetadataBuilder()

    first = builder.build_chunk_id("doc-1", "Fault Codes", "fault_code", "E03", "E03 overheat")
    second = builder.build_chunk_id("doc-1", "Fault Codes", "fault_code", "E03", "E03 overheat")

    assert first == second


def test_manual_indexer_adds_chunks_to_retriever() -> None:
    indexer = ManualIndexer()

    result = indexer.index_document(
        MANUAL_TEXT.encode("utf-8"),
        filename="mx100.md",
        doc_id="manual-indexer",
        device_name="Compressor",
        device_model="MX100",
    )
    search_results = asyncio.run(indexer.retriever.search("E03 overheat", top_n_rerank=2))

    assert result.status == "indexed"
    assert result.chunks_count >= 3
    assert result.content_type_stats["fault_code"] >= 1
    assert search_results
    assert any(result.fault_code == "E03" for result in search_results)


def test_manual_index_api_returns_chunks_count() -> None:
    client = TestClient(app)

    upload_response = client.post(
        "/api/manual/upload",
        files={"file": ("api-manual.md", MANUAL_TEXT.encode("utf-8"), "text/markdown")},
    )
    assert upload_response.status_code == 200
    doc_id = upload_response.json()["doc_id"]

    index_response = client.post(
        "/api/manual/index",
        json={"doc_id": doc_id, "device_name": "Compressor", "device_model": "MX100"},
    )

    assert index_response.status_code == 200
    body = index_response.json()
    assert body["doc_id"] == doc_id
    assert body["chunks_count"] >= 3
    assert body["content_type_stats"]["fault_code"] >= 1


def test_manual_hybrid_search_finds_indexed_chunk() -> None:
    indexer = get_manual_indexer()
    indexer.index_document(
        b"# Fault Codes\nP001 pump cavitation detected near inlet valve.",
        filename="pump.md",
        doc_id="manual-pump-p001",
        device_name="Pump",
        device_model="PX900",
    )
    executor = MCPToolExecutor(create_default_tool_registry())

    result = asyncio.run(
        executor.execute_tool(
            "manual_hybrid_search",
            {
                "query": "P001 cavitation",
                "device_model": "PX900",
                "top_k_bm25": 3,
                "top_k_dense": 3,
                "top_n_rerank": 2,
            },
        )
    )

    assert result.status == "success"
    assert result.data is not None
    assert result.data["results"]
    assert result.data["results"][0]["doc_id"] == "manual-pump-p001"


def test_document_chunk_schema_fields_match_retrieval_metadata() -> None:
    document = ManualDocument(
        doc_id="doc-schema",
        filename="schema.md",
        text="E03 overheat",
        device_name="Compressor",
        device_model="MX100",
    )

    assert document.doc_id == "doc-schema"
    assert document.device_model == "MX100"
