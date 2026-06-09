import asyncio

from app.retrieval import (
    HybridRetrieverImpl,
    InMemoryDenseRetriever,
    LocalBM25Retriever,
    MockEmbeddingProvider,
    MockReranker,
    merge_retrieval_results,
)
from app.schemas import DocumentChunk, RetrievalResult


def sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="chunk-e03-a",
            doc_id="manual-a",
            device_name="Compressor",
            device_model="MX100",
            section_title="Fault code table",
            page=10,
            content_type="fault_code",
            fault_code="E03",
            source_file="manual-a.pdf",
            text="E03 indicates motor overheat. Check cooling fan and temperature sensor.",
        ),
        DocumentChunk(
            chunk_id="chunk-pressure-a",
            doc_id="manual-a",
            device_name="Compressor",
            device_model="MX100",
            section_title="Parameter table",
            page=22,
            content_type="parameter",
            source_file="manual-a.pdf",
            text="Pressure threshold should remain below 0.8 MPa during startup.",
        ),
        DocumentChunk(
            chunk_id="chunk-e03-b",
            doc_id="manual-b",
            device_name="Pump",
            device_model="PX200",
            section_title="Fault code table",
            page=8,
            content_type="fault_code",
            fault_code="E03",
            source_file="manual-b.pdf",
            text="E03 on PX200 indicates inlet blockage.",
        ),
    ]


def test_bm25_hits_fault_code_e03() -> None:
    retriever = LocalBM25Retriever()
    retriever.add_documents(sample_chunks())

    results = asyncio.run(retriever.search("E03 fault", top_k=2))

    assert results
    assert results[0].fault_code == "E03"
    assert results[0].bm25_score is not None
    assert results[0].source == "bm25"


def test_dense_mock_returns_semantic_result() -> None:
    retriever = InMemoryDenseRetriever(MockEmbeddingProvider())
    retriever.add_documents(sample_chunks())

    results = asyncio.run(retriever.search("motor temperature overheat", top_k=1))

    assert results
    assert results[0].chunk_id == "chunk-e03-a"
    assert results[0].dense_score is not None


def test_hybrid_merges_bm25_and_dense_results() -> None:
    retriever = HybridRetrieverImpl(reranker=MockReranker())
    retriever.add_documents(sample_chunks())

    results = asyncio.run(
        retriever.search(
            "E03 motor overheat",
            top_k=3,
            top_k_bm25=3,
            top_k_dense=3,
            top_n_rerank=3,
        )
    )

    hybrid_hits = [result for result in results if result.chunk_id == "chunk-e03-a"]
    assert hybrid_hits
    assert hybrid_hits[0].source == "hybrid"
    assert hybrid_hits[0].bm25_score is not None
    assert hybrid_hits[0].dense_score is not None


def test_merge_deduplicates_same_chunk_id() -> None:
    chunks = sample_chunks()
    bm25_result = [
        RetrievalResult.from_chunk(
            chunks[0],
            score=2.0,
            source="bm25",
            bm25_score=2.0,
        )
    ]
    dense_result = [
        RetrievalResult.from_chunk(
            chunks[0],
            score=0.5,
            source="dense",
            dense_score=0.5,
        )
    ]

    merged = merge_retrieval_results(bm25_result, dense_result)

    assert len(merged) == 1
    assert merged[0].chunk_id == "chunk-e03-a"
    assert merged[0].source == "hybrid"


def test_metadata_filter_limits_device_model() -> None:
    retriever = LocalBM25Retriever()
    retriever.add_documents(sample_chunks())

    results = asyncio.run(
        retriever.search("E03", top_k=5, metadata_filter={"device_model": "PX200"})
    )

    assert results
    assert all(result.device_model == "PX200" for result in results)


def test_mock_reranker_returns_top_n() -> None:
    retriever = HybridRetrieverImpl(reranker=MockReranker())
    retriever.add_documents(sample_chunks())

    results = asyncio.run(retriever.search("E03 pressure motor", top_n_rerank=2))

    assert len(results) == 2
    assert all(result.rerank_score is not None for result in results)


def test_rerank_failure_falls_back_to_fused_score() -> None:
    retriever = HybridRetrieverImpl(reranker=MockReranker(should_fail=True))
    retriever.add_documents(sample_chunks())

    results = asyncio.run(
        retriever.search("E03 motor overheat", top_k=2, top_k_bm25=3, top_k_dense=3)
    )

    assert len(results) == 2
    assert results[0].score >= results[1].score
