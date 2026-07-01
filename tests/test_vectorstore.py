from app.embeddings import MockEmbeddingClient
from app.retrieval.vector_dense import VectorDenseRetriever
from app.schemas.retrieval import DocumentChunk
from app.vectorstore import InMemoryVectorStore, VectorStoreError


def sample_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="a100-e03",
            doc_id="a100",
            text="A100 E03 temperature sensor abnormal. Check cooling fan.",
            device_model="A100",
            fault_code="E03",
            content_type="fault_code",
            source_file="a100_manual.md",
            page=3,
        ),
        DocumentChunk(
            chunk_id="b200-f01",
            doc_id="b200",
            text="B200 F01 hydraulic pressure insufficient.",
            device_model="B200",
            fault_code="F01",
            content_type="fault_code",
            source_file="b200_manual.md",
            page=5,
        ),
    ]


def test_in_memory_vectorstore_upsert_and_search() -> None:
    embedding = MockEmbeddingClient(dimension=32)
    store = InMemoryVectorStore()
    chunks = sample_chunks()
    store.upsert_chunks(chunks, embedding.embed_texts([chunk.text for chunk in chunks]))

    results = store.search(embedding.embed_text("temperature sensor E03"), top_k=1)

    assert results
    assert results[0].chunk_id == "a100-e03"
    assert results[0].source_refs == ["a100_manual.md"]
    assert results[0].page == 3


def test_in_memory_vectorstore_metadata_filter() -> None:
    embedding = MockEmbeddingClient(dimension=32)
    store = InMemoryVectorStore()
    chunks = sample_chunks()
    store.upsert_chunks(chunks, embedding.embed_texts([chunk.text for chunk in chunks]))

    results = store.search(
        embedding.embed_text("pressure fault"),
        top_k=5,
        filters={"device_model": "B200", "fault_code": "F01", "source_file": "b200_manual.md"},
    )

    assert results
    assert all(result.device_model == "B200" for result in results)
    assert all(result.fault_code == "F01" for result in results)
    assert all(result.source_file == "b200_manual.md" for result in results)


def test_in_memory_vectorstore_upsert_replaces_existing_chunk() -> None:
    embedding = MockEmbeddingClient(dimension=16)
    store = InMemoryVectorStore()
    chunk = sample_chunks()[0]
    store.upsert_chunks([chunk], embedding.embed_texts([chunk.text]))
    updated = chunk.model_copy(update={"text": "A100 E03 updated temperature sensor guidance."})
    store.upsert_chunks([updated], embedding.embed_texts([updated.text]))

    results = store.search(embedding.embed_text("updated guidance"), top_k=5)

    assert len([result for result in results if result.chunk_id == "a100-e03"]) == 1
    assert results[0].text == updated.text


def test_in_memory_vectorstore_delete_doc() -> None:
    embedding = MockEmbeddingClient(dimension=16)
    store = InMemoryVectorStore()
    chunks = sample_chunks()
    store.upsert_chunks(chunks, embedding.embed_texts([chunk.text for chunk in chunks]))

    store.delete_doc("a100")
    results = store.search(embedding.embed_text("temperature pressure fault"), top_k=5)

    assert results
    assert all(result.doc_id != "a100" for result in results)


def test_vector_dense_retriever_uses_doc_id_filter() -> None:
    embedding = MockEmbeddingClient(dimension=32)
    retriever = VectorDenseRetriever(embedding_client=embedding, vector_store=InMemoryVectorStore())
    retriever.add_documents(sample_chunks())

    import asyncio

    results = asyncio.run(
        retriever.search("temperature pressure fault", top_k=5, metadata_filter={"doc_id": ["b200"]})
    )

    assert results
    assert all(result.doc_id == "b200" for result in results)


def test_vector_dense_retriever_falls_back_to_memory_on_milvus_error() -> None:
    embedding = MockEmbeddingClient(dimension=32)
    retriever = VectorDenseRetriever(
        embedding_client=embedding,
        vector_store=FailingVectorStore(),
        fallback_store=InMemoryVectorStore(),
    )
    retriever.add_documents(sample_chunks())

    import asyncio

    results = asyncio.run(retriever.search("temperature sensor E03", top_k=1))

    assert results
    assert results[0].doc_id == "a100"


class FailingVectorStore(InMemoryVectorStore):
    backend = "milvus"

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        raise VectorStoreError("milvus_upsert_failed")

    def search(self, query_embedding, top_k=5, filters=None):
        raise VectorStoreError("milvus_search_failed")
