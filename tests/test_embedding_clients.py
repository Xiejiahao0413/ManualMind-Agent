import importlib

from app.embeddings import MockEmbeddingClient, OpenAIEmbeddingClient
from app.vectorstore.factory import create_embedding_client


def test_mock_embedding_is_deterministic() -> None:
    client = MockEmbeddingClient(dimension=16)

    first = client.embed_text("A100 E03 temperature sensor")
    second = client.embed_text("A100 E03 temperature sensor")

    assert first == second
    assert len(first) == 16


def test_mock_embedding_texts_match_single_calls() -> None:
    client = MockEmbeddingClient(dimension=12)
    texts = ["E03 overheat", "B200 pressure"]

    assert client.embed_texts(texts) == [client.embed_text(text) for text in texts]


def test_openai_embedding_client_import_does_not_call_api() -> None:
    module = importlib.import_module("app.embeddings.openai_embedding")

    assert hasattr(module, "OpenAIEmbeddingClient")
    client = OpenAIEmbeddingClient(api_key="not-a-real-key", model="text-embedding-3-small")
    assert client.provider == "openai"


def test_openai_embedding_provider_without_key_falls_back_to_mock() -> None:
    client, fallback_used, reason = create_embedding_client(
        {"MANUALMIND_EMBEDDING_PROVIDER": "openai"}
    )

    assert isinstance(client, MockEmbeddingClient)
    assert fallback_used is True
    assert reason == "openai_api_key_missing"
