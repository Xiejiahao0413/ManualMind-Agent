from app.embeddings.base import EmbeddingClient


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        dimension: int = 1536,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._dimension = dimension
        self.timeout_seconds = timeout_seconds

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider(self) -> str:
        return "openai"

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai_sdk_not_installed") from exc

        try:
            client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
            response = client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as exc:
            raise RuntimeError("openai_embedding_error") from exc
