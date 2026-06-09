from abc import ABC, abstractmethod
from collections import Counter
from math import sqrt


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class MockEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token, count in Counter(_tokenize_for_embedding(text)).items():
            vector[hash(token) % self.dimensions] += float(count)
        norm = sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=False))


def _tokenize_for_embedding(text: str) -> list[str]:
    normalized = text.lower()
    tokens: list[str] = []
    current = ""
    for char in normalized:
        if char.isalnum():
            current += char
        else:
            if current:
                tokens.append(current)
                current = ""
            if "\u4e00" <= char <= "\u9fff":
                tokens.append(char)
    if current:
        tokens.append(current)
    tokens.extend(normalized[index : index + 2] for index in range(max(0, len(normalized) - 1)))
    return [token for token in tokens if token.strip()]
