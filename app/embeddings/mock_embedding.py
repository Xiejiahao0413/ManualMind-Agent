import hashlib
import re
from collections import Counter
from math import sqrt

from app.embeddings.base import EmbeddingClient


TOKEN_PATTERN = re.compile(r"[A-Za-z]+\d+|\d+[A-Za-z]+|[A-Za-z0-9_-]+|[\u4e00-\u9fff]")


class MockEmbeddingClient(EmbeddingClient):
    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider(self) -> str:
        return "mock"

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token, count in Counter(_tokens(text)).items():
            index = _stable_index(token, self.dimension)
            vector[index] += float(count)
        norm = sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def _tokens(text: str) -> list[str]:
    normalized = text.lower()
    tokens = TOKEN_PATTERN.findall(normalized)
    tokens.extend(normalized[index : index + 2] for index in range(max(0, len(normalized) - 1)))
    return [token for token in tokens if token.strip()]


def _stable_index(token: str, dimension: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dimension
