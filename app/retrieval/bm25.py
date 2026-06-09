import math
import re
from collections import Counter
from typing import Any

from app.retrieval.filters import matches_metadata_filter
from app.schemas.retrieval import DocumentChunk, RetrievalResult

TOKEN_PATTERN = re.compile(r"[A-Za-z]+\d+|\d+[A-Za-z]+|[A-Za-z0-9_-]+|[\u4e00-\u9fff]")


class LocalBM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[DocumentChunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._avg_doc_len = 0.0

    def add_documents(self, chunks: list[DocumentChunk]) -> None:
        self._chunks.extend(chunks)
        self._rebuild_index()

    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        query_tokens = tokenize_text(query)
        scored: list[RetrievalResult] = []
        for chunk, tokens in zip(self._chunks, self._doc_tokens, strict=True):
            if not matches_metadata_filter(chunk, metadata_filter):
                continue
            score = self._score(query_tokens, tokens)
            if score <= 0:
                continue
            scored.append(
                RetrievalResult.from_chunk(
                    chunk,
                    score=score,
                    source="bm25",
                    bm25_score=score,
                )
            )
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]

    def _rebuild_index(self) -> None:
        self._doc_tokens = [tokenize_chunk(chunk) for chunk in self._chunks]
        self._doc_freq = Counter()
        for tokens in self._doc_tokens:
            self._doc_freq.update(set(tokens))
        total_len = sum(len(tokens) for tokens in self._doc_tokens)
        self._avg_doc_len = total_len / len(self._doc_tokens) if self._doc_tokens else 0.0

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        token_counts = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        score = 0.0
        for token in query_tokens:
            freq = token_counts.get(token, 0)
            if freq == 0:
                continue
            idf = self._idf(token)
            denominator = freq + self.k1 * (
                1 - self.b + self.b * doc_len / (self._avg_doc_len or 1.0)
            )
            score += idf * freq * (self.k1 + 1) / denominator
        return score

    def _idf(self, token: str) -> float:
        doc_count = len(self._doc_tokens)
        freq = self._doc_freq.get(token, 0)
        return math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))


def tokenize_chunk(chunk: DocumentChunk) -> list[str]:
    weighted_text = " ".join(
        value
        for value in [
            chunk.fault_code,
            chunk.fault_code,
            chunk.device_model,
            chunk.device_model,
            chunk.section_title,
            chunk.content_type,
            chunk.text,
        ]
        if value
    )
    return tokenize_text(weighted_text)


def tokenize_text(text: str) -> list[str]:
    tokens = [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
    compact = text.lower()
    tokens.extend(compact[index : index + 2] for index in range(max(0, len(compact) - 1)))
    return [token for token in tokens if token.strip()]
