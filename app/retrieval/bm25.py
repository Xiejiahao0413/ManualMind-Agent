import math
import re
from collections import Counter
from typing import Any

from app.retrieval.filters import matches_metadata_filter
from app.schemas.retrieval import DocumentChunk, RetrievalResult

TOKEN_PATTERN = re.compile(r"[A-Za-z]+\d+|\d+[A-Za-z]+|[A-Za-z0-9_-]+|[\u4e00-\u9fff]")

PROCEDURE_QUERY_TERMS = (
    "如何",
    "怎么",
    "怎样",
    "增加",
    "创建",
    "新建",
    "添加",
    "插入",
    "示教",
    "步骤",
    "操作",
    "create",
    "add",
    "insert",
    "teach",
    "step",
    "procedure",
)
PROCEDURE_TOPIC_TERMS = ("动作指令", "指令", "程序", "参数", "action instruction", "instruction")
PROCEDURE_ACTION_TERMS = (
    "创建",
    "新建",
    "增加",
    "添加",
    "插入",
    "示教",
    "步骤",
    "点击",
    "选择",
    "进入",
    "设置",
    "确认",
    "保存",
    "create",
    "add",
    "insert",
    "teach",
    "click",
    "select",
    "enter",
    "set",
    "confirm",
    "save",
)
DEFINITION_TERMS = ("定义", "构成", "是指", "如下信息构成", "包括以下信息", "consists of", "is defined")


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
            score *= procedure_score_multiplier(query, chunk)
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


def is_procedure_query(query: str) -> bool:
    lower_query = query.lower()
    return any(term.lower() in lower_query for term in PROCEDURE_QUERY_TERMS)


def procedure_score_multiplier(query: str, chunk: DocumentChunk) -> float:
    if not is_procedure_query(query):
        return 1.0

    combined = f"{chunk.section_title or ''}\n{chunk.text}".lower()
    has_topic = any(term.lower() in combined for term in PROCEDURE_TOPIC_TERMS)
    action_hits = sum(1 for term in PROCEDURE_ACTION_TERMS if term.lower() in combined)
    definition_hits = sum(1 for term in DEFINITION_TERMS if term.lower() in combined)
    has_steps = bool(re.search(r"(^|\n)\s*(?:\d+[.、)]|[（(]\d+[）)])", chunk.text))

    multiplier = 1.0
    if has_topic and action_hits:
        multiplier += 0.7
    if has_topic and action_hits >= 2:
        multiplier += 0.4
    if has_steps:
        multiplier += 0.35
    if definition_hits and action_hits == 0:
        multiplier *= 0.45
    elif definition_hits:
        multiplier *= 0.8
    return multiplier
