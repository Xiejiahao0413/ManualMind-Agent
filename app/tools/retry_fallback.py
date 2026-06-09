from enum import StrEnum


class ToolErrorType(StrEnum):
    MILVUS_TIMEOUT = "milvus_timeout"
    BM25_ERROR = "bm25_error"
    RERANK_ERROR = "rerank_error"
    LLM_ERROR = "llm_error"
    MISSING_DEVICE_MODEL = "missing_device_model"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    HIGH_RISK_WITHOUT_EVIDENCE = "high_risk_without_evidence"


class FallbackAction(StrEnum):
    BM25_ONLY = "bm25_only"
    DENSE_ONLY = "dense_only"
    FUSED_SCORE = "fused_score"
    TEMPLATE_REPORT = "template_report"
    CLARIFICATION = "clarification"
    HUMAN_HANDOFF = "human_handoff"
    CIRCUIT_BREAKER_HANDOFF = "circuit_breaker_handoff"
    RETRY = "retry"


FALLBACK_RULES: dict[ToolErrorType, FallbackAction] = {
    ToolErrorType.MILVUS_TIMEOUT: FallbackAction.BM25_ONLY,
    ToolErrorType.BM25_ERROR: FallbackAction.DENSE_ONLY,
    ToolErrorType.RERANK_ERROR: FallbackAction.FUSED_SCORE,
    ToolErrorType.LLM_ERROR: FallbackAction.TEMPLATE_REPORT,
    ToolErrorType.MISSING_DEVICE_MODEL: FallbackAction.CLARIFICATION,
    ToolErrorType.INSUFFICIENT_EVIDENCE: FallbackAction.CLARIFICATION,
    ToolErrorType.HIGH_RISK_WITHOUT_EVIDENCE: FallbackAction.HUMAN_HANDOFF,
}


class RetryFallbackManager:
    def __init__(self, max_retry: int = 3) -> None:
        self.max_retry = max_retry

    def decide(self, error_type: ToolErrorType | str, retry_count: int = 0) -> FallbackAction:
        if retry_count >= self.max_retry:
            return FallbackAction.CIRCUIT_BREAKER_HANDOFF

        normalized_error = ToolErrorType(error_type)
        return FALLBACK_RULES.get(normalized_error, FallbackAction.RETRY)
