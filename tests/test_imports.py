from app.main import app
from app.agents import DiagnosisWorkflow, create_diagnosis_graph
from app.memory import (
    CaseMemory,
    InMemoryMemoryManager,
    SafetyMemory,
    SessionMemory,
    TaskMemory,
    ToolMemory,
)
from app.retrieval import (
    BGEReranker,
    BM25Retriever,
    HybridRetriever,
    HybridRetrieverImpl,
    InMemoryDenseRetriever,
    LocalBM25Retriever,
    MilvusDenseRetriever,
    MockEmbeddingProvider,
    MockReranker,
)
from app.schemas import (
    DiagnosisRequest,
    DiagnosisResponse,
    DiagnosisState,
    DocumentChunk,
    HandoffPayload,
    RetrievalResult,
    ToolCallRecord,
)
from app.tools import (
    CircuitBreaker,
    RetryFallbackManager,
    ToolCallGuard,
    ToolIntent,
    ToolResultValidator,
    ToolRouter,
)


def test_app_imports() -> None:
    assert app.title == "ManualMind-Agent"
    assert DiagnosisWorkflow
    assert create_diagnosis_graph


def test_schema_imports() -> None:
    assert DiagnosisRequest
    assert DiagnosisResponse
    assert DiagnosisState
    assert DocumentChunk
    assert HandoffPayload
    assert RetrievalResult
    assert ToolCallRecord


def test_interface_imports() -> None:
    assert SessionMemory
    assert TaskMemory
    assert ToolMemory
    assert SafetyMemory
    assert CaseMemory
    assert InMemoryMemoryManager
    assert BM25Retriever
    assert MilvusDenseRetriever
    assert HybridRetriever
    assert BGEReranker
    assert LocalBM25Retriever
    assert InMemoryDenseRetriever
    assert HybridRetrieverImpl
    assert MockEmbeddingProvider
    assert MockReranker
    assert RetryFallbackManager
    assert ToolResultValidator
    assert CircuitBreaker


def test_tool_guard_duplicate_signature() -> None:
    guard = ToolCallGuard(tool_whitelist={"manual_hybrid_search"})
    router = ToolRouter(guard)
    intent = ToolIntent(
        tool_name="manual_hybrid_search",
        args={"query": "E01"},
        required_args={"query"},
    )

    first = router.validate_intent(intent)
    assert first.allowed is True
    assert first.args_signature is not None

    guard.remember_signature(first.args_signature)
    second = router.validate_intent(intent)
    assert second.allowed is False
    assert second.reason == "duplicate_tool_call_signature"
