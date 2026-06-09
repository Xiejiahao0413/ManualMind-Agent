from app.ingestion import InMemoryManualStore, ManualIndexer
from app.memory import InMemoryMemoryManager
from app.tracing import InMemoryTraceManager


memory_manager = InMemoryMemoryManager()
trace_manager = InMemoryTraceManager()
manual_store = InMemoryManualStore()
manual_indexer = ManualIndexer()


def get_memory_manager() -> InMemoryMemoryManager:
    return memory_manager


def get_trace_manager() -> InMemoryTraceManager:
    return trace_manager


def get_manual_store() -> InMemoryManualStore:
    return manual_store


def get_manual_indexer() -> ManualIndexer:
    return manual_indexer
