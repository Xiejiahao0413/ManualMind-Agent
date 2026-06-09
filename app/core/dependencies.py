from app.memory import InMemoryMemoryManager
from app.tracing import InMemoryTraceManager


memory_manager = InMemoryMemoryManager()
trace_manager = InMemoryTraceManager()


def get_memory_manager() -> InMemoryMemoryManager:
    return memory_manager


def get_trace_manager() -> InMemoryTraceManager:
    return trace_manager
