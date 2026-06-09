from app.memory import InMemoryMemoryManager


memory_manager = InMemoryMemoryManager()


def get_memory_manager() -> InMemoryMemoryManager:
    return memory_manager
