from typing import Any

from app.memory.base import CaseMemory, SafetyMemory, SessionMemory, TaskMemory, ToolMemory
from app.schemas.diagnosis import DiagnosisState
from app.schemas.tools import ToolCallRecord


class InMemoryMemoryManager(SessionMemory, TaskMemory, ToolMemory, SafetyMemory, CaseMemory):
    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, Any]]] = {}
        self._tasks: dict[str, DiagnosisState] = {}
        self._tool_calls: dict[str, list[ToolCallRecord]] = {}
        self._safety_events: dict[str, list[dict[str, Any]]] = {}
        self._cases: list[dict[str, Any]] = []

    async def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._sessions.get(session_id, []))

    async def append_message(self, session_id: str, message: dict[str, Any]) -> None:
        self._sessions.setdefault(session_id, []).append(message)

    async def get_task(self, task_id: str) -> DiagnosisState | None:
        return self._tasks.get(task_id)

    async def save_task(self, state: DiagnosisState) -> None:
        self._tasks[state.task_id] = state

    async def list_tool_calls(self, task_id: str) -> list[ToolCallRecord]:
        return list(self._tool_calls.get(task_id, []))

    async def append_tool_call(self, record: ToolCallRecord) -> None:
        self._tool_calls.setdefault(record.task_id, []).append(record)

    async def get_tool_call_by_signature(
        self,
        task_id: str,
        args_signature: str,
    ) -> ToolCallRecord | None:
        for record in self._tool_calls.get(task_id, []):
            if record.args_signature == args_signature:
                return record
        return None

    async def list_events(self, task_id: str) -> list[dict[str, Any]]:
        return list(self._safety_events.get(task_id, []))

    async def append_event(self, task_id: str, event: dict[str, Any]) -> None:
        self._safety_events.setdefault(task_id, []).append(event)

    async def append_case(self, case: dict[str, Any]) -> None:
        self._cases.append(case)

    async def search_cases(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        normalized_query = query.lower()
        matches = [
            case
            for case in self._cases
            if normalized_query in " ".join(str(value).lower() for value in case.values())
        ]
        return matches[:top_k]
