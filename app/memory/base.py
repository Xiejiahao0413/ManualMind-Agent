from abc import ABC, abstractmethod
from typing import Any

from app.schemas.diagnosis import DiagnosisState
from app.schemas.tools import ToolCallRecord


class SessionMemory(ABC):
    @abstractmethod
    async def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def append_message(self, session_id: str, message: dict[str, Any]) -> None:
        raise NotImplementedError


class TaskMemory(ABC):
    @abstractmethod
    async def get_task(self, task_id: str) -> DiagnosisState | None:
        raise NotImplementedError

    @abstractmethod
    async def save_task(self, state: DiagnosisState) -> None:
        raise NotImplementedError


class ToolMemory(ABC):
    @abstractmethod
    async def list_tool_calls(self, task_id: str) -> list[ToolCallRecord]:
        raise NotImplementedError

    @abstractmethod
    async def append_tool_call(self, record: ToolCallRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_tool_call_by_signature(
        self,
        task_id: str,
        args_signature: str,
    ) -> ToolCallRecord | None:
        raise NotImplementedError


class SafetyMemory(ABC):
    @abstractmethod
    async def list_events(self, task_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def append_event(self, task_id: str, event: dict[str, Any]) -> None:
        raise NotImplementedError


class CaseMemory(ABC):
    @abstractmethod
    async def append_case(self, case: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def search_cases(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        raise NotImplementedError
