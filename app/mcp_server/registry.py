from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

ToolExecute = Callable[[dict[str, Any]], Awaitable[BaseModel | dict[str, Any]]]


@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str
    input_schema: type[BaseModel]
    execute: ToolExecute


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, MCPTool] = {}

    def register(self, tool: MCPTool) -> None:
        self._tools[tool.name] = tool

    def get(self, tool_name: str) -> MCPTool | None:
        return self._tools.get(tool_name)

    def list_tools(self) -> list[MCPTool]:
        return list(self._tools.values())
