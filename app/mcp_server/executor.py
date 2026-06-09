from time import perf_counter

from pydantic import ValidationError

from app.mcp_server.registry import ToolRegistry
from app.mcp_server.schemas import ToolExecutionResult


class MCPToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def execute_tool(self, tool_name: str, arguments: dict) -> ToolExecutionResult:
        start = perf_counter()
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolExecutionResult(
                tool_name=tool_name,
                status="error",
                error_type="tool_not_found",
                error_message=f"Tool not found: {tool_name}",
                latency_ms=self._latency_ms(start),
            )

        try:
            validated = tool.input_schema.model_validate(arguments)
            result = await tool.execute(validated.model_dump())
            data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
            return ToolExecutionResult(
                tool_name=tool_name,
                status="success",
                data=data,
                latency_ms=self._latency_ms(start),
            )
        except ValidationError as exc:
            return ToolExecutionResult(
                tool_name=tool_name,
                status="error",
                error_type="validation_error",
                error_message=str(exc),
                latency_ms=self._latency_ms(start),
            )
        except Exception as exc:
            return ToolExecutionResult(
                tool_name=tool_name,
                status="error",
                error_type=exc.__class__.__name__,
                error_message=str(exc),
                latency_ms=self._latency_ms(start),
            )

    def _latency_ms(self, start: float) -> float:
        return (perf_counter() - start) * 1000
