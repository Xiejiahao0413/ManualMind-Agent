import asyncio

from app.mcp_server import MCPTool, MCPToolExecutor, ToolRegistry, create_default_tool_registry
from app.mcp_server.schemas import ManualHybridSearchRequest
from app.schemas.handoff import HandoffPayload


async def broken_tool(arguments: dict) -> dict:
    raise RuntimeError("boom")


def run_tool(tool_name: str, arguments: dict):
    registry = create_default_tool_registry()
    executor = MCPToolExecutor(registry)
    return asyncio.run(executor.execute_tool(tool_name, arguments))


def test_tool_registry_registers_and_gets_tool() -> None:
    registry = ToolRegistry()
    tool = MCPTool(
        name="manual_hybrid_search",
        description="demo",
        input_schema=ManualHybridSearchRequest,
        execute=broken_tool,
    )

    registry.register(tool)

    assert registry.get("manual_hybrid_search") == tool
    assert registry.list_tools() == [tool]


def test_execute_tool_runs_manual_hybrid_search() -> None:
    result = run_tool(
        "manual_hybrid_search",
        {
            "query": "E03 motor overheat",
            "device_model": "MX100",
            "top_k_bm25": 3,
            "top_k_dense": 3,
            "top_n_rerank": 2,
        },
    )

    assert result.status == "success"
    assert result.data is not None
    assert result.data["retrieval_mode"] == "hybrid"
    assert result.data["results"]


def test_fault_code_lookup_finds_e03() -> None:
    result = run_tool(
        "fault_code_lookup",
        {
            "fault_code": "E03",
            "device_model": "MX100",
        },
    )

    assert result.status == "success"
    assert result.data is not None
    assert result.data["status"] == "found"
    assert "overheat" in result.data["description"]


def test_parameter_lookup_flags_temperature_abnormal() -> None:
    result = run_tool(
        "parameter_lookup",
        {
            "parameter_name": "temperature",
            "device_model": "MX100",
            "observed_value": 95,
        },
    )

    assert result.status == "success"
    assert result.data is not None
    assert result.data["is_abnormal"] is True


def test_safety_rule_search_returns_rules() -> None:
    result = run_tool(
        "safety_rule_search",
        {
            "operation": "高压拆卸",
            "risk_level": "high",
            "device_model": "MX100",
        },
    )

    assert result.status == "success"
    assert result.data is not None
    assert result.data["safety_rules"]
    assert result.data["source_refs"]


def test_source_trace_returns_source_info() -> None:
    result = run_tool(
        "source_trace",
        {
            "chunk_ids": ["demo-e03-mx100"],
        },
    )

    assert result.status == "success"
    assert result.data is not None
    assert result.data["sources"][0]["source_file"] == "mx100_manual.pdf"
    assert result.data["sources"][0]["content_type"] == "fault_code"


def test_handoff_ticket_create_generates_handoff_id() -> None:
    payload = HandoffPayload(
        task_id="task-1",
        session_id="session-1",
        handoff_reason="high_risk_without_evidence",
        risk_level="high",
    )
    result = run_tool(
        "handoff_ticket_create",
        {
            "payload": payload.model_dump(),
        },
    )

    assert result.status == "success"
    assert result.data is not None
    assert result.data["handoff_id"].startswith("handoff-")
    assert result.data["status"] == "created"


def test_missing_tool_returns_tool_not_found() -> None:
    result = run_tool("missing_tool", {})

    assert result.status == "error"
    assert result.error_type == "tool_not_found"


def test_tool_execution_exception_is_structured() -> None:
    registry = ToolRegistry()
    registry.register(
        MCPTool(
            name="broken_tool",
            description="raises",
            input_schema=ManualHybridSearchRequest,
            execute=broken_tool,
        )
    )
    executor = MCPToolExecutor(registry)

    result = asyncio.run(
        executor.execute_tool(
            "broken_tool",
            {
                "query": "E03",
            },
        )
    )

    assert result.status == "error"
    assert result.error_type == "RuntimeError"
    assert result.error_message == "boom"
