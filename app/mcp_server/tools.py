from uuid import uuid4

from app.mcp_server.registry import MCPTool, ToolRegistry
from app.mcp_server.schemas import (
    FaultCodeLookupRequest,
    FaultCodeLookupResponse,
    HandoffTicketCreateRequest,
    HandoffTicketCreateResponse,
    ManualHybridSearchRequest,
    ManualHybridSearchResponse,
    ParameterLookupRequest,
    ParameterLookupResponse,
    SafetyRuleSearchRequest,
    SafetyRuleSearchResponse,
    SourceTraceItem,
    SourceTraceRequest,
    SourceTraceResponse,
)
from app.retrieval import HybridRetrieverImpl
from app.schemas.retrieval import DocumentChunk


DEMO_CHUNKS: list[DocumentChunk] = [
    DocumentChunk(
        chunk_id="demo-e03-mx100",
        doc_id="manual-mx100",
        device_name="Compressor",
        device_model="MX100",
        section_title="Fault code table",
        page=10,
        content_type="fault_code",
        fault_code="E03",
        source_file="mx100_manual.pdf",
        text="E03 indicates motor overheat. Check cooling fan, temperature sensor, and ventilation.",
    ),
    DocumentChunk(
        chunk_id="demo-temp-mx100",
        doc_id="manual-mx100",
        device_name="Compressor",
        device_model="MX100",
        section_title="Parameter table",
        page=22,
        content_type="parameter",
        source_file="mx100_manual.pdf",
        text="Temperature should remain between 0 and 80 C during normal operation.",
        metadata={"parameter_name": "temperature", "min": 0, "max": 80},
    ),
    DocumentChunk(
        chunk_id="demo-pressure-mx100",
        doc_id="manual-mx100",
        device_name="Compressor",
        device_model="MX100",
        section_title="Pressure parameter",
        page=23,
        content_type="parameter",
        source_file="mx100_manual.pdf",
        text="Pressure threshold should remain below 0.8 MPa during startup.",
        metadata={"parameter_name": "pressure", "min": 0, "max": 0.8},
    ),
    DocumentChunk(
        chunk_id="demo-safety-mx100",
        doc_id="manual-mx100",
        device_name="Compressor",
        device_model="MX100",
        section_title="Safety rules",
        page=4,
        content_type="safety_rule",
        source_file="mx100_manual.pdf",
        text="Before high-risk operation, isolate power, release pressure, and verify lockout.",
        metadata={"operation": "high_risk_operation", "risk_level": "high"},
    ),
]


PARAMETER_DATA = {
    "temperature": {"standard_range": "0-80 C", "min": 0.0, "max": 80.0, "source_refs": ["mx100_manual.pdf:22"]},
    "温度": {"standard_range": "0-80 C", "min": 0.0, "max": 80.0, "source_refs": ["mx100_manual.pdf:22"]},
    "pressure": {"standard_range": "0-0.8 MPa", "min": 0.0, "max": 0.8, "source_refs": ["mx100_manual.pdf:23"]},
    "压力": {"standard_range": "0-0.8 MPa", "min": 0.0, "max": 0.8, "source_refs": ["mx100_manual.pdf:23"]},
    "voltage": {"standard_range": "380-400 V", "min": 380.0, "max": 400.0, "source_refs": ["mx100_manual.pdf:24"]},
    "电压": {"standard_range": "380-400 V", "min": 380.0, "max": 400.0, "source_refs": ["mx100_manual.pdf:24"]},
    "maintenance_cycle": {"standard_range": "30 days", "min": 0.0, "max": 30.0, "source_refs": ["mx100_manual.pdf:30"]},
    "维护周期": {"standard_range": "30 days", "min": 0.0, "max": 30.0, "source_refs": ["mx100_manual.pdf:30"]},
}


def build_demo_hybrid_retriever() -> HybridRetrieverImpl:
    retriever = HybridRetrieverImpl()
    retriever.add_documents(DEMO_CHUNKS)
    return retriever


async def manual_hybrid_search(arguments: dict) -> ManualHybridSearchResponse:
    request = ManualHybridSearchRequest.model_validate(arguments)
    metadata_filter: dict[str, object] = {}
    if request.device_name:
        metadata_filter["device_name"] = request.device_name
    if request.device_model:
        metadata_filter["device_model"] = request.device_model
    if request.content_types:
        metadata_filter["content_type"] = request.content_types

    results = await build_demo_hybrid_retriever().search(
        request.query,
        metadata_filter=metadata_filter or None,
        top_k_bm25=request.top_k_bm25,
        top_k_dense=request.top_k_dense,
        top_n_rerank=request.top_n_rerank,
    )
    source_refs = sorted({source for result in results for source in result.source_refs})
    return ManualHybridSearchResponse(results=results, source_refs=source_refs)


async def fault_code_lookup(arguments: dict) -> FaultCodeLookupResponse:
    request = FaultCodeLookupRequest.model_validate(arguments)
    fault_code = request.fault_code.upper()
    for chunk in DEMO_CHUNKS:
        if chunk.fault_code != fault_code:
            continue
        if request.device_model and chunk.device_model != request.device_model:
            continue
        return FaultCodeLookupResponse(
            status="found",
            fault_code=fault_code,
            device_model=chunk.device_model,
            description=chunk.text,
            source_refs=[f"{chunk.source_file}:{chunk.page}"] if chunk.source_file else [],
        )
    return FaultCodeLookupResponse(
        status="not_found",
        fault_code=fault_code,
        device_model=request.device_model,
    )


async def parameter_lookup(arguments: dict) -> ParameterLookupResponse:
    request = ParameterLookupRequest.model_validate(arguments)
    key = request.parameter_name.lower()
    data = PARAMETER_DATA.get(key) or PARAMETER_DATA.get(request.parameter_name)
    if data is None:
        return ParameterLookupResponse(status="not_found", parameter_name=request.parameter_name)

    is_abnormal = None
    if request.observed_value is not None:
        is_abnormal = not (float(data["min"]) <= request.observed_value <= float(data["max"]))

    return ParameterLookupResponse(
        status="found",
        parameter_name=request.parameter_name,
        standard_range=str(data["standard_range"]),
        observed_value=request.observed_value,
        is_abnormal=is_abnormal,
        source_refs=list(data["source_refs"]),
    )


async def safety_rule_search(arguments: dict) -> SafetyRuleSearchResponse:
    request = SafetyRuleSearchRequest.model_validate(arguments)
    rules: list[str] = []
    source_refs: list[str] = []
    for chunk in DEMO_CHUNKS:
        if chunk.content_type != "safety_rule":
            continue
        if request.device_model and chunk.device_model != request.device_model:
            continue
        if request.risk_level == "high" or "拆卸" in request.operation or "高压" in request.operation:
            rules.append(chunk.text)
            if chunk.source_file:
                source_refs.append(f"{chunk.source_file}:{chunk.page}")

    return SafetyRuleSearchResponse(
        safety_rules=rules,
        source_refs=sorted(set(source_refs)),
        risk_level=request.risk_level,
    )


async def source_trace(arguments: dict) -> SourceTraceResponse:
    request = SourceTraceRequest.model_validate(arguments)
    requested_ids = set(request.chunk_ids)
    sources = [
        SourceTraceItem(
            chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
            page=chunk.page,
            section_title=chunk.section_title,
            content_type=chunk.content_type,
        )
        for chunk in DEMO_CHUNKS
        if chunk.chunk_id in requested_ids
    ]
    return SourceTraceResponse(sources=sources)


async def handoff_ticket_create(arguments: dict) -> HandoffTicketCreateResponse:
    request = HandoffTicketCreateRequest.model_validate(arguments)
    reason = request.payload.handoff_reason or request.payload.reason
    return HandoffTicketCreateResponse(
        handoff_id=f"handoff-{uuid4().hex[:12]}",
        status="created",
        handoff_reason=reason,
        risk_level=request.payload.risk_level,
    )


def create_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        MCPTool(
            name="manual_hybrid_search",
            description="Search manual chunks using BM25, dense retrieval, merge, and rerank.",
            input_schema=ManualHybridSearchRequest,
            execute=manual_hybrid_search,
        )
    )
    registry.register(
        MCPTool(
            name="fault_code_lookup",
            description="Look up fault code information by code and optional device model.",
            input_schema=FaultCodeLookupRequest,
            execute=fault_code_lookup,
        )
    )
    registry.register(
        MCPTool(
            name="parameter_lookup",
            description="Look up operational parameter ranges and abnormality.",
            input_schema=ParameterLookupRequest,
            execute=parameter_lookup,
        )
    )
    registry.register(
        MCPTool(
            name="safety_rule_search",
            description="Search safety rules by operation, risk level, and optional device model.",
            input_schema=SafetyRuleSearchRequest,
            execute=safety_rule_search,
        )
    )
    registry.register(
        MCPTool(
            name="source_trace",
            description="Return source metadata for chunk ids.",
            input_schema=SourceTraceRequest,
            execute=source_trace,
        )
    )
    registry.register(
        MCPTool(
            name="handoff_ticket_create",
            description="Create a structured human handoff ticket.",
            input_schema=HandoffTicketCreateRequest,
            execute=handoff_ticket_create,
        )
    )
    return registry
