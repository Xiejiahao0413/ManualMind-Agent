from uuid import uuid4
import re

from app.core.dependencies import get_manual_indexer
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

PARAMETER_ALIASES = {
    "temperature": ("temperature", "温度"),
    "pressure": ("pressure", "压力"),
    "voltage": ("voltage", "电压"),
    "maintenance_cycle": ("maintenance", "maintenance_cycle", "维护周期", "保养周期"),
}

MANUAL_QA_QUERY_TERMS = ("如何", "怎么", "怎样", "增加", "创建", "新建", "添加", "插入", "示教", "动作指令", "指令")
ACTION_INSTRUCTION_EXPANSIONS = (
    "创建动作指令",
    "新增动作指令",
    "添加动作指令",
    "插入动作指令",
    "示教动作指令",
    "动作指令的创建",
    "操作步骤",
    "设置",
    "确认",
    "保存",
)


def build_demo_hybrid_retriever() -> HybridRetrieverImpl:
    retriever = HybridRetrieverImpl()
    retriever.add_documents(DEMO_CHUNKS)
    return retriever


def get_indexed_chunks() -> list[DocumentChunk]:
    return get_manual_indexer().list_chunks()


def get_scoped_indexed_chunks(doc_ids: list[str] | None = None) -> list[DocumentChunk]:
    return get_manual_indexer().list_chunks(doc_ids=doc_ids)


def get_searchable_chunks(doc_ids: list[str] | None = None) -> list[DocumentChunk]:
    if doc_ids:
        return get_scoped_indexed_chunks(doc_ids)
    return [*get_indexed_chunks(), *DEMO_CHUNKS]


def matches_device_model(chunk: DocumentChunk, device_model: str | None) -> bool:
    return not device_model or chunk.device_model == device_model


async def manual_hybrid_search(arguments: dict) -> ManualHybridSearchResponse:
    request = ManualHybridSearchRequest.model_validate(arguments)
    metadata_filter: dict[str, object] = {}
    if request.device_name:
        metadata_filter["device_name"] = request.device_name
    if request.device_model:
        metadata_filter["device_model"] = request.device_model
    if request.doc_ids:
        metadata_filter["doc_id"] = request.doc_ids
    if request.content_types:
        metadata_filter["content_type"] = request.content_types

    indexer = get_manual_indexer()
    retriever = indexer.retriever if indexer.has_chunks() else build_demo_hybrid_retriever()
    search_query = expand_manual_qa_query(request.query)
    results = await retriever.search(
        search_query,
        metadata_filter=metadata_filter or None,
        top_k_bm25=request.top_k_bm25,
        top_k_dense=request.top_k_dense,
        top_n_rerank=request.top_n_rerank,
    )
    if not results and indexer.has_chunks() and not request.doc_ids:
        results = await build_demo_hybrid_retriever().search(
            search_query,
            metadata_filter=metadata_filter or None,
            top_k_bm25=request.top_k_bm25,
            top_k_dense=request.top_k_dense,
            top_n_rerank=request.top_n_rerank,
        )
    normalized_results = [
        result.model_copy(update={"source_refs": source_refs_from_result(result)})
        for result in results
    ]
    source_refs = normalize_source_refs([source for result in normalized_results for source in result.source_refs])
    return ManualHybridSearchResponse(results=normalized_results, source_refs=source_refs)


async def fault_code_lookup(arguments: dict) -> FaultCodeLookupResponse:
    request = FaultCodeLookupRequest.model_validate(arguments)
    fault_code = request.fault_code.upper()
    for chunk in get_searchable_chunks(request.doc_ids):
        if chunk.fault_code != fault_code:
            continue
        if not matches_device_model(chunk, request.device_model):
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
    aliases = PARAMETER_ALIASES.get(key, (key, request.parameter_name))
    for chunk in get_scoped_indexed_chunks(request.doc_ids):
        if chunk.content_type != "parameter":
            continue
        if not matches_device_model(chunk, request.device_model):
            continue
        parameter_name = str(chunk.metadata.get("parameter_name") or request.parameter_name)
        searchable_text = f"{chunk.section_title or ''}\n{chunk.text}".lower()
        if not any(alias.lower() in searchable_text for alias in aliases) and key not in parameter_name.lower():
            continue
        if request.observed_value is not None and (
            chunk.metadata.get("min") is None or chunk.metadata.get("max") is None
        ):
            continue
        is_abnormal = None
        if request.observed_value is not None:
            is_abnormal = not (
                float(chunk.metadata["min"]) <= request.observed_value <= float(chunk.metadata["max"])
            )
        return ParameterLookupResponse(
            status="found",
            parameter_name=request.parameter_name,
            standard_range=chunk.text,
            observed_value=request.observed_value,
            is_abnormal=is_abnormal,
            source_refs=[f"{chunk.source_file}:{chunk.page}"] if chunk.source_file else [],
        )

    if request.doc_ids:
        return ParameterLookupResponse(status="not_found", parameter_name=request.parameter_name)

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
    for chunk in get_searchable_chunks(request.doc_ids):
        if chunk.content_type != "safety_rule":
            continue
        if not matches_device_model(chunk, request.device_model):
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
        for chunk in get_searchable_chunks(request.doc_ids)
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


def expand_manual_qa_query(query: str) -> str:
    if not _looks_like_manual_qa_query(query):
        return query
    additions = [item for item in ACTION_INSTRUCTION_EXPANSIONS if item not in query]
    return " ".join([query, *additions])


def _looks_like_manual_qa_query(query: str) -> bool:
    lower_query = query.lower()
    return any(term.lower() in lower_query for term in MANUAL_QA_QUERY_TERMS)


def source_refs_from_result(result) -> list[str]:
    source_file = result.source_file
    page = result.page
    section_title = result.section_title
    if source_file and page is not None:
        return [f"{source_file}:{page}"]
    if source_file and section_title and not _is_page_number_title(section_title):
        return [f"{source_file} / {section_title}"]
    if source_file:
        return [str(source_file)]
    return []


def normalize_source_refs(source_refs: list[str]) -> list[str]:
    page_refs: dict[tuple[str, int], str] = {}
    section_refs: set[str] = set()
    bare_refs: set[str] = set()
    files_with_specific_refs: set[str] = set()

    for source in source_refs:
        if not source:
            continue
        page_match = re.match(r"^(?P<file>.+):(?P<page>\d+)$", source)
        if page_match:
            filename = page_match.group("file")
            page = int(page_match.group("page"))
            page_refs[(filename, page)] = source
            files_with_specific_refs.add(filename)
            continue

        if " / " in source:
            filename, section = source.split(" / ", 1)
            if section and not _is_page_number_title(section):
                section_refs.add(source)
                files_with_specific_refs.add(filename)
            continue

        bare_refs.add(source)

    ordered = [
        page_refs[key]
        for key in sorted(page_refs, key=lambda item: (item[0], item[1]))
    ]
    ordered.extend(sorted(section_refs))
    ordered.extend(sorted(source for source in bare_refs if source not in files_with_specific_refs))
    return ordered


def _is_page_number_title(title: str) -> bool:
    return bool(re.match(r"^(?:\d+\s*/\s*\d+|\d+|第\s*\d+\s*页|page\s*\d+)(?:\s*/\s*\d+)?$", title.strip(), re.IGNORECASE))
