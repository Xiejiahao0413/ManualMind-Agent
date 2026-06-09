from typing import Any

from app.mcp_server import MCPToolExecutor, create_default_tool_registry
from app.memory import InMemoryMemoryManager
from app.schemas.diagnosis import DiagnosisState
from app.schemas.retrieval import RetrievalResult
from app.schemas.tools import ToolCallRecord
from app.tools import (
    FallbackAction,
    RetryFallbackManager,
    ToolCallGuard,
    ToolIntent,
    ToolResultValidator,
    ToolRouter,
)


class WorkflowToolService:
    def __init__(
        self,
        router: ToolRouter,
        guard: ToolCallGuard,
        memory: InMemoryMemoryManager,
        executor: MCPToolExecutor | None = None,
        result_validator: ToolResultValidator | None = None,
        retry_fallback_manager: RetryFallbackManager | None = None,
    ) -> None:
        self.router = router
        self.guard = guard
        self.memory = memory
        self.executor = executor or MCPToolExecutor(create_default_tool_registry())
        self.result_validator = result_validator or ToolResultValidator()
        self.retry_fallback_manager = retry_fallback_manager or RetryFallbackManager()
        self._result_cache: dict[str, dict[str, Any]] = {}

    async def run_retrieval_tools(self, state: DiagnosisState) -> DiagnosisState:
        query_type = state.query_type or "general_fault_symptom"
        tool_names = self.router.route_query_type(query_type)
        state.workflow_events.append(
            {"event": "tool_routing_started", "tool_names": list(tool_names)}
        )

        for tool_name in tool_names:
            if self.executor.registry.get(tool_name) is None:
                await self._record_call(
                    state=state,
                    tool_name=tool_name,
                    args={},
                    args_signature=f"unregistered:{tool_name}",
                    status="skipped",
                    error_type="tool_not_registered",
                    result_summary="tool_not_registered",
                )
                continue

            args = self._build_tool_args(tool_name, state)
            intent = ToolIntent(tool_name=tool_name, args=args, required_args=self._required_args(tool_name))
            decision = self.router.validate_intent(
                intent,
                retry_count=state.retry_count,
                task_tool_call_count=len(state.tool_call_history),
            )

            if decision.status == "circuit_breaker_required":
                state.handoff_required = True
                state.handoff_reason = "circuit_breaker_retry_limit_reached"
                await self._record_call(
                    state=state,
                    tool_name=tool_name,
                    args=args,
                    args_signature=decision.args_signature or f"blocked:{tool_name}",
                    status="skipped",
                    error_type=decision.reason,
                    result_summary=decision.reason,
                )
                break

            if decision.status == "reuse_cached" and decision.args_signature:
                cached_data = self._result_cache.get(decision.args_signature)
                await self._record_call(
                    state=state,
                    tool_name=tool_name,
                    args=args,
                    args_signature=decision.args_signature,
                    status="success",
                    result_summary="reuse_cached",
                )
                if cached_data is not None:
                    self._apply_tool_data(state, tool_name, cached_data)
                continue

            if not decision.allowed or not decision.args_signature:
                await self._record_call(
                    state=state,
                    tool_name=tool_name,
                    args=args,
                    args_signature=decision.args_signature or f"blocked:{tool_name}",
                    status="skipped",
                    error_type=decision.reason,
                    result_summary=decision.reason,
                )
                if decision.reason == "sanitized_required":
                    state.handoff_required = True
                    state.handoff_reason = "sanitized_required"
                continue

            result = await self.executor.execute_tool(tool_name, args)
            if result.status == "error":
                fallback = self._fallback_for_error(result.error_type, state.retry_count)
                state.fallback_decision = str(fallback)
                state.retrieval_mode = str(fallback)
                if fallback in {FallbackAction.HUMAN_HANDOFF, FallbackAction.CIRCUIT_BREAKER_HANDOFF}:
                    state.handoff_required = True
                    state.handoff_reason = str(fallback)
                await self._record_call(
                    state=state,
                    tool_name=tool_name,
                    args=args,
                    args_signature=decision.args_signature,
                    status="failed",
                    retry_count=state.retry_count,
                    fallback_used=fallback != FallbackAction.RETRY,
                    error_type=result.error_type,
                    latency_ms=result.latency_ms,
                    result_summary=result.error_message,
                )
                continue

            data = result.data or {}
            validation_payload = self._validation_payload(tool_name, data)
            validation = self.result_validator.validate(validation_payload, risk_level=state.risk_level)
            if not validation.valid:
                fallback = self.retry_fallback_manager.decide(
                    validation.error_type.value if validation.error_type else "insufficient_evidence",
                    retry_count=state.retry_count,
                )
                state.fallback_decision = str(fallback)
                if fallback in {FallbackAction.HUMAN_HANDOFF, FallbackAction.CIRCUIT_BREAKER_HANDOFF}:
                    state.handoff_required = True
                    state.handoff_reason = validation.reason

            self._apply_tool_data(state, tool_name, data)
            self._result_cache[decision.args_signature] = data
            record = await self._record_call(
                state=state,
                tool_name=tool_name,
                args=args,
                args_signature=decision.args_signature,
                status="success",
                retry_count=state.retry_count,
                latency_ms=result.latency_ms,
                result_summary=self._summarize_tool_result(tool_name, data),
            )
            self.guard.remember_success(record)

        state.source_refs = sorted(set(state.source_refs))
        if state.handoff_required:
            state.retrieval_status = "handoff_required"
        elif state.source_refs or state.retrieved_chunks or state.fault_info or state.parameter_info:
            state.retrieval_status = "completed"
        else:
            state.retrieval_status = "empty"
        await self.memory.save_task(state)
        return state

    def _build_tool_args(self, tool_name: str, state: DiagnosisState) -> dict[str, Any]:
        query = state.sanitized_query or state.user_query
        if tool_name == "manual_hybrid_search":
            content_types = []
            if state.query_type == "fault_code":
                content_types = ["fault_code"]
            elif state.query_type == "parameter":
                content_types = ["parameter"]
            return {
                "query": query,
                "device_name": state.device_name,
                "device_model": state.device_model,
                "content_types": content_types,
                "top_k_bm25": 5,
                "top_k_dense": 5,
                "top_n_rerank": 5,
            }
        if tool_name == "fault_code_lookup":
            return {"fault_code": state.fault_code or "", "device_model": state.device_model}
        if tool_name == "parameter_lookup":
            return {
                "parameter_name": self._parameter_name_from_query(query),
                "device_model": state.device_model,
                "observed_value": None,
            }
        if tool_name == "safety_rule_search":
            return {
                "operation": query,
                "risk_level": state.risk_level,
                "device_model": state.device_model,
            }
        return {"query": query}

    def _required_args(self, tool_name: str) -> set[str]:
        if tool_name == "fault_code_lookup":
            return {"fault_code"}
        if tool_name == "parameter_lookup":
            return {"parameter_name"}
        if tool_name == "safety_rule_search":
            return {"operation"}
        return {"query"}

    def _apply_tool_data(self, state: DiagnosisState, tool_name: str, data: dict[str, Any]) -> None:
        state.workflow_events.append({"event": "tool_call_completed", "tool_name": tool_name})
        source_refs = data.get("source_refs") or []
        state.source_refs.extend(str(source) for source in source_refs)

        if tool_name == "manual_hybrid_search":
            state.retrieval_mode = str(data.get("retrieval_mode") or "hybrid")
            results = data.get("results") or []
            state.retrieved_chunks = results
            state.retrieved_evidence = [RetrievalResult.model_validate(result) for result in results]
        elif tool_name == "fault_code_lookup":
            if data.get("status") == "found":
                state.fault_info = data
        elif tool_name == "parameter_lookup":
            if data.get("status") == "found":
                state.parameter_info = data
        elif tool_name == "safety_rule_search":
            state.safety_rules = [str(rule) for rule in data.get("safety_rules") or []]
            for index, rule in enumerate(state.safety_rules):
                state.retrieved_chunks.append(
                    {
                        "chunk_id": f"safety_rule_{index}",
                        "text": rule,
                        "content_type": "safety_rule",
                        "source_refs": data.get("source_refs") or [],
                        "safety_evidence": True,
                    }
                )

    def _validation_payload(self, tool_name: str, data: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        if tool_name == "manual_hybrid_search":
            return data.get("results") or []
        if tool_name == "safety_rule_search":
            return {
                "source_refs": data.get("source_refs") or [],
                "safety_evidence": bool(data.get("safety_rules")),
            }
        return {"source_refs": data.get("source_refs") or []}

    async def _record_call(
        self,
        state: DiagnosisState,
        tool_name: str,
        args: dict[str, Any],
        args_signature: str,
        status: str,
        retry_count: int = 0,
        fallback_used: bool = False,
        error_type: str | None = None,
        latency_ms: float | None = None,
        result_summary: str | None = None,
    ) -> ToolCallRecord:
        record = ToolCallRecord(
            task_id=state.task_id,
            tool_name=tool_name,
            args=args,
            args_signature=args_signature,
            status=status,
            retry_count=retry_count,
            fallback_used=fallback_used,
            error_type=error_type,
            latency_ms=latency_ms,
            result_summary=result_summary,
        )
        state.tool_call_history.append(record)
        await self.memory.append_tool_call(record)
        return record

    def _fallback_for_error(self, error_type: str | None, retry_count: int) -> FallbackAction:
        known_errors = {
            "milvus_timeout",
            "bm25_error",
            "rerank_error",
            "llm_error",
            "missing_device_model",
            "insufficient_evidence",
            "high_risk_without_evidence",
        }
        normalized = error_type if error_type in known_errors else "insufficient_evidence"
        return self.retry_fallback_manager.decide(normalized, retry_count=retry_count)

    def _parameter_name_from_query(self, query: str) -> str:
        if "温度" in query or "temperature" in query.lower():
            return "temperature"
        if "电压" in query or "voltage" in query.lower():
            return "voltage"
        if "压力" in query or "pressure" in query.lower():
            return "pressure"
        if "维护周期" in query or "maintenance" in query.lower():
            return "maintenance_cycle"
        return "temperature"

    def _summarize_tool_result(self, tool_name: str, data: dict[str, Any]) -> str:
        if tool_name == "manual_hybrid_search":
            return f"results={len(data.get('results') or [])}"
        if tool_name == "safety_rule_search":
            return f"safety_rules={len(data.get('safety_rules') or [])}"
        return str(data.get("status") or "ok")
