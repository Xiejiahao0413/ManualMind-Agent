from app.agents import DiagnosisWorkflow
from app.evaluation.schemas import (
    AgentEvalResult,
    EvalResult,
    EvalRunResponse,
    EvalSample,
    RetrievalEvalResult,
)
from app.memory import InMemoryMemoryManager
from app.schemas.diagnosis import DiagnosisState
from app.tracing import InMemoryTraceManager


class EvaluationRunner:
    def __init__(self, trace_manager: InMemoryTraceManager | None = None) -> None:
        self.trace_manager = trace_manager or InMemoryTraceManager()

    async def run_eval(self, samples: list[EvalSample]) -> EvalRunResponse:
        results: list[EvalResult] = []
        for sample in samples:
            memory = InMemoryMemoryManager()
            workflow = DiagnosisWorkflow(memory=memory, trace_manager=self.trace_manager)
            state = DiagnosisState(
                task_id=f"eval-{sample.sample_id}",
                session_id=f"eval-session-{sample.sample_id}",
                user_query=sample.query,
                raw_query=sample.query,
                sanitized_query=sample.query,
            )
            final_state = await workflow.run(state)
            events = self.trace_manager.list_events(final_state.trace_id or "")
            tool_names = [
                str(event.metadata.get("tool_name"))
                for event in events
                if event.event_type == "tool_call_completed" and event.metadata.get("tool_name")
            ]

            source_text = " ".join(final_state.source_refs + [final_state.final_answer or ""])
            matched_source_keywords = [
                keyword for keyword in sample.expected_source_keywords if keyword in source_text
            ]
            safety_text = " ".join(final_state.safety_rules + [final_state.final_answer or ""])
            safety_covered = self.evaluate_safety_coverage(sample, safety_text)

            tool_selection_correct = self.evaluate_tool_selection(sample, tool_names)
            fault_code_correct = self.evaluate_fault_code_extraction(
                sample,
                final_state.fault_code,
            )
            handoff_correct = self.evaluate_handoff_decision(
                sample,
                final_state.handoff_required,
            )
            source_covered = self.evaluate_source_coverage(sample, matched_source_keywords)

            results.append(
                EvalResult(
                    sample_id=sample.sample_id,
                    trace_id=final_state.trace_id,
                    task_id=final_state.task_id,
                    tool_names=tool_names,
                    extracted_fault_code=final_state.fault_code,
                    handoff_required=final_state.handoff_required,
                    retrieval=RetrievalEvalResult(
                        source_covered=source_covered,
                        matched_keywords=matched_source_keywords,
                    ),
                    agent=AgentEvalResult(
                        tool_selection_correct=tool_selection_correct,
                        fault_code_correct=fault_code_correct,
                        handoff_correct=handoff_correct,
                        safety_covered=safety_covered,
                    ),
                )
            )

        total = len(results)
        return EvalRunResponse(
            total=total,
            tool_selection_accuracy=self._ratio(
                result.agent.tool_selection_correct for result in results
            ),
            fault_code_accuracy=self._ratio(result.agent.fault_code_correct for result in results),
            handoff_accuracy=self._ratio(result.agent.handoff_correct for result in results),
            source_coverage=self._ratio(result.retrieval.source_covered for result in results),
            safety_coverage=self._ratio(result.agent.safety_covered for result in results),
            results=results,
        )

    def evaluate_tool_selection(self, sample: EvalSample, tool_names: list[str]) -> bool:
        if not sample.expected_tool_names:
            return True
        return set(sample.expected_tool_names).issubset(set(tool_names))

    def evaluate_fault_code_extraction(
        self,
        sample: EvalSample,
        extracted_fault_code: str | None,
    ) -> bool:
        if sample.expected_fault_code is None:
            return True
        return extracted_fault_code == sample.expected_fault_code

    def evaluate_handoff_decision(self, sample: EvalSample, handoff_required: bool) -> bool:
        if sample.expected_handoff is None:
            return True
        return handoff_required is sample.expected_handoff

    def evaluate_source_coverage(
        self,
        sample: EvalSample,
        matched_source_keywords: list[str],
    ) -> bool:
        if not sample.expected_source_keywords:
            return True
        return set(sample.expected_source_keywords).issubset(set(matched_source_keywords))

    def evaluate_safety_coverage(self, sample: EvalSample, safety_text: str) -> bool:
        if not sample.expected_safety_keywords:
            return True
        return all(keyword in safety_text for keyword in sample.expected_safety_keywords)

    def _ratio(self, values) -> float:
        items = list(values)
        if not items:
            return 0.0
        return sum(1 for item in items if item) / len(items)
