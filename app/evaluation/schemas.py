from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    sample_id: str
    query: str
    expected_fault_code: str | None = None
    expected_tool_names: list[str] = Field(default_factory=list)
    expected_source_keywords: list[str] = Field(default_factory=list)
    expected_handoff: bool | None = None
    expected_safety_keywords: list[str] = Field(default_factory=list)


class RetrievalEvalResult(BaseModel):
    source_covered: bool
    matched_keywords: list[str] = Field(default_factory=list)


class AgentEvalResult(BaseModel):
    tool_selection_correct: bool
    fault_code_correct: bool
    handoff_correct: bool
    safety_covered: bool


class EvalResult(BaseModel):
    sample_id: str
    trace_id: str | None = None
    task_id: str
    tool_names: list[str] = Field(default_factory=list)
    extracted_fault_code: str | None = None
    handoff_required: bool = False
    retrieval: RetrievalEvalResult
    agent: AgentEvalResult


class EvalRunRequest(BaseModel):
    samples: list[EvalSample]


class EvalRunResponse(BaseModel):
    total: int
    tool_selection_accuracy: float
    fault_code_accuracy: float
    handoff_accuracy: float
    source_coverage: float
    safety_coverage: float
    results: list[EvalResult] = Field(default_factory=list)
