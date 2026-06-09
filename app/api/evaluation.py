from fastapi import APIRouter

from app.core.dependencies import get_trace_manager
from app.evaluation import EvalRunFileRequest, EvalRunRequest, EvalRunResponse, EvaluationRunner

router = APIRouter(tags=["evaluation"])


@router.post("/eval/run", response_model=EvalRunResponse)
async def run_eval(request: EvalRunRequest) -> EvalRunResponse:
    runner = EvaluationRunner(trace_manager=get_trace_manager())
    return await runner.run_eval(request.samples)


@router.post("/eval/run-file", response_model=EvalRunResponse)
async def run_eval_file(request: EvalRunFileRequest) -> EvalRunResponse:
    runner = EvaluationRunner(trace_manager=get_trace_manager())
    return await runner.run_eval_file(request.path)
