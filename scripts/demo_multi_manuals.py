import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents import DiagnosisWorkflow
from app.schemas.diagnosis import DiagnosisRequest
from scripts.manuals import MANUALS_DIR, index_manuals


DEMO_QUERIES = [
    "A100 显示 E03 怎么处理？",
    "B200 液压压力不足怎么办？",
    "C300 输送电机过载是什么原因？",
    "H800 机械臂急停后如何排查？",
]


async def main() -> None:
    index_summary = index_manuals(MANUALS_DIR)
    workflow = DiagnosisWorkflow()
    results = []

    for index, query in enumerate(DEMO_QUERIES, start=1):
        state = DiagnosisWorkflow.from_request(
            DiagnosisRequest(
                session_id=f"multi-manual-demo-{index}",
                message=query,
            )
        )
        final_state = await workflow.run(state)
        results.append(
            {
                "query": query,
                "trace_id": final_state.trace_id,
                "source_refs": final_state.source_refs,
                "final_answer": final_state.final_answer,
            }
        )

    print(
        json.dumps(
            {
                "index_summary": index_summary,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
