import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents import DiagnosisWorkflow
from app.schemas.diagnosis import DiagnosisRequest
from scripts.pdf_manuals import PDF_MANUALS_DIR, index_pdf_manuals


DEMO_QUERIES = [
    "PDF manual A100 E03 how should it be handled?",
    "What safety rules are listed in the B200 PDF manual?",
]


async def main() -> None:
    index_summary = index_pdf_manuals(PDF_MANUALS_DIR)
    workflow = DiagnosisWorkflow()
    results = []

    for index, query in enumerate(DEMO_QUERIES, start=1):
        state = DiagnosisWorkflow.from_request(
            DiagnosisRequest(
                session_id=f"pdf-manual-demo-{index}",
                message=query,
            )
        )
        final_state = await workflow.run(state)
        pages = sorted(
            {
                int(chunk["page"])
                for chunk in final_state.retrieved_chunks
                if chunk.get("page") is not None
            }
        )
        results.append(
            {
                "query": query,
                "trace_id": final_state.trace_id,
                "source_refs": final_state.source_refs,
                "pages": pages,
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
