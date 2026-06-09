import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents import DiagnosisWorkflow
from app.core.dependencies import get_manual_indexer
from app.schemas.diagnosis import DiagnosisRequest


DEMO_MANUAL_PATH = PROJECT_ROOT / "data/demo_manuals/a100_manual.md"
DEMO_DOC_ID = "demo-a100-manual"
DEMO_QUERY = "空压机 A100 报 E03，应该如何排查？"


async def main() -> None:
    manual_path = DEMO_MANUAL_PATH
    content = manual_path.read_bytes()

    index_result = get_manual_indexer().index_document(
        content=content,
        filename=manual_path.name,
        doc_id=DEMO_DOC_ID,
        device_name="空压机 A100",
        device_model="A100",
    )

    workflow = DiagnosisWorkflow()
    state = DiagnosisWorkflow.from_request(
        DiagnosisRequest(
            session_id="demo-session",
            message=DEMO_QUERY,
        )
    )
    final_state = await workflow.run(state)

    print(f"indexed_doc_id: {index_result.doc_id}")
    print(f"chunks_count: {index_result.chunks_count}")
    print(f"trace_id: {final_state.trace_id}")
    print(f"source_refs: {final_state.source_refs}")
    print("final_answer:")
    print(final_state.final_answer)


if __name__ == "__main__":
    asyncio.run(main())
