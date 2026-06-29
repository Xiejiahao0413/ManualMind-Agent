import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents import DiagnosisWorkflow
from app.core.dependencies import get_manual_indexer
from app.schemas.diagnosis import DiagnosisRequest


DEMO_MANUAL_PATH = PROJECT_ROOT / "data" / "demo_manuals" / "a100_manual.md"
DEMO_QUERY = "A100 E03 temperature sensor fault, please generate a diagnosis report."


async def main() -> None:
    get_manual_indexer().index_document(
        content=DEMO_MANUAL_PATH.read_bytes(),
        filename=DEMO_MANUAL_PATH.name,
        doc_id="demo-llm-report-a100",
        device_name="Air Compressor A100",
        device_model="A100",
    )
    workflow = DiagnosisWorkflow()
    state = DiagnosisWorkflow.from_request(
        DiagnosisRequest(session_id="demo-llm-report", message=DEMO_QUERY)
    )
    final_state = await workflow.run(state)
    configured_provider = os.getenv("MANUALMIND_LLM_PROVIDER", "").lower()
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))

    print(
        json.dumps(
            {
                "trace_id": final_state.trace_id,
                "source_refs": final_state.source_refs,
                "llm_provider": final_state.llm_provider,
                "llm_enabled": final_state.llm_enabled,
                "configured_provider": configured_provider or "template",
                "openai_api_key_present": has_api_key,
                "fallback_used": final_state.fallback_used,
                "llm_error_type": final_state.llm_error_type,
                "final_answer": final_state.final_answer,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
