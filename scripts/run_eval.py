import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.dependencies import get_manual_indexer
from app.evaluation import EvaluationRunner
from app.tracing import InMemoryTraceManager


DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "data/eval_samples/equipment_fault_eval_50.json"
ADVERSARIAL_SAMPLE_PATH = PROJECT_ROOT / "data/eval_samples/equipment_fault_adversarial_30.json"
DEMO_MANUAL_PATH = PROJECT_ROOT / "data/demo_manuals/a100_manual.md"
EVAL_NOTE = "This evaluation uses local demo data and mock/in-memory components."


async def main() -> None:
    args = parse_args()
    _index_demo_manual()

    runner = EvaluationRunner(trace_manager=InMemoryTraceManager())
    if args.standard:
        response = await runner.run_eval_files([str(DEFAULT_SAMPLE_PATH)], expected_counts=[50])
    elif args.adversarial:
        response = await runner.run_eval_files([str(ADVERSARIAL_SAMPLE_PATH)], expected_counts=[30])
    else:
        response = await runner.run_eval_files(
            [str(DEFAULT_SAMPLE_PATH), str(ADVERSARIAL_SAMPLE_PATH)],
            expected_counts=[50, 30],
        )
    summary = {
        "note": EVAL_NOTE,
        "total_samples": response.total,
        "tool_selection_accuracy": response.tool_selection_accuracy,
        "fault_code_accuracy": response.fault_code_accuracy,
        "handoff_accuracy": response.handoff_accuracy,
        "source_coverage": response.source_coverage,
        "safety_coverage": response.safety_coverage,
        "sensitive_mask_accuracy": response.sensitive_mask_accuracy,
        "adversarial_safety_accuracy": response.adversarial_safety_accuracy,
        "boundary_handling_accuracy": response.boundary_handling_accuracy,
        "category_breakdown": response.category_breakdown,
        "scenario_breakdown": response.scenario_breakdown,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ManualMind-Agent evaluation samples.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--standard", action="store_true", help="Run only the 50 standard samples.")
    group.add_argument(
        "--adversarial",
        action="store_true",
        help="Run only the 30 adversarial/boundary samples.",
    )
    group.add_argument("--all", action="store_true", help="Run all 80 samples. This is the default.")
    return parser.parse_args()


def _index_demo_manual() -> None:
    get_manual_indexer().index_document(
        content=DEMO_MANUAL_PATH.read_bytes(),
        filename=DEMO_MANUAL_PATH.name,
        doc_id="eval-demo-a100-manual",
        device_name="空压机 A100",
        device_model="A100",
    )


if __name__ == "__main__":
    asyncio.run(main())
