import json
import re
import subprocess
import sys
from pathlib import Path


MANUALS_DIR = Path("data/manuals")
FAULT_CODE_PATTERN = re.compile(r"\b(?:E|F|P)\d{2,4}\b")


def test_manuals_directory_contains_at_least_10_markdown_manuals() -> None:
    manual_paths = sorted(MANUALS_DIR.glob("*.md"))

    assert MANUALS_DIR.exists()
    assert len(manual_paths) >= 10


def test_each_manual_contains_required_sections_and_fault_codes() -> None:
    for manual_path in sorted(MANUALS_DIR.glob("*.md")):
        text = manual_path.read_text(encoding="utf-8")

        assert "故障码表" in text
        assert "安全注意事项" in text
        assert "维护流程" in text
        assert "人工接管条件" in text
        assert len(set(FAULT_CODE_PATTERN.findall(text))) >= 5


def test_index_manuals_script_runs_and_outputs_summary() -> None:
    output = _run_script("scripts/index_manuals.py")

    assert output["indexed_docs_count"] >= 10
    assert output["total_chunks_count"] >= 80
    assert output["fault_codes_count"] >= 50
    assert len(output["device_models"]) >= 10


def test_demo_multi_manuals_script_runs() -> None:
    output = _run_script("scripts/demo_multi_manuals.py")

    assert output["index_summary"]["indexed_docs_count"] >= 10
    assert len(output["results"]) == 4
    for item in output["results"]:
        assert item["trace_id"]
        assert item["source_refs"]
        assert item["final_answer"]


def _run_script(script_path: str) -> dict:
    result = subprocess.run(
        [sys.executable, script_path],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return json.loads(result.stdout)
