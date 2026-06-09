import asyncio
import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from app.evaluation import EvaluationRunner, load_eval_samples
from app.main import app
from app.tracing import InMemoryTraceManager


SAMPLE_PATH = Path("data/eval_samples/equipment_fault_eval_50.json")
ADVERSARIAL_SAMPLE_PATH = Path("data/eval_samples/equipment_fault_adversarial_30.json")
REQUIRED_FIELDS = {
    "sample_id",
    "query",
    "expected_fault_code",
    "expected_tool_names",
    "expected_source_keywords",
    "expected_handoff",
    "expected_safety_keywords",
    "category",
}
ADVERSARIAL_REQUIRED_FIELDS = REQUIRED_FIELDS | {
    "scenario_type",
    "expected_sensitive_masks",
    "expected_behavior",
}
REQUIRED_CATEGORIES = {"fault_code", "symptom", "parameter", "safety", "handoff"}
REQUIRED_SCENARIOS = {"boundary", "adversarial", "failure", "high_risk", "security"}


def load_raw_samples() -> list[dict]:
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


def load_raw_adversarial_samples() -> list[dict]:
    return json.loads(ADVERSARIAL_SAMPLE_PATH.read_text(encoding="utf-8"))


def test_eval_sample_file_exists() -> None:
    assert SAMPLE_PATH.exists()


def test_adversarial_eval_sample_file_exists() -> None:
    assert ADVERSARIAL_SAMPLE_PATH.exists()


def test_eval_sample_file_contains_50_samples() -> None:
    assert len(load_raw_samples()) == 50


def test_adversarial_eval_sample_file_contains_30_samples() -> None:
    assert len(load_raw_adversarial_samples()) == 30


def test_eval_samples_have_required_fields() -> None:
    for sample in load_raw_samples():
        assert REQUIRED_FIELDS.issubset(sample)
        assert sample["sample_id"]
        assert sample["query"]
        assert isinstance(sample["expected_tool_names"], list)


def test_adversarial_eval_samples_have_required_fields() -> None:
    for sample in load_raw_adversarial_samples():
        assert ADVERSARIAL_REQUIRED_FIELDS.issubset(sample)
        assert sample["sample_id"]
        assert sample["query"]
        assert sample["scenario_type"]
        assert isinstance(sample["expected_tool_names"], list)
        assert isinstance(sample["expected_sensitive_masks"], list)


def test_eval_sample_categories_cover_required_set() -> None:
    categories = {sample["category"] for sample in load_raw_samples()}

    assert REQUIRED_CATEGORIES.issubset(categories)


def test_adversarial_scenario_types_cover_required_set() -> None:
    scenarios = {sample["scenario_type"] for sample in load_raw_adversarial_samples()}

    assert REQUIRED_SCENARIOS.issubset(scenarios)


def test_load_eval_samples_loads_50_samples() -> None:
    samples = load_eval_samples(str(SAMPLE_PATH))

    assert len(samples) == 50
    assert samples[0].sample_id
    assert samples[0].category in REQUIRED_CATEGORIES


def test_load_eval_samples_loads_30_adversarial_samples() -> None:
    samples = load_eval_samples(str(ADVERSARIAL_SAMPLE_PATH), expected_count=30)

    assert len(samples) == 30
    assert samples[0].scenario_type in REQUIRED_SCENARIOS


def test_run_eval_file_returns_total_and_category_breakdown() -> None:
    runner = EvaluationRunner(trace_manager=InMemoryTraceManager())

    response = asyncio.run(runner.run_eval_file(str(SAMPLE_PATH)))

    assert response.total == 50
    assert REQUIRED_CATEGORIES.issubset(response.category_breakdown)
    assert sum(item["total"] for item in response.category_breakdown.values()) == 50
    assert "normal" in response.scenario_breakdown


def test_eval_run_file_api_returns_total_and_breakdown() -> None:
    client = TestClient(app)

    response = client.post("/api/eval/run-file", json={"path": str(SAMPLE_PATH)})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 50
    assert REQUIRED_CATEGORIES.issubset(body["category_breakdown"])
    assert "scenario_breakdown" in body


def test_handoff_samples_are_marked_expected_handoff() -> None:
    handoff_samples = [
        sample for sample in load_raw_samples() if sample["category"] == "handoff"
    ]

    assert len(handoff_samples) >= 5
    assert all(sample["expected_handoff"] is True for sample in handoff_samples)


def test_high_risk_adversarial_samples_include_expected_handoff() -> None:
    high_risk_samples = [
        sample
        for sample in load_raw_adversarial_samples()
        if sample["scenario_type"] == "high_risk"
    ]

    assert len(high_risk_samples) >= 5
    assert all(sample["expected_handoff"] is True for sample in high_risk_samples)


def test_security_adversarial_samples_include_expected_masks() -> None:
    security_samples = [
        sample
        for sample in load_raw_adversarial_samples()
        if sample["scenario_type"] == "security"
    ]

    assert len(security_samples) >= 5
    assert all(sample["expected_sensitive_masks"] for sample in security_samples)


def test_safety_samples_have_safety_keywords() -> None:
    safety_samples = [sample for sample in load_raw_samples() if sample["category"] == "safety"]

    assert safety_samples
    assert all(sample["expected_safety_keywords"] for sample in safety_samples)


def test_run_eval_files_generates_scenario_breakdown_and_new_metrics() -> None:
    runner = EvaluationRunner(trace_manager=InMemoryTraceManager())

    response = asyncio.run(
        runner.run_eval_files(
            [str(SAMPLE_PATH), str(ADVERSARIAL_SAMPLE_PATH)],
            expected_counts=[50, 30],
        )
    )

    assert response.total == 80
    assert REQUIRED_SCENARIOS.issubset(response.scenario_breakdown)
    assert "normal" in response.scenario_breakdown
    assert isinstance(response.sensitive_mask_accuracy, float)
    assert isinstance(response.boundary_handling_accuracy, float)


def test_run_eval_cli_default_loads_80_samples() -> None:
    output = _run_eval_cli()

    assert output["total_samples"] == 80
    assert "scenario_breakdown" in output


def test_run_eval_cli_standard_loads_50_samples() -> None:
    output = _run_eval_cli("--standard")

    assert output["total_samples"] == 50


def test_run_eval_cli_adversarial_loads_30_samples() -> None:
    output = _run_eval_cli("--adversarial")

    assert output["total_samples"] == 30
    assert REQUIRED_SCENARIOS.issubset(output["scenario_breakdown"])
    assert output["category_breakdown"]["handoff"]["handoff_accuracy"] > 0
    assert output["scenario_breakdown"]["high_risk"]["handoff_accuracy"] > 0


def _run_eval_cli(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, "scripts/run_eval.py", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)
