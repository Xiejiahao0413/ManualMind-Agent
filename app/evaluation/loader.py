import json
from pathlib import Path
from typing import Any

from app.evaluation.schemas import EvalSample


DEFAULT_EVAL_SAMPLE_COUNT = 50


def load_eval_samples(path: str, expected_count: int | None = DEFAULT_EVAL_SAMPLE_COUNT) -> list[EvalSample]:
    sample_path = Path(path)
    if not sample_path.exists():
        raise FileNotFoundError(f"Evaluation sample file not found: {path}")

    data = json.loads(sample_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Evaluation sample file must contain a JSON list.")

    samples: list[EvalSample] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation sample #{index} must be an object.")
        _validate_raw_sample(item, index)
        samples.append(EvalSample.model_validate(item))

    if expected_count is not None and len(samples) != expected_count:
        raise ValueError(
            f"Evaluation sample file must contain {expected_count} samples, got {len(samples)}."
        )
    return samples


def _validate_raw_sample(item: dict[str, Any], index: int) -> None:
    sample_id = item.get("sample_id")
    query = item.get("query")
    category = item.get("category")
    expected_tool_names = item.get("expected_tool_names")
    expected_sensitive_masks = item.get("expected_sensitive_masks", [])

    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"Evaluation sample #{index} has empty sample_id.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"Evaluation sample {sample_id} has empty query.")
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"Evaluation sample {sample_id} has empty category.")
    if not isinstance(expected_tool_names, list):
        raise ValueError(f"Evaluation sample {sample_id} expected_tool_names must be a list.")
    if not isinstance(expected_sensitive_masks, list):
        raise ValueError(f"Evaluation sample {sample_id} expected_sensitive_masks must be a list.")
    if "scenario_type" in item and (
        not isinstance(item["scenario_type"], str) or not item["scenario_type"].strip()
    ):
        raise ValueError(f"Evaluation sample {sample_id} has empty scenario_type.")
