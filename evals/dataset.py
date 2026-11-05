import json
from pathlib import Path
from typing import Any


def load_cases(path: Path, limit: int = 30) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if not path.is_file():
        raise FileNotFoundError(
            f"Reviewed dataset not found: {path}. Review evals/dataset.jsonl "
            "and save accepted rows as evals/reviewed_dataset.jsonl."
        )

    cases = []
    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            for field in ("question", "ground_truth", "source"):
                if not isinstance(case.get(field), str) or not case[field].strip():
                    raise ValueError(f"{path}:{line_number}: expected non-empty string field {field!r}")
            if not isinstance(case.get("reference_contexts"), list) or not case["reference_contexts"]:
                raise ValueError(f"{path}:{line_number}: expected non-empty reference_contexts list")
            cases.append(case)
            if len(cases) == limit:
                break

    if not cases:
        raise ValueError(f"No valid evaluation cases found in {path}")
    return cases
