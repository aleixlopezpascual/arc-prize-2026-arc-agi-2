from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_grid(grid: Any) -> tuple[bool, str]:
    """Validates that a grid is a 2D list of integers 0-9 with dimensions 1x1 to 30x30."""
    if not isinstance(grid, list):
        return False, f"Expected list, got {type(grid).__name__}"
    if len(grid) == 0:
        return False, "Grid height is 0 (empty list)"
    if len(grid) > 30:
        return False, f"Grid height {len(grid)} exceeds maximum 30"

    width = None
    for r_idx, row in enumerate(grid):
        if not isinstance(row, list):
            return False, f"Row {r_idx} is not a list: {type(row).__name__}"
        if width is None:
            width = len(row)
            if width == 0:
                return False, f"Row {r_idx} is empty"
            if width > 30:
                return False, f"Grid width {width} exceeds maximum 30"
        elif len(row) != width:
            return False, f"Inconsistent row length at row {r_idx}: expected {width}, got {len(row)}"

        for c_idx, val in enumerate(row):
            if not isinstance(val, (int,)):
                return False, f"Cell ({r_idx}, {c_idx}) is not integer: {type(val).__name__}"
            if not (0 <= val <= 9):
                return False, f"Cell ({r_idx}, {c_idx}) value {val} out of bounds [0, 9]"

    return True, "Valid"


def validate_submission_dict(
    submission: dict[str, Any],
    challenges: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Validates submission dictionary against challenges specification.

    Returns:
        (is_valid, list_of_errors)
    """
    errors: list[str] = []

    if not isinstance(submission, dict):
        return False, ["Submission root must be a dictionary"]

    # Check missing tasks
    missing_tasks = set(challenges.keys()) - set(submission.keys())
    if missing_tasks:
        errors.append(f"Missing {len(missing_tasks)} tasks from submission: sample {list(missing_tasks)[:5]}")

    # Check extra tasks
    extra_tasks = set(submission.keys()) - set(challenges.keys())
    if extra_tasks:
        errors.append(f"Submission contains {len(extra_tasks)} unknown task keys: sample {list(extra_tasks)[:5]}")

    for task_id, chal in challenges.items():
        if task_id not in submission:
            continue

        preds = submission[task_id]
        if not isinstance(preds, list):
            errors.append(f"Task {task_id}: expected list of prediction dicts, got {type(preds).__name__}")
            continue

        expected_len = len(chal.get("test", []))
        if len(preds) != expected_len:
            errors.append(
                f"Task {task_id}: expected {expected_len} test predictions, got {len(preds)}"
            )
            continue

        for test_idx, pred_item in enumerate(preds):
            if not isinstance(pred_item, dict):
                errors.append(f"Task {task_id}[{test_idx}]: expected dict, got {type(pred_item).__name__}")
                continue

            for attempt_key in ["attempt_1", "attempt_2"]:
                if attempt_key not in pred_item:
                    errors.append(f"Task {task_id}[{test_idx}]: missing key '{attempt_key}'")
                else:
                    grid = pred_item[attempt_key]
                    is_valid, msg = validate_grid(grid)
                    if not is_valid:
                        errors.append(f"Task {task_id}[{test_idx}].{attempt_key}: invalid grid - {msg}")

    return len(errors) == 0, errors


def validate_submission_file(
    submission_path: str | Path,
    challenges_path: str | Path,
) -> tuple[bool, list[str]]:
    """Validates a submission JSON file against challenges JSON file."""
    sub_p = Path(submission_path)
    chal_p = Path(challenges_path)

    if not sub_p.exists():
        return False, [f"Submission file does not exist: {sub_p}"]
    if not chal_p.exists():
        return False, [f"Challenges file does not exist: {chal_p}"]

    try:
        sub_data = json.loads(sub_p.read_text())
    except Exception as e:
        return False, [f"Failed to parse submission JSON: {e}"]

    try:
        chal_data = json.loads(chal_p.read_text())
    except Exception as e:
        return False, [f"Failed to parse challenges JSON: {e}"]

    return validate_submission_dict(sub_data, chal_data)
