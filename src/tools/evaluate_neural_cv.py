from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config import SEED
from src.neural.cv_folds import assign_task_folds, score_predictions_by_fold
from src.validator import validate_submission_dict


def _read_json_object(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk, rejecting other root types."""
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at {source}.")
    return value


def evaluate_files(
    predictions_path: str | Path,
    challenges_path: str | Path,
    solutions_path: str | Path,
    output_path: str | Path,
    n_splits: int = 5,
    seed: int = SEED,
) -> dict[str, Any]:
    """Validate neural predictions and write task-ID-grouped CV metrics."""
    predictions = _read_json_object(predictions_path)
    challenges = _read_json_object(challenges_path)
    solutions = _read_json_object(solutions_path)

    if set(challenges) != set(solutions):
        raise ValueError("Challenge and solution files must contain the same task IDs.")

    is_valid, errors = validate_submission_dict(predictions, challenges)
    if not is_valid:
        details = "; ".join(errors[:5])
        raise ValueError(f"prediction validation failed: {details}")

    assignments = assign_task_folds(sorted(challenges), n_splits=n_splits, seed=seed)
    scores = score_predictions_by_fold(
        predictions,
        solutions,
        assignments,
        n_splits=n_splits,
    )
    report = {
        "metric": "src.metrics.compute_competition_metric (exact-match outputs, up to two attempts)",
        "folding_unit": "task_id",
        "n_splits": n_splits,
        "seed": seed,
        "task_count": len(challenges),
        "protocol_note": (
            "The neural worker restores the base adapter for each task. This report groups one complete "
            "task-local inference run into deterministic task-ID folds; it is not five separate global model fits."
        ),
        "fold_assignments": assignments,
        **scores,
    }

    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing CV report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--challenges", required=True, type=Path)
    parser.add_argument("--solutions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    report = evaluate_files(
        args.predictions,
        args.challenges,
        args.solutions,
        args.output,
        n_splits=args.folds,
        seed=args.seed,
    )
    overall = report["overall"]
    print(
        f"Wrote {args.output}: {overall['correct_test_outputs']}/"
        f"{overall['total_test_outputs']} outputs correct "
        f"({overall['accuracy']:.4%}) across {report['task_count']} tasks."
    )


if __name__ == "__main__":
    main()
