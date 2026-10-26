from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

from src.metrics import GridType, compute_competition_metric


def assign_task_folds(
    task_ids: Sequence[str],
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, int]:
    """Assign each unique task ID to one stable, balanced fold.

    A SHA-256 ordering avoids Python's process-randomized ``hash`` and remains
    reproducible across local and Kaggle Python versions.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise TypeError("n_splits must be an integer.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    ids = list(task_ids)
    if any(not isinstance(task_id, str) or not task_id for task_id in ids):
        raise ValueError("Task IDs must be non-empty strings.")
    if len(set(ids)) != len(ids):
        raise ValueError("Task IDs must be unique.")
    if len(ids) < n_splits:
        raise ValueError("There must be at least one task per fold.")

    seed_prefix = f"{seed}:".encode()
    ordered_ids = sorted(
        ids,
        key=lambda task_id: (sha256(seed_prefix + task_id.encode("utf-8")).digest(), task_id),
    )
    return {task_id: index % n_splits for index, task_id in enumerate(ordered_ids)}


def score_predictions_by_fold(
    predictions: Mapping[str, list[dict[str, Any]]],
    solutions: Mapping[str, list[GridType]],
    fold_assignments: Mapping[str, int],
    n_splits: int = 5,
) -> dict[str, Any]:
    """Score complete task-level predictions per fold using ``src.metrics``."""
    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise TypeError("n_splits must be an integer.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two.")

    task_ids = set(solutions)
    if set(predictions) != task_ids or set(fold_assignments) != task_ids:
        raise ValueError("predictions, solutions, and fold assignments must have the same task IDs.")

    for task_id, fold_id in fold_assignments.items():
        if isinstance(fold_id, bool) or not isinstance(fold_id, int):
            raise TypeError(f"Fold ID for task {task_id} must be an integer.")
        if not 0 <= fold_id < n_splits:
            raise ValueError(f"Invalid fold IDs: values must be in [0, {n_splits - 1}].")

    if set(fold_assignments.values()) != set(range(n_splits)):
        raise ValueError("Every fold ID must be assigned at least one task.")

    overall = compute_competition_metric(dict(predictions), dict(solutions))
    folds: dict[str, Any] = {}
    for fold_id in range(n_splits):
        ids = sorted(task_id for task_id, assigned_fold in fold_assignments.items() if assigned_fold == fold_id)
        fold_predictions = {task_id: predictions[task_id] for task_id in ids}
        fold_solutions = {task_id: solutions[task_id] for task_id in ids}
        folds[str(fold_id)] = {
            "task_ids": ids,
            "metrics": compute_competition_metric(fold_predictions, fold_solutions),
        }

    return {"n_splits": n_splits, "folds": folds, "overall": overall}


def compare_fold_reports(
    control_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare paired per-output outcomes from two reports with identical folds."""
    n_splits = control_report.get("n_splits")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("Control report must contain a valid n_splits value.")
    if candidate_report.get("n_splits") != n_splits:
        raise ValueError("Reports must use the same number of folds.")
    if control_report.get("seed") != candidate_report.get("seed"):
        raise ValueError("Reports must use the same fold seed.")

    assignments = control_report.get("fold_assignments")
    if not isinstance(assignments, Mapping) or assignments != candidate_report.get("fold_assignments"):
        raise ValueError("Reports must use the same fold assignments.")
    if set(assignments.values()) != set(range(n_splits)):
        raise ValueError("Every fold ID must be assigned at least one task.")

    control_details = control_report.get("overall", {}).get("per_task_detail", {})
    candidate_details = candidate_report.get("overall", {}).get("per_task_detail", {})
    task_ids = set(assignments)
    if set(control_details) != task_ids or set(candidate_details) != task_ids:
        raise ValueError("Both reports must contain per-task results for the assigned task IDs.")

    fold_summaries: dict[str, Any] = {}
    totals = {
        "control_correct": 0,
        "candidate_correct": 0,
        "total_test_outputs": 0,
        "candidate_only_solved": 0,
        "control_only_solved": 0,
        "both_solved": 0,
        "neither_solved": 0,
    }
    for fold_id in range(n_splits):
        fold_ids = sorted(task_id for task_id, value in assignments.items() if value == fold_id)
        counts = {key: 0 for key in totals}
        for task_id in fold_ids:
            control_results = control_details[task_id].get("test_results")
            candidate_results = candidate_details[task_id].get("test_results")
            if not isinstance(control_results, list) or not isinstance(candidate_results, list):
                raise TypeError(f"Task {task_id} is missing test-level results.")
            if len(control_results) != len(candidate_results):
                raise ValueError(f"Task {task_id} has mismatched output counts between reports.")
            for control_correct, candidate_correct in zip(control_results, candidate_results):
                if type(control_correct) is not bool or type(candidate_correct) is not bool:
                    raise TypeError("Per-output results must be booleans.")
                counts["total_test_outputs"] += 1
                counts["control_correct"] += control_correct
                counts["candidate_correct"] += candidate_correct
                if candidate_correct and control_correct:
                    counts["both_solved"] += 1
                elif candidate_correct:
                    counts["candidate_only_solved"] += 1
                elif control_correct:
                    counts["control_only_solved"] += 1
                else:
                    counts["neither_solved"] += 1

        delta = counts["candidate_correct"] - counts["control_correct"]
        fold_summaries[str(fold_id)] = {
            "task_ids": fold_ids,
            "control_correct_test_outputs": counts["control_correct"],
            "candidate_correct_test_outputs": counts["candidate_correct"],
            "total_test_outputs": counts["total_test_outputs"],
            "delta_correct_test_outputs": delta,
            "delta_accuracy": delta / counts["total_test_outputs"],
            "candidate_only_solved": counts["candidate_only_solved"],
            "control_only_solved": counts["control_only_solved"],
            "both_solved": counts["both_solved"],
            "neither_solved": counts["neither_solved"],
        }
        for key in totals:
            totals[key] += counts[key]

    pooled_delta = totals["candidate_correct"] - totals["control_correct"]
    nonnegative_folds = sum(
        summary["delta_correct_test_outputs"] >= 0 for summary in fold_summaries.values()
    )
    minimum_fold_delta = min(summary["delta_correct_test_outputs"] for summary in fold_summaries.values())
    gates = {
        "positive_pooled_gain": pooled_delta > 0,
        "nonnegative_folds": nonnegative_folds >= n_splits - 1,
        "minimum_nonnegative_folds_required": n_splits - 1,
        "minimum_fold_loss_at_most_one": minimum_fold_delta >= -1,
        "passes_cv_accuracy_gates": (
            pooled_delta > 0 and nonnegative_folds >= n_splits - 1 and minimum_fold_delta >= -1
        ),
    }
    return {
        "n_splits": n_splits,
        "seed": control_report.get("seed"),
        "overall": {
            "control_correct_test_outputs": totals["control_correct"],
            "candidate_correct_test_outputs": totals["candidate_correct"],
            "total_test_outputs": totals["total_test_outputs"],
            "delta_correct_test_outputs": pooled_delta,
            "delta_accuracy": pooled_delta / totals["total_test_outputs"],
            "candidate_only_solved": totals["candidate_only_solved"],
            "control_only_solved": totals["control_only_solved"],
            "both_solved": totals["both_solved"],
            "neither_solved": totals["neither_solved"],
        },
        "folds": fold_summaries,
        "gates": gates,
    }

