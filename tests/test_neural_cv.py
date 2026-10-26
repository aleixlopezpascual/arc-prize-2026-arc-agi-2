from __future__ import annotations

from typing import Any

import pytest

from src.neural.cv_folds import (
    assign_task_folds,
    compare_fold_reports,
    score_predictions_by_fold,
)


def test_assign_task_folds_is_deterministic_balanced_and_task_grouped() -> None:
    task_ids = [f"task-{i:02d}" for i in range(13)]

    assignments = assign_task_folds(task_ids, n_splits=5, seed=42)

    assert assignments == assign_task_folds(list(reversed(task_ids)), n_splits=5, seed=42)
    assert set(assignments) == set(task_ids)
    assert set(assignments.values()) == set(range(5))
    fold_sizes = [sum(fold == index for fold in assignments.values()) for index in range(5)]
    assert max(fold_sizes) - min(fold_sizes) <= 1


def test_assign_task_folds_rejects_invalid_ids_and_split_counts() -> None:
    with pytest.raises(ValueError, match="unique"):
        assign_task_folds(["same", "same"], n_splits=2, seed=42)
    with pytest.raises(ValueError, match="at least two"):
        assign_task_folds(["one", "two"], n_splits=1, seed=42)
    with pytest.raises(ValueError, match="at least one task per fold"):
        assign_task_folds(["one"], n_splits=2, seed=42)


def test_score_predictions_by_fold_uses_official_metric_and_covers_all_tasks() -> None:
    predictions = {
        "task-a": [{"attempt_1": [[1]], "attempt_2": [[0]]}],
        "task-b": [{"attempt_1": [[9]], "attempt_2": [[2]]}],
        "task-c": [{"attempt_1": [[3]], "attempt_2": [[4]]}],
    }
    solutions = {
        "task-a": [[[1]]],
        "task-b": [[[2]]],
        "task-c": [[[4]]],
    }
    assignments = {"task-a": 0, "task-b": 1, "task-c": 1}

    report = score_predictions_by_fold(
        predictions,
        solutions,
        assignments,
        n_splits=2,
    )

    assert report["overall"]["correct_test_outputs"] == 3
    assert report["overall"]["total_test_outputs"] == 3
    assert report["folds"]["0"]["metrics"]["accuracy"] == 1.0
    assert report["folds"]["1"]["metrics"]["accuracy"] == 1.0
    assert report["folds"]["0"]["task_ids"] == ["task-a"]
    assert report["folds"]["1"]["task_ids"] == ["task-b", "task-c"]


def test_score_predictions_by_fold_rejects_incomplete_or_invalid_assignments() -> None:
    predictions = {"task-a": [{"attempt_1": [[1]], "attempt_2": [[0]]}]}
    solutions = {"task-a": [[[1]]], "task-b": [[[2]]]}
    assignments = {"task-a": 0, "task-b": 1}

    with pytest.raises(ValueError, match="same task IDs"):
        score_predictions_by_fold(predictions, solutions, assignments, n_splits=2)

    with pytest.raises(ValueError, match="fold IDs"):
        score_predictions_by_fold(
            {**predictions, "task-b": [{"attempt_1": [[2]], "attempt_2": [[0]]}]},
            solutions,
            {"task-a": 0, "task-b": 2},
            n_splits=2,
        )


def test_compare_fold_reports_computes_paired_deltas_and_accuracy_gates() -> None:
    assignments = {"task-a": 0, "task-b": 0, "task-c": 1, "task-d": 1, "task-e": 1}
    control_results = {
        "task-a": [True],
        "task-b": [False],
        "task-c": [True],
        "task-d": [False],
        "task-e": [False],
    }
    candidate_results = {
        "task-a": [True],
        "task-b": [True],
        "task-c": [False],
        "task-d": [False],
        "task-e": [False],
    }

    def report(results: dict[str, list[bool]]) -> dict[str, Any]:
        return {
            "n_splits": 2,
            "seed": 42,
            "fold_assignments": assignments,
            "overall": {
                "per_task_detail": {
                    task_id: {"test_results": output_results}
                    for task_id, output_results in results.items()
                }
            },
        }

    comparison = compare_fold_reports(report(control_results), report(candidate_results))

    assert comparison["overall"]["delta_correct_test_outputs"] == 0
    assert comparison["overall"]["candidate_only_solved"] == 1
    assert comparison["overall"]["control_only_solved"] == 1
    assert comparison["folds"]["0"]["delta_correct_test_outputs"] == 1
    assert comparison["folds"]["1"]["delta_correct_test_outputs"] == -1
    assert comparison["gates"]["minimum_fold_loss_at_most_one"] is True
    assert comparison["gates"]["positive_pooled_gain"] is False


def test_compare_fold_reports_rejects_different_fold_assignments() -> None:
    base = {
        "n_splits": 2,
        "seed": 42,
        "fold_assignments": {"task-a": 0, "task-b": 1},
        "overall": {
            "per_task_detail": {
                "task-a": {"test_results": [True]},
                "task-b": {"test_results": [False]},
            }
        },
    }
    candidate = {
        **base,
        "fold_assignments": {"task-a": 1, "task-b": 0},
    }

    with pytest.raises(ValueError, match="same fold assignments"):
        compare_fold_reports(base, candidate)
