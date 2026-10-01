from __future__ import annotations

from typing import Any, Sequence
import numpy as np


GridType = Sequence[Sequence[int]] | np.ndarray


def grid_equal(g1: GridType | None, g2: GridType | None) -> bool:
    """Checks exact dimensional and element-wise equality between two ARC grids."""
    if g1 is None or g2 is None:
        return False

    # Convert to standard numpy array if not already
    arr1 = np.asarray(g1, dtype=int)
    arr2 = np.asarray(g2, dtype=int)

    if arr1.shape != arr2.shape:
        return False

    return bool(np.array_equal(arr1, arr2))


def score_single_test_pair(
    attempts: dict[str, GridType] | Sequence[GridType],
    target: GridType,
) -> bool:
    """Evaluates whether either attempt_1 or attempt_2 exactly matches target."""
    if isinstance(attempts, dict):
        candidate_list = [attempts.get("attempt_1"), attempts.get("attempt_2")]
    elif isinstance(attempts, (list, tuple)):
        candidate_list = list(attempts)
    else:
        return False

    for cand in candidate_list:
        if cand is not None and grid_equal(cand, target):
            return True
    return False


def score_task(
    task_preds: list[dict[str, Any]],
    task_solutions: list[GridType],
) -> dict[str, Any]:
    """Scores a single task across all its test pairs.

    Returns:
        Dictionary with:
            - n_tests: total test pairs
            - n_correct: number of solved test pairs
            - fully_solved: True if all test pairs in task are solved
            - test_results: boolean list per test pair
    """
    n_tests = len(task_solutions)
    if len(task_preds) < n_tests:
        # Malformed or incomplete predictions
        n_preds = len(task_preds)
    else:
        n_preds = n_tests

    test_results: list[bool] = []
    for i in range(n_tests):
        if i < n_preds:
            passed = score_single_test_pair(task_preds[i], task_solutions[i])
        else:
            passed = False
        test_results.append(passed)

    n_correct = sum(test_results)
    return {
        "n_tests": n_tests,
        "n_correct": n_correct,
        "fully_solved": n_correct == n_tests and n_tests > 0,
        "test_results": test_results,
    }


def compute_competition_metric(
    predictions: dict[str, list[dict[str, Any]]],
    solutions: dict[str, list[GridType]],
) -> dict[str, Any]:
    """Computes the exact official Kaggle ARC-AGI-2 competition metric.

    Formula:
        Total correct test outputs / Total test outputs

    Args:
        predictions: dict mapping task_id to list of attempt dicts:
                     [{'attempt_1': grid, 'attempt_2': grid}, ...]
        solutions: dict mapping task_id to list of ground truth grids:
                   [grid_1, grid_2, ...]

    Returns:
        dict containing:
            - accuracy: float in [0.0, 1.0] (the official leaderboard metric)
            - total_test_outputs: total count of test outputs across all tasks
            - correct_test_outputs: count of test outputs solved in <= 2 attempts
            - total_tasks: count of tasks evaluated
            - fully_solved_tasks: count of tasks where every test output was solved
            - task_accuracy: float in [0.0, 1.0] (fraction of fully solved tasks)
            - solved_task_ids: list of fully solved task IDs
            - per_task_detail: detailed score dict per task_id
    """
    total_test_outputs = 0
    correct_test_outputs = 0
    fully_solved_tasks = 0
    solved_task_ids: list[str] = []
    per_task_detail: dict[str, Any] = {}

    for task_id, task_solutions in solutions.items():
        task_preds = predictions.get(task_id, [])
        res = score_task(task_preds, task_solutions)

        total_test_outputs += res["n_tests"]
        correct_test_outputs += res["n_correct"]
        if res["fully_solved"]:
            fully_solved_tasks += 1
            solved_task_ids.append(task_id)

        per_task_detail[task_id] = res

    accuracy = (
        correct_test_outputs / total_test_outputs if total_test_outputs > 0 else 0.0
    )
    task_accuracy = (
        fully_solved_tasks / len(solutions) if len(solutions) > 0 else 0.0
    )

    return {
        "accuracy": accuracy,
        "total_test_outputs": total_test_outputs,
        "correct_test_outputs": correct_test_outputs,
        "total_tasks": len(solutions),
        "fully_solved_tasks": fully_solved_tasks,
        "task_accuracy": task_accuracy,
        "solved_task_ids": solved_task_ids,
        "per_task_detail": per_task_detail,
    }
