import pytest
import numpy as np

from src.metrics import (
    grid_equal,
    score_single_test_pair,
    score_task,
    compute_competition_metric,
)
from src.validator import validate_grid, validate_submission_dict


def test_grid_equal():
    g1 = [[1, 2], [3, 4]]
    g2 = [[1, 2], [3, 4]]
    g3 = [[1, 2], [3, 5]]
    g4 = [[1, 2, 0], [3, 4, 0]]
    arr1 = np.array(g1)

    assert grid_equal(g1, g2)
    assert grid_equal(g1, arr1)
    assert not grid_equal(g1, g3)
    assert not grid_equal(g1, g4)
    assert not grid_equal(g1, None)
    assert not grid_equal(None, None)


def test_score_single_test_pair():
    target = [[1, 2], [3, 4]]
    wrong = [[0, 0], [0, 0]]

    # Match on attempt_1
    assert score_single_test_pair({"attempt_1": target, "attempt_2": wrong}, target)
    # Match on attempt_2
    assert score_single_test_pair({"attempt_1": wrong, "attempt_2": target}, target)
    # Both match
    assert score_single_test_pair({"attempt_1": target, "attempt_2": target}, target)
    # Both fail
    assert not score_single_test_pair({"attempt_1": wrong, "attempt_2": wrong}, target)
    # List format
    assert score_single_test_pair([wrong, target], target)


def test_score_task_multi_test():
    target1 = [[1, 1]]
    target2 = [[2, 2]]

    # Task with 2 test pairs, 1 correct
    task_preds = [
        {"attempt_1": target1, "attempt_2": [[0, 0]]},
        {"attempt_1": [[0, 0]], "attempt_2": [[0, 0]]},
    ]
    res = score_task(task_preds, [target1, target2])
    assert res["n_tests"] == 2
    assert res["n_correct"] == 1
    assert not res["fully_solved"]
    assert res["test_results"] == [True, False]

    # Both correct
    task_preds_all_correct = [
        {"attempt_1": target1, "attempt_2": [[0, 0]]},
        {"attempt_1": target2, "attempt_2": [[0, 0]]},
    ]
    res2 = score_task(task_preds_all_correct, [target1, target2])
    assert res2["n_correct"] == 2
    assert res2["fully_solved"]


def test_compute_competition_metric():
    solutions = {
        "task_a": [[[1]]],
        "task_b": [[[2]], [[3]]],
    }

    # All correct: 3 total test outputs
    preds_all = {
        "task_a": [{"attempt_1": [[1]], "attempt_2": [[0]]}],
        "task_b": [
            {"attempt_1": [[0]], "attempt_2": [[2]]},
            {"attempt_1": [[3]], "attempt_2": [[0]]},
        ],
    }
    metric_all = compute_competition_metric(preds_all, solutions)
    assert metric_all["accuracy"] == 1.0
    assert metric_all["correct_test_outputs"] == 3
    assert metric_all["total_test_outputs"] == 3
    assert metric_all["fully_solved_tasks"] == 2
    assert metric_all["task_accuracy"] == 1.0

    # Partial correct: 1 out of 3
    preds_partial = {
        "task_a": [{"attempt_1": [[1]], "attempt_2": [[0]]}],
        "task_b": [
            {"attempt_1": [[9]], "attempt_2": [[8]]},
            {"attempt_1": [[7]], "attempt_2": [[6]]},
        ],
    }
    metric_partial = compute_competition_metric(preds_partial, solutions)
    assert pytest.approx(metric_partial["accuracy"], 1e-4) == 1 / 3
    assert metric_partial["correct_test_outputs"] == 1
    assert metric_partial["fully_solved_tasks"] == 1  # task_a is solved
    assert metric_partial["solved_task_ids"] == ["task_a"]


def test_validate_grid():
    valid_grid = [[0, 1, 2], [3, 4, 5]]
    ok, _ = validate_grid(valid_grid)
    assert ok

    # Empty
    ok, msg = validate_grid([])
    assert not ok

    # Ragged
    ok, msg = validate_grid([[1, 2], [1]])
    assert not ok and "Inconsistent row length" in msg

    # Out of range color (>9)
    ok, msg = validate_grid([[10]])
    assert not ok and "out of bounds" in msg

    # Dimension too large (>30)
    big_grid = [[0] * 31]
    ok, msg = validate_grid(big_grid)
    assert not ok and "exceeds maximum 30" in msg


def test_validate_submission_dict():
    challenges = {
        "task_1": {"train": [], "test": [{"input": [[1]]}]},
        "task_2": {"train": [], "test": [{"input": [[2]]}, {"input": [[3]]}]},
    }
    valid_sub = {
        "task_1": [{"attempt_1": [[1]], "attempt_2": [[0]]}],
        "task_2": [
            {"attempt_1": [[2]], "attempt_2": [[0]]},
            {"attempt_1": [[3]], "attempt_2": [[0]]},
        ],
    }
    ok, errors = validate_submission_dict(valid_sub, challenges)
    assert ok, f"Expected valid, got errors: {errors}"

    # Missing task
    bad_sub = {"task_1": valid_sub["task_1"]}
    ok, errors = validate_submission_dict(bad_sub, challenges)
    assert not ok
    assert any("Missing 1 tasks" in e for e in errors)

    # Missing attempt_2
    bad_sub_attempt = {
        "task_1": [{"attempt_1": [[1]]}],
        "task_2": valid_sub["task_2"],
    }
    ok, errors = validate_submission_dict(bad_sub_attempt, challenges)
    assert not ok
    assert any("missing key 'attempt_2'" in e for e in errors)
