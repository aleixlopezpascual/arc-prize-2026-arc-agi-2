from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.evaluate_neural_cv import evaluate_files


def test_evaluate_files_writes_five_fold_metrics_from_official_metric(tmp_path: Path) -> None:
    task_ids = [f"task-{index}" for index in range(5)]
    challenges = {
        task_id: {"train": [], "test": [{"input": [[0]]}]}
        for task_id in task_ids
    }
    solutions = {task_id: [[[index]]] for index, task_id in enumerate(task_ids)}
    predictions = {
        task_id: [{"attempt_1": [[index]], "attempt_2": [[9]]}]
        for index, task_id in enumerate(task_ids)
    }
    challenge_path = tmp_path / "challenges.json"
    solution_path = tmp_path / "solutions.json"
    prediction_path = tmp_path / "predictions.json"
    output_path = tmp_path / "cv-report.json"
    challenge_path.write_text(json.dumps(challenges), encoding="utf-8")
    solution_path.write_text(json.dumps(solutions), encoding="utf-8")
    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")

    report = evaluate_files(
        prediction_path,
        challenge_path,
        solution_path,
        output_path,
        n_splits=5,
        seed=42,
    )

    assert report["overall"]["accuracy"] == 1.0
    assert report["overall"]["total_test_outputs"] == 5
    assert set(report["folds"]) == {"0", "1", "2", "3", "4"}
    assert all(fold["metrics"]["accuracy"] == 1.0 for fold in report["folds"].values())
    assert set(report["fold_assignments"]) == set(task_ids)
    assert report["folding_unit"] == "task_id"
    assert json.loads(output_path.read_text(encoding="utf-8")) == report


def test_evaluate_files_rejects_incomplete_predictions(tmp_path: Path) -> None:
    challenges = {"task-a": {"train": [], "test": [{"input": [[0]]}]}}
    solutions = {"task-a": [[[1]]]}
    predictions = {}
    challenge_path = tmp_path / "challenges.json"
    solution_path = tmp_path / "solutions.json"
    prediction_path = tmp_path / "predictions.json"
    output_path = tmp_path / "cv-report.json"
    challenge_path.write_text(json.dumps(challenges), encoding="utf-8")
    solution_path.write_text(json.dumps(solutions), encoding="utf-8")
    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")

    with pytest.raises(ValueError, match="prediction validation failed"):
        evaluate_files(
            prediction_path,
            challenge_path,
            solution_path,
            output_path,
            n_splits=2,
            seed=42,
        )
    assert not output_path.exists()
