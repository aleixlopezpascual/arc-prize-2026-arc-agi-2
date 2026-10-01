from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from src import config
from src.metrics import compute_competition_metric
from src.models.baseline_heuristics import BaselineHeuristicSolver
from src.models.hybrid_solver import HybridSolver
from src.validator import validate_submission_dict


def run_evaluation(solver_name: str = "hybrid") -> dict:
    if solver_name == "hybrid":
        solver = HybridSolver()
        exp_id = "hybrid_01"
    else:
        solver = BaselineHeuristicSolver()
        exp_id = "baseline_01"

    print(f"=== Running Evaluation for [{exp_id}] ({solver.__class__.__name__}) ===")
    print(f"Loading evaluation challenges from {config.EVAL_CHALLENGES_PATH}...")
    eval_chal = json.loads(config.EVAL_CHALLENGES_PATH.read_text())
    eval_sol = json.loads(config.EVAL_SOLUTIONS_PATH.read_text())

    t0 = time.time()
    preds = solver.predict_all(eval_chal)
    elapsed = time.time() - t0

    # Validate output schema
    is_valid, errors = validate_submission_dict(preds, eval_chal)
    if not is_valid:
        raise ValueError(f"Validation failed with {len(errors)} errors: {errors[:5]}")
    print(f"[+] Predictions verified: 100% compliant with schema across {len(preds)} tasks.")

    # Compute metric
    results = compute_competition_metric(preds, eval_sol)
    print(f"[+] Evaluation finished in {elapsed:.2f}s:")
    print(f"    - Official Accuracy (per test output): {results['accuracy']:.4f} ({results['correct_test_outputs']}/{results['total_test_outputs']})")
    print(f"    - Task Accuracy (all tests per task):  {results['task_accuracy']:.4f} ({results['fully_solved_tasks']}/{results['total_tasks']})")
    print(f"    - Solved Tasks: {results['solved_task_ids']}")

    # Save to oof
    oof_path = config.OOF_DIR / f"eval_{exp_id}_{results['accuracy']:.4f}.json"
    oof_path.write_text(json.dumps(preds))
    print(f"[+] Saved evaluation predictions to {oof_path}")

    # Generate submission on test challenges
    test_chal = json.loads(config.TEST_CHALLENGES_PATH.read_text())
    test_preds = solver.predict_all(test_chal)
    is_valid_test, test_errors = validate_submission_dict(test_preds, test_chal)
    if not is_valid_test:
        raise ValueError(f"Test submission validation failed: {test_errors[:5]}")

    sub_path = config.SUBMISSION_DIR / f"submission_{exp_id}_{results['accuracy']:.4f}.json"
    sub_path.write_text(json.dumps(test_preds))
    print(f"[+] Saved verified test submission to {sub_path}")

    # Also write default submission.json in SUBMISSION_DIR
    default_sub = config.SUBMISSION_DIR / "submission.json"
    default_sub.write_text(json.dumps(test_preds))
    print(f"[+] Saved default submission to {default_sub}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["hybrid", "baseline"], default="hybrid")
    args = parser.parse_args()
    run_evaluation(solver_name=args.solver)
