# Agent Guidelines: ARC Prize 2026 - ARC-AGI-2

## Competition Essence
- **Benchmark:** ARC-AGI-2 (Abstraction and Reasoning Corpus)
- **Goal:** Predict exact 2D output grids (integers 0-9, dimensions 1x1 to 30x30) for test inputs given 2 to 5 demonstration pairs per task.
- **Attempts:** Up to 2 attempts (`attempt_1`, `attempt_2`) per test output grid.
- **Scoring:** Binary exact-match score per test instance (1 if attempt 1 or attempt 2 matches ground truth, 0 otherwise). Final score is average across all test instances.

---

## Engineering & Code Standards
- **Python Version:** Python 3.10+
- **Code Style:** Type annotations for all public functions, formatted via `ruff`.
- **Testing:** Unit tests in `tests/`, verified with `pytest`.
- **Reproducibility:** Fix random seeds using `src/config.py` (`SEED = 42`).
- **Memory & Resource Discipline:**
  - ARC solver loops process hundreds of tasks sequentially.
  - Call `gc.collect()` and clear accelerator cache after each task.
  - Implement per-task timeouts (default: 60-120 seconds) to prevent infinite loops.

---

## Validation Gate Mandate
1. **Never rely on the public leaderboard alone.**
2. All solver improvements, DSL primitives, or model changes must be evaluated against the official local evaluation set (`input/arc-agi_evaluation_challenges.json` + `input/arc-agi_evaluation_solutions.json`) or a 5-fold cross-validation split on `input/arc-agi_training_challenges.json`.
3. Compute score exclusively using `src/metrics.py`.

---

## Artifact Contract
- **Trained Models / Weights:** `models/<model_name>/`
- **Validation / Evaluation Outputs:** `oof/eval_<model_name>_<score>.json`
- **Submissions:** `submissions/submission_<model_name>_<score>.json`
- **Every submission MUST pass verification via `src/validator.py` before submission.**

---

## Experiment Tracking
Log every experiment, validation score, and architectural hypothesis in `docs/experiments.md`.

---

## Kaggle Cloud Execution & Submission Protocol
1. **Hardware Environment:** 4x NVIDIA L4 GPUs (96 GB VRAM) on BYOD Python 3.11 image (`gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868`).
2. **Deficit-First Scheduling:** In test runs, Tier-1 solved tasks receive a 90s budget for attempt_2, reserving 720s for unsolved hard tasks.
3. **Daily Limit Discipline:** Exactly 1 submission per day (resets 00:00 UTC). Always verify remaining quota via `kaggle competitions submission-limits -c arc-prize-2026-arc-agi-2` before submitting.
4. **Git Remote Push:** Always prefix with `GIT_SSH_COMMAND="ssh"`: `GIT_SSH_COMMAND="ssh" git push origin master`.
