# Experiment Ledger: ARC Prize 2026 - ARC-AGI-2

## Validation Standard
- **Official Evaluation Split:** 120 tasks, 172 test outputs (`input/arc-agi_evaluation_challenges.json`).
- **Metric:** Percentage of exact-match test outputs in $\le 2$ attempts (`src/metrics.py`).
- **Submission Schema:** Checked with `src/validator.py`.

---

## Log

| Exp ID | Model / Strategy | Local CV (Eval) | Train CV | Public LB | Notes / Findings |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline-01` | Geometric transforms (rot/flip/transpose) + 1-to-1 color mapping | 0.0000 (0/172) | 0.0102 (11/1076) | - | End-to-end evaluation harness and submission verification pipeline verified. 11 training tasks solved by simple symmetry/color-swap; 0 evaluation tasks solved, confirming ARC-AGI-2 requirement for multi-step program synthesis or deep priors. |
| `dsl-01` | Symbolic DSL & Object Perception Engine (connected components, gravity, content crop, symmetries, 1- & 2-step search) | 0.0000 (0/172) | 0.0269 (29/1076) | - | 28 training tasks solved (2.5x increase over baseline-01). 22 unit tests passing in tests/test_dsl.py. Instant execution (~0.1s total). |
