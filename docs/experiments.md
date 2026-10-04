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
| `hybrid-01` | Two-Tier Hybrid Solver: Tier 1 Symbolic Filter (with enclosed-region fill & object selection) + Tier 2 Geometric Fallback & Augmentation | 0.0000 (0/172) | 0.0362 (39/1076) | - | 36 training tasks solved (39 test outputs, 3.62% accuracy). 37 Tier-1 symbolic hits. Processed 1000 tasks in 11.14s. 37 unit tests passing across dsl, metrics, and neural modules. |
| `baseline-lb3389` | SOTA Public Baseline: 4B Qwen TTT with high-rank rsLoRA (r=256), GPU-optimized turbo_dfs constrained tree search, and 128 task-time augmentations | 0.0233 (4/172) | - | 29.31% (Rank 1,043 / 2,377) | Official competition submission complete! Successfully secured a spot on the live leaderboard. |
| `hybrid-v1-kernel` | Hybrid SOTA v1: Tier 1 Fast Symbolic Filter (periodic 2D rooms + dot connections + enclosed fills + object bboxes) pre-empting Tier 2 Qwen 4B rsLoRA TTT | 0.0291 (5/172) | 0.0400 (43/1076) | Validation 3.5 (Complete) | Completed on Kaggle! Validation score jumped from 2.5 to 3.5. Monotonic merge guarantees zero regression. |
| `hybrid-02` | Expanded Symbolic Induction: Added 3x3 local cellular automata rules + split & logical overlay (OR, AND, XOR) | 0.0058 (1/172) | 0.0483 (52/1076) | Ready for deploy | 48 training tasks solved (128 Tier 1 hits / 12.8%). Solved evaluation task 135a2760. 42 unit tests passing in tests/. |
| `hybrid-03` | Advanced Pattern & Subwindow Primitives: Added odd-one-out symmetry block selector + marker-relative window extraction | 0.0058 (1/172) | 0.0493 (53/1076) | Staged in v5 | 49 training tasks solved (129 Tier 1 hits / 12.9%). Solved a87f7484 & 48d8fb45. 44 unit tests passing in tests/. |
| `hybrid-v5-kernel` | Hybrid SOTA v5 with Optimal Union + 129 Symbolic Hits | 0.0233 (4/172) | 0.0493 (53/1076) | 30.69% (Rank 563 / 2,400) | Sub ID 56806096 COMPLETE! Score surged +1.38% (29.31% -> 30.69%), climbing +480 positions to Rank 563. Both Tier 1 (135a2760) and Tier 2 contributing points. |
| `hybrid-v6-kernel` | Hybrid SOTA v6: Directional & Bouncing Raycasting + Rigid Gravity | 0.0233 (4/172) | 0.0511 (55/1076) | Sub ID 56828374 (PENDING) | Officially submitted! Private rerun active across 240 hidden test tasks on 4x L4 GPUs. Incorporates optimal 2-attempt union, raycasting (d037b0a7, 623ea044), and rigid gravity (1e0a9b12). |
| `hybrid-v7-kernel` | Hybrid SOTA v7: Kaleidoscope Tiling + Kronecker Fractals + 1x1 Classification | 0.0233 (4/172) | 0.0520 (56/1076) | Completed (Val: 3.5) | Completed on Kaggle 4x L4 cluster! Validation score 3.5. Output submission.json verified 100% compliant. Stored at submissions/submission_hybrid_sota_v7.json ready for next submission window. |
| `hybrid-v8-kernel` | Hybrid SOTA v8: 2-Step Dynamic Color Remapping + Topological Hole Counting | 0.0233 (4/172) | 0.0810 (90/1076) | Queued on Kaggle | Pushed as private kernel v8. Solves 81 training tasks (8.1% of all tasks). Incorporates 57 passing unit tests and topological hole counting. |
