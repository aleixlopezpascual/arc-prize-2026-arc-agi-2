# Adaptive Pass Baseline & Uncertainty Testing Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Goal
Establish a reproducible neural cross-validation pipeline to measure baseline parity between the public NVARC baseline and our hybrid, and subsequently evaluate a one-factor uncertainty-targeted Pass B mechanism and length-bucketed DFS batching without relying on Kaggle submissions for feedback.

## Architecture
- **Neural/Symbolic Hybrid:** The builder (`src/tools/build_submission_notebook.py`) merges local symbolic code with the exact NVARC neural baseline parent (`notebooks/public_baselines/lb33_89/arc-agi2-lb33-89-minimal-perfpatch.ipynb`).
- **Pass A & B Runner:** Orchestrated within the builder template (starter source around lines 308-410; multi-pass runner around 421-455), utilizing Qwen3-4B rsLoRA and GPU ARC-token logit gather.
- **Validation Route:** Requires a dedicated neural validation runner since `src/evaluate.py` currently only supports local `BaselineHeuristicSolver` or symbolic `HybridSolver`.

## Tech Stack
- Python, Qwen3-4B/NVARC (via Kaggle/local), `src/metrics.py`, `src/evaluate.py`, `src/tools/build_submission_notebook.py`.

## Spec
- `docs/competition-brief.md#4-cross-validation-validation-gate`
- `src/metrics.py`
- `src/validator.py`
- `src/evaluate.py`
- `src/tools/build_submission_notebook.py`
- `notebooks/public_baselines/lb33_89/arc-agi2-lb33-89-minimal-perfpatch.ipynb`
- `docs/experiments.md`

## Global Constraints
- **Validation Route:** A full neural train/eval route and task-ID grouped scoring tools are now implemented in `src/tools/build_submission_notebook.py`, `src/neural/cv_folds.py`, and `src/tools/evaluate_neural_cv.py`. Private predictions were generated and scored. The intended Pass-B policy comparison remains unresolved because Pass B did not start in either completed run.
- **Never Substitute:** Do not substitute local heuristic evaluator, Kaggle public score, or test submission predictions for true neural CV evidence.
- **Historical snapshot (2026-10-21 18:54 CEST):** v25 ref57475078 pending/no score; v24 ref57439702 complete 22.78%; historical public best 30.97% refs 56836175/56875914; v24 local official eval 9.30% (16/172), cluster score 14.0.
- **Active v25 State:** Code already implemented includes Qwen3-4B/NVARC lineage, GPU ARC-token logit gather, cheap-first, score_ensemble, symbolic union, Pass A seed1 + broad Pass B seed137 when >15min remains.
- **Gate:** Positive pooled 5-fold exact-match gain, nonnegative gain on at least 4/5 folds, no fold loses more than one output, official-eval confirmation does not regress, and <=12-hour budget. Equal score with less compute is an efficiency finding, not accuracy gain.
- **Safety:** No Kaggle submission without explicit approval and quota check. Preserve pre-existing `auto_submit.log`. Do not present Pass-A-only preflight scores as evidence for the Pass-B policy.

## Current Execution Status (updated 2026-10-25 14:22 UTC)
- The private broad control (kernel v2) and uncertainty-route run (kernel v1) completed. Both produced complete predictions for 1,000 tasks / 1,076 test outputs; the outputs passed `src.validator` and were scored with `src.metrics`.
- The five folds use the same deterministic SHA-256 task-ID assignment (seed 42), with 200 tasks per fold and 217, 218, 221, 211, and 209 test outputs. This is task-level inference grouped for reporting, not five separate global model fits; the worker restores the base adapter per task.
- **The intended Pass-B comparison did not occur.** Pass A ended with -1.6 minutes (broad) and -1.3 minutes (uncertainty) remaining; the runner only starts Pass B when more than 15 minutes remain. Neither log contains a Pass-B start. The candidate did write an uncertainty map for 759 task IDs, but no uncertainty-prioritized Pass B consumed it.
- Measured Pass-A-only outputs: broad 850/1076 (78.9963%); uncertainty-route run 831/1076 (77.2305%). Paired outcomes were 3 candidate-only solves and 22 control-only solves; fold deltas were -3, -1, -4, -7, -4. These values are recorded in `docs/experiments.md` as a preflight, not as a causal Pass-B result or promotion decision.
- A paired runner is implemented and locally tested in `build(..., validation_mode="train", pass_b_policy="paired")`: it caps shared Pass A, freezes that output directory, computes uncertainty once, then runs broad and uncertainty Pass B with equal budgets and distinct output directories. Private kernel v1 completed Pass A (uncertainty map: 545 task IDs), then errored during broad Pass B. The log reports `AttributeError: module 'UnslothGRPOTrainer' has no attribute 'UnslothGRPOTrainer'` on import. Likely cause: persistent `/kaggle/worker0..3` barrier files from Pass A cause Pass-B ranks to skip serialization and race through Unsloth monkey-patching. This is strongly supported by Pass-A's staggered imports and Pass-B's near-simultaneous starts, but still needs a regression test/rerun to confirm.
- Authorized v25 rerun submitted as ref `57609744` using production kernel version 25; current status is `PENDING`. Today's one-submission allowance is consumed. This repeats the v25 method and is not an experiment on the new public contributions.

## Review Focus
- Ensure the neural validation gap is fully addressed before testing the targeted Pass B.
- Confirm exact-match gates are strictly followed and not bypassed.
- Verify separation between code already implemented (v25 state) and planned validation tasks.

---

## Stage 1: Wait and Refresh
**Files/Artifacts:** `docs/experiments.md`
**Entry Criteria:** Task starts.
**Exit Criteria:** v25 status confirmed.
- [ ] Monitor and await completion of pending v25 (ref57475078).
- [ ] Verify no duplicate runs are queued or triggered.

## Stage 2: Lock Anchors & Inspect Paths
**Files/Artifacts:** `src/tools/build_submission_notebook.py`, `notebooks/public_baselines/lb33_89/arc-agi2-lb33-89-minimal-perfpatch.ipynb`
**Entry Criteria:** Stage 1 completes.
**Exit Criteria:** Anchors logged, builder injection paths identified (Pass-A/Pass-B around lines 308-410 and 421-455).
- [ ] Lock public and local anchor metrics (v24/v25 states vs historical).
- [ ] Inspect the exact parent path in `src/tools/build_submission_notebook.py` referencing `notebooks/public_baselines/lb33_89/arc-agi2-lb33-89-minimal-perfpatch.ipynb`.

## Stage 3: Resolve Neural Validation Gap
**Files/Artifacts:** Local evaluation scripts, Kaggle configuration (if needed)
**Entry Criteria:** Stage 2 completes.
**Exit Criteria:** Traceable full NVARC predictions/weights available, or execution paused for user approval if Kaggle compute is required.
- [x] Trace and establish a reproducible full NVARC validation route using the exact neural model/code (since `src/evaluate.py` only handles symbolic/heuristic).
- [x] Obtain explicit approval before using private Kaggle kernels; keep all kernels private and make no competition submission.

## Stage 4: Baseline Parity on 5-fold CV
**Files/Artifacts:** CV folds setup, evaluation output.
**Entry Criteria:** Stage 3 completes (NVARC evaluation capability achieved).
**Exit Criteria:** 5-fold parity run completes and is evaluated.
- [ ] Run a baseline-parity comparison between the bare parent NVARC notebook and our current hybrid.
- [ ] Use an identical task-ID-grouped 5-fold training split.
- [ ] Ensure fixed seeds, same compute budget, and exact official metric evaluated via `src/metrics.py`.
- [ ] Run the 120-task official evaluation only as a predeclared confirmation (do not use for tuning).

## Stage 5: Uncertainty-Targeted Pass B vs Broad Pass B
**Files/Artifacts:** Pass B implementation modifications.
**Entry Criteria:** Stage 4 completes (baseline parity verified) and a shared Pass-A artifact plus reserved Pass-B budgets are implemented.
**Exit Criteria:** Pass B variant evaluated against strict exact-match gates.
- [x] Implement/test the deterministic uncertainty score and ranking helper, plus the validation-only builder hook; keep default submission behavior unchanged.
- [x] Implement and locally verify a paired validation runner that freezes one Pass-A candidate set and gives broad and uncertainty Pass B equal, reserved budgets.
- [x] Push the paired private run after GPU quota became available; no competition submission was made.
- [x] Inspect the v1 error logs and identify the stale-worker-barrier/import-race hypothesis.
- [ ] Add a regression test for sequential runner invocations with pre-existing worker markers; isolate or clear markers before spawning each arm.
- [ ] Rerun the private pair; verify explicit start markers and nonempty outputs for both Pass-B arms before scoring.
- [ ] Calibrate uncertainty solely on training folds based on likelihood-vote disagreement and top-candidate margin.
- [ ] Validate against the strict exact-match gain gates (positive pooled 5-fold, >=0 on 4/5 folds, <=1 output lost per fold, official-eval confirmation no regression, <=12-hour budget).

## Stage 6: Length-Aware Batching
**Files/Artifacts:** Batching modifications.
**Entry Criteria:** Stage 5 completes and passes validation gates.
**Exit Criteria:** Length-bucketed DFS evaluated against exact-match gates.
- [ ] Only after Stage 5 is fully measured and validated, test length-bucketed DFS batching as a separate, isolated ablation.

## Stage 7: Reporting
**Files/Artifacts:** `docs/experiments.md`
**Entry Criteria:** Stages 4-6 experiment outputs available.
**Exit Criteria:** Measured outcomes recorded accurately; no unmeasured claims.
- [x] Record the measured Pass-A-only preflight results and explicitly mark them invalid for the intended Pass-B comparison.
- [ ] Record a valid paired Pass-B comparison and official-evaluation confirmation only after those runs complete.
- [ ] Confirm no unmeasured results are written and no Kaggle submission is triggered without approval.

## Follow-up Backlog from Public Contribution Review (2026-10-24)

**Gate:** do not combine these ideas with the paired broad-vs-uncertainty Pass-B comparison. First inspect/fix the private kernel v1 error and obtain a complete paired result; keep the baseline-parity gate in Stage 4 visible. Detailed evidence is in `docs/intel/top_notebooks.md#recent-public-contribution-snapshot`. These are hypotheses, not experiment results.

1. **P1 — Offline replay of online-kappa scheduling.** Estimate actual/predicted training and decode costs from existing task logs, then simulate budget reallocation over the current cheap-first order. Continue only if the replay predicts more completed/solved tasks without starving the tail. If the logs do not contain enough timing detail, add logging only to a later approved validation run; do not change the current paired runner.
2. **P2 — Structurally diverse attempt 2.** On frozen validation candidate pools, preserve attempt 1 and compare the current second-choice rule with one that prefers a sufficiently different candidate when the current runner-up is near-duplicate. Score the same task-ID folds through `src/metrics.py`; reject the idea if it adds no net exact matches or hides losses on a fold.
3. **P3 — Depth-4 guided DSL beam.** Run a CPU-only, capped screen against the current one-/two-step DSL on training tasks. Keep the change only if it solves new tasks under a declared time budget; then add it as a separate CV ablation. The public examples show no positive public-score evidence.
4. **Deprioritize demonstration-consistency reranking.** The reviewed report shows more losses than wins on the actual candidate set. Do not replace `score_ensemble` or use the score for task priority without new, isolated evidence on our exact candidate stream.
5. **Defer SOAR-7B porting.** Resolve model availability, metadata/source lineage, and the default-off execution gate before considering an experiment; do not infer execution from the notebook title.