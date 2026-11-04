# Scope: neural-only train run + held-out gating (2026-10-29)

Goal: replace the contaminated 120-task eval as the promotion gate with a held-out
measurement of whether symbolic output helps or hurts versus the neural guess.
See `local_vs_lb.md` for why (local eval vs LB r = -0.86; 42/44 eval solves come from
single-task primitives).

## State of the tooling (already built and committed)

| Piece | Where | Status |
|---|---|---|
| Neural-only validation output (no Tier-1 pre-pass) | `build_submission_notebook --validation-mode train --no-symbolic-merge` | done, tested |
| Subset restriction to the 186 train tasks where a level-2 DSL program fits | `--task-ids-file docs/intel/train_symbolic_fit_task_ids.json` | done, tested |
| Kernel as pushed | `notebooks/neural_only_train/` (`aleixlopez/arc-agi2-neural-only-train-subset`, v1) | **run status unknown** (Kaggle API blocked from the sandbox) |
| A/B scorer: neural / production / union / union_no_ca / union_loo, paired gains/losses | `src/tools/heldout_proxy.py` | done, 9 tests, refuses hybrid files (attempt_1 == S on >= 95%) |
| Gate inside the shipped notebook | - | **not built**: LOO family gating exists only in the proxy |

## Steps

1. **User (Kaggle CLI):** check the kernel and fetch its output.
   ```bash
   kaggle kernels status aleixlopez/arc-agi2-neural-only-train-subset
   kaggle kernels output aleixlopez/arc-agi2-neural-only-train-subset -p $TMPDIR/neural_only
   ```
   Expected file: `validation_predictions_neural_only.json`, `{task_id: [{attempt_1, attempt_2}, ...]}`.
   If v1 timed out or failed, rerun; the 186-task subset was chosen to fit the 12h shared deadline.
2. **Score:** `python3 -m src.tools.heldout_proxy --split train --neural <that file>`
   (about 30s; symbolic cache is built on first run). Output: policy table with gains/losses vs neural-only,
   per-family S / N1 / S_only / N2_only counts, and `oof/heldout_proxy_report_train.json`.
3. **Noise check before trusting a delta:** one repeat of the kernel with a different seed (or a bootstrap over
   tasks in the proxy). Only deltas that exceed run-to-run variation count.
4. **Decide the gate** from the result (rule below), then implement it in the notebook builder as a
   one-factor change, with a unit test.

## Decision rule (fixed before seeing results)

Per symbolic family, symbolic top-1 may take attempt 1 only if, on held-out train:
- leave-one-task-out precision >= 0.5 (existing `LOO_THRESHOLD`), and
- family has >= 3 fired tasks (below that, it is neural-first), and
- family net (S_only wins minus N1 wins displaced) >= 0.

Everything else keeps neural attempt 1 and may use symbolic as attempt 2 only if it is structurally different.
Promote the gated policy only if `union_loo` net >= 0 vs neural and strictly beats `production` on the same tasks.
If `production` net is negative, the shipped policy is hurting and the gate ships regardless of eval score.
Report results via `src/metrics.py` only; log in `docs/experiments.md`.

## Caveats to state with any result

- **Power:** 186 tasks, roughly 200 outputs; expect single-digit gains/losses. Treat differences of 1-2 outputs as noise
  (identical v25 kernel moved 23.47 -> 25.56 on the LB).
- **Bias direction:** the base model (`sorokin/qwen3_4b_grids15_sft139`) was trained on ARC training data, so neural accuracy on
  train is optimistic. That makes this test conservative for symbolic gains and lenient for symbolic losses; a family that only
  loses here is a safe cut, a family that wins here is real evidence.
- **Subset selection:** only tasks where a program fits the demos are run, which is exactly the population where the symbolic
  override fires; neural-only accuracy on all 1000 tasks is not measured and is not needed for this A/B.
- **Fit is not generalization:** primitives written for a single eval task have no train fire rate, so this run says nothing about
  them; they remain unvalidated. The gate's default for zero-train-evidence families is neural-first.
- Public LB stays the final check: one submission/day, and not for deltas under ~2 pts.

## Out of scope for this step

Directional-symmetry gate for D4 augmentation, kappa scheduling replay, and any new primitives. They are queued behind this result.
