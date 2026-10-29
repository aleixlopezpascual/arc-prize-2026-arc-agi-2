# Does local eval predict the public LB? (2026-10-29)

Script: `python3 src/tools/local_vs_lb.py` (data hard-coded from `docs/experiments.md` local eval column and `auto_submit.log` public scores; v25 first run, v5-v24 only, n=16).

## Result

| Metric | Value |
|---|---|
| Pearson r (local eval % vs public LB %) | **-0.86** |
| Spearman r | **-0.88** |
| LB change per +1 pt local eval | -0.88 |
| Local eval 1.16% -> 9.30%, LB 30.69% -> 22.78% | monotone-ish decline |

Local eval has been **anti-correlated** with the LB across v5-v24.

## Caveats (do not over-read)

- Correlation, n=16, not causal. Other things changed between versions (scheduling, reranker, CA fallback, symbolic-first ordering) and any of them could drive the LB drop.
- LB noise is about 2 pts: the identical v25 kernel scored 23.47 then 25.56 (`57609744`).
- v26, v27, v28 have no public score in the listing (v26/v27 `COMPLETE` but blank, v28 `PENDING`), so the versions where local eval jumped to 9.9% / 18.6% / 33.7% are not in the fit.
- Local eval is contaminated: see `primitive_audit.md`. 42 of 44 local eval solves come from primitives written for a single eval task and that fire 0 times on train or public test. The metric measures hand-fitting, not generalization.

## Working hypothesis

New symbolic programs that fit the demos become attempt 1 and displace the neural top guess. Task-specific families do nothing on the hidden test, but the generic fallbacks (CA, 7% precision) and scheduling changes can cost neural points.

## Proposed proxies (not yet built)

1. Treat eval as burned for symbolic development. Gate changes on held-out train (never targeted) plus the public-test fire rate.
2. A/B the neural-only vs hybrid output on the same tasks, and require the symbolic top-1 to beat the neural guess on held-out train before it can take attempt 1.
3. Use the real LB sparingly: one run per day, repeat reruns to estimate noise before trusting a delta under ~2 pts.

## Held-out A/B proxy (`src/tools/heldout_proxy.py`, 2026-10-29)

Compares attempt policies (neural-only, production diverse-pick, documented union, union without CA, union gated by leave-one-task-out family precision) with `src.metrics`. 9 unit tests in `tests/test_heldout_proxy.py`.

**Train A/B is not possible with the artifacts we have.** `oof/val_broad/validation_predictions.json` (the 850/1076 "neural CV") already contains the symbolic prediction as `attempt_1` on 200/200 outputs where a program fits, so it is a hybrid output, not neural-only. Every policy ties by construction. This also means the 0.7900 figure in `docs/experiments.md` and `primitive_audit.md` is not a pure neural number: it includes symbolic overrides. The tool now refuses such files (agreement >= 95%).

**Eval A/B with the neural-only baseline** (`submissions/submission_sota_kernel_eval_0.0233.json`, 4/172), generic families only (those that also fire on train):

| Policy | Correct | Gains | Losses |
|---|---|---|---|
| neural | 4/172 | - | - |
| union / production / no_ca / loo | 5/172 | 1 | 0 |

Only 2 eval tasks keep firing once task-specific families are removed (46 -> 2), and one of them (`recolor_motif`) is the single gain. n=1, no statistical power.

## What is still missing

A **neural-only run on train** (same Pass-B settings, no symbolic merge) with per-task attempt_1/attempt_2 saved. That is a Kaggle cluster job; with it, `python3 -m src.tools.heldout_proxy --neural <file>` gives a real held-out gains/losses table on 1,000 tasks, with leave-one-out family gating, in about 30s. Until then there is no valid local estimate of whether symbolic changes help the LB.
