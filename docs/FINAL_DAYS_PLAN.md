# ARC-AGI-2 — Final Days Plan (Oct 30 → Nov 2, 2026)

Written 2026-10-30 ~12:30 UTC. All times are **UTC**.

## 1. Where we stand

| Fact | Value |
|---|---|
| Hard deadline | **Nov 2, 23:59** |
| Scoring time | 24–36 h; a submission still pending at the deadline **cannot be safely selected** |
| Last safe submission | **Nov 1, ~00:00–06:00** |
| Usable daily slots | **Oct 30 (today), Oct 31, Nov 1** (Nov 2 is too late to score) |
| GPU quota | **30h weekly quota reset verified at 00:00 UTC Oct 31** |
| Submissions in flight | v28: **23.06** (complete), v29: **23.89** (complete), Oct 31: **57844167 PENDING** (Boyko dual-pass v3) |
| Our best public score | **30.97** (kernel versions 7 and 8) |
| Bronze cutoff (top 10%, #352 of 3,527) | **32.22** (silver #176: 33.06), snapshot Oct 30 08:42 |
| Kaggle health | General submission errors and slow nodes reported; errored submissions still use the day's slot |

### What the evidence says
- **Our public score has gone down since v16** (v7/v8 30.97 → v17–v25 22.8–26.9) while the local benchmark rose to 36.67%.
  The local benchmark scores hand-written primitives on the same evaluation tasks they were reverse-engineered from,
  so it measures memorisation. **Ignore it for decisions.** v28 (pending) is not expected to beat 30.97.
- **Same code swings ~±3 points run to run** (the 33.89 public notebook scored 29.31 for us).
- **Best candidate we don't own yet: Boyko dual-pass NVARC** (`evgendvorkin/arc-2-2026-4-4-local-boyko-nvarc`):
  4 scored runs 31.81–32.64, average **32.12**. Our fork is ready in `notebooks/boyko_dualpass_fork/` (commit `9ba7488`).
- **Track B (Qwen3.8) is a no-go:** Flash-Next can't fit a single L4; the best public 27B kernel scores 15.69.
  Details: shared note `trackB_go_nogo.md`.

## 2. Day-by-day plan

### Friday Oct 30 (today) — COMPLETED
- [x] Running hybrid kernel finished.
- [x] Submission v29 was submitted (Oct 30 13:23 UTC, currently PENDING).
- [x] Submission v28 completed scoring at **23.06** (consistent with expected public drop from memorized evaluation primitives).
- [x] Confirmed versions 7 and 8 remain COMPLETE as our 30.97 reference baseline.

### Night Oct 30 → Oct 31 (right after the ~23:57 quota reset) — IN PROGRESS
- [x] Quota reset confirmed at **00:00:19 UTC** Oct 31.
- [x] Fixed `isGpuEnabled: true` in notebook JSON metadata (preventing CPU allocation trap).
- [x] Fixed input path resolution: Kaggle CLI mounts attached kernel sources and datasets under /kaggle/input/<slug> rather than /kaggle/input/notebooks/... or /kaggle/input/datasets/... Added dynamic directory/file resolvers for `unsloth_dir`, `packages_dir` (flash-attn), `model_name`, and `test_path`.
- [x] Pushed Kernel version 3 (`aleixlopez/arc-agi2-boyko-dualpass-fork`) to Kaggle.
- [x] Deployed background watcher (`watch_and_push_boyko.py`) to monitor execution, retrieve outputs, and validate `submission.json`.
- [ ] Await verification run completion (~1.5h) and confirm logs (`[fork] fallback submission.json written`, `[V22 Astra] promote=False`, `[V22 selector] submission.json saved`).

### Saturday Oct 31 — slot 2 — COMPLETED
- [x] Boyko verification run (version 3) completed with status COMPLETE and passed all schema validators.
- [x] Submitted for Oct 31 slot:
  - **Submission Ref:** `57844167`
  - **Kernel:** `aleixlopez/arc-agi2-boyko-dualpass-fork` (version 3)
  - **Message:** "Boyko dual-pass fork v3 (Oct 31 slot)"
  - **Status:** `SubmissionStatus.PENDING`
- [ ] Check status periodically until scoring completes (expected ~Nov 1).

### Sunday Nov 1 — slot 3 (the last safe one), submit between 00:00 and 06:00
Pick by this rule:

| Situation | Submit on Nov 1 |
|---|---|
| Oct 31 errored for a Kaggle-side reason | **Retry the same submission** |
| Oct 31 was Boyko, still scoring or scored | **Kernel version 8 re-run** |
| Oct 31 was version 8 (Boyko test run had failed) | **Boyko fork** if its test run now completes; else **version 7 re-run** |

Then stop submitting.

### Monday Nov 2 — selection day (no new submissions)
- [ ] By **~20:00**, open My Submissions and select the **two highest COMPLETE scores**.
  - Default if nothing new beats it: **v7 or v8 (30.97)** + the best new submission.
  - Never select a submission that is still PENDING or ERROR.
- [ ] Keep at least one 30.97 version selected unless two newer ones both score above it.
- [ ] If Kaggle grants a deadline extension (forum topic 753674), re-read this plan; don't improvise new code.

## 3. Rules for the remaining days
1. **One change per decision, decided by public scores only.** No decisions from the local benchmark.
2. **No new primitives, no new pipelines, no Qwen3.8.** Anything needing more than one GPU run before Nov 1 morning is out.
3. **Every candidate must write a complete fallback `submission.json` early** and have per-task timeouts. A timeout must never score 0.00.
4. **Treat a single error as Kaggle noise**, retry once. Only treat it as a code bug if the same version fails twice.
5. **Don't cancel the hybrid kernel run** (it is today's submission) and don't touch `notebooks/submissions/hybrid_sota_v1` or the symbolic merge (another session owns them).

## 4. If something goes wrong

| Problem | Do this |
|---|---|
| Boyko push says quota reached | Wait for the reset time shown on Kaggle's settings page; meanwhile submit version 8 for Oct 31 |
| Boyko push says a model/dataset/kernel source isn't accessible | Attach it once in the Kaggle web editor (the sources are public: `konstantinboyko/flash-attn-2-8-2`, `konstantinboyko/unsloth-2025-9-7-torch-2-8-0-py3-12-patched`, `konstantinboyko/qwen3-4b-bfloat16-v02-01-01`), save, then submit that version |
| Boyko test run errors | Read the log tail; if it's an environment/install failure, don't debug past ~06:00 Oct 31, submit version 8 instead |
| A submission shows ERROR | Retry the same version next slot (rule 4) |
| All new submissions score below 30.97 | Select v7/v8 + the best new one; that's the floor |
| `submit -v 8` fails because version numbers don't match labels | Use the web UI: kernel page → Versions → pick the 30.97 run → Submit |

## 5. Realistic odds
Every candidate is the same Qwen-4B NVARC family. The Boyko fork's 4-run average (32.12) sits right at today's bronze
cutoff, which will drift up a little by the deadline. Medals are decided on the private leaderboard, where run luck
reshuffles this crowded band. Two well-run extra submissions give a real, but well under 50%, chance of bronze.

## 6. Files and references
- Boyko fork: `notebooks/boyko_dualpass_fork/` (builder `build_fork.py`, notebook `submission_kernel.ipynb`)
- Track B probe (unused): `notebooks/trackb_qwen38_probe/`
- Shared notes (outside repo): `trackB_go_nogo.md`, `final_72h_plan.md` in the session-notes folder
- Validation: `src/validator.py`; metric: `src/metrics.py`
