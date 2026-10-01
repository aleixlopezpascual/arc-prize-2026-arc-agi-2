# Competition Brief: ARC Prize 2026 - ARC-AGI-2

## 1. Problem Statement & Formulation
The **ARC Prize 2026 - ARC-AGI-2** (Abstraction and Reasoning Corpus) competition benchmark measures AI generalization and abstract visual reasoning on novel, previously unseen puzzle tasks.

Each task presents:
- **`train` pairs (2 to 5 demonstrations):** Each with an `"input"` grid and an `"output"` grid illustrating an underlying transformation rule.
- **`test` pairs (1 or more inputs, typically 1, sometimes 2):** Test `"input"` grids where the algorithm must predict the target `"output"` grid.

Grids are 2D rectangular matrices (lists of lists) of integers $0 \dots 9$, representing 10 distinct colors, with dimensions ranging from $1 \times 1$ up to $30 \times 30$.

The goal is to generate exact output grids for all test inputs of each task using at most **2 attempts** per test input.

---

## 2. Official Evaluation Metric & Submission Protocol

### Metric Mechanics
- For each test input $i$ in task $T$, the system provides two predictions: `attempt_1` and `attempt_2`.
- Ground truth is an exact target grid $Y_i$.
- **Exact Match:** A prediction $\hat{Y}$ is correct if and only if:
  $$\text{dim}(\hat{Y}) == \text{dim}(Y_i) \quad \text{and} \quad \forall (r, c), \hat{Y}[r, c] == Y_i[r, c]$$
- **Per-Test Scoring:**
  $$\text{Score}(i) = \begin{cases} 1 & \text{if } \hat{Y}^{(1)}_i = Y_i \text{ or } \hat{Y}^{(2)}_i = Y_i \\ 0 & \text{otherwise} \end{cases}$$
- **Overall Metric:**
  $$\text{Accuracy} = \frac{\sum_{i=1}^{N_{\text{test}}} \text{Score}(i)}{N_{\text{test}}}$$
  where $N_{\text{test}}$ is the total number of test pairs across all tasks in the evaluation set.

### Submission File Format
The submission must be a single file named `submission.json`:
```json
{
  "task_id_1": [
    {
      "attempt_1": [[0, 1], [1, 0]],
      "attempt_2": [[0, 0], [0, 0]]
    }
  ],
  "task_id_2": [
    {
      "attempt_1": [[...]],
      "attempt_2": [[...]]
    },
    {
      "attempt_2": [[...]],
      "attempt_2": [[...]]
    }
  ]
}
```
**Strict Requirements:**
1. Every task ID in the evaluation/test challenge file must be present as a top-level key.
2. The list of predictions for a task must match the exact length and ordering of the task's `"test"` list.
3. Each item in the list must contain both `"attempt_1"` and `"attempt_2"` keys.
4. Each attempt must be a valid 2D array of integers between 0 and 9.

---

## 3. Dataset Audit & Inventory

The dataset is partitioned into:
- **`arc-agi_training_challenges.json`:** 1,000 tasks (1,076 total test pairs; 69 tasks have $>1$ test pairs).
- **`arc-agi_training_solutions.json`:** Ground truth test outputs for the 1,000 training tasks.
- **`arc-agi_evaluation_challenges.json`:** 120 tasks (172 total test pairs; 49 tasks have $>1$ test pairs).
- **`arc-agi_evaluation_solutions.json`:** Ground truth test outputs for the 120 evaluation tasks.
- **`arc-agi_test_challenges.json`:** 240 tasks (local placeholder; replaced with rerun hidden set during notebook submission).
- **`sample_submission.json`:** 240 tasks template.

Total compressed download size: ~6.6 MB.

---

## 4. Cross-Validation & Validation Gate

### Local Validation Split
- **Primary Benchmark:** The 120-task official evaluation set (`arc-agi_evaluation_challenges.json` & `solutions.json`).
  - Total test instances: 172 test outputs.
  - Mirrors official Kaggle evaluation format.
- **Training Cross-Validation (K-Fold on Tasks):**
  - Grouping unit: `task_id` (a task's train and test pairs must never be split across folds).
  - 5-Fold split across the 1,000 training tasks (200 tasks per fold, ~215 test pairs per fold).
  - Stratification criteria: task characteristics such as dimension-change behavior (isometric vs resize), grid size bins, and number of training demonstration pairs.

### Validation Gate Mandate
No modeling change or heuristic may be promoted without measuring its exact impact on local CV score using `src/metrics.py`.

---

## 5. Critical Gotchas, Traps & Community Intel

1. **ARC-1 vs ARC-2 Discrepancy:**
   - ARC-1 evaluation tasks were largely integrated into the ARC-2 training set.
   - Any public claim of 90%+ accuracy on ARC-1 is invalid for ARC-2.
   - ARC-2 tasks exhibit significantly higher compositional depth, requiring multi-step program induction.

2. **Rerun Timeout & Budget Management:**
   - Kaggle execution limit: 12 hours for 240 unseen test tasks (~180 seconds average per task).
   - Any test-time training (TTT), LLM beam search, or combinatorial DSL synthesis must enforce strict per-task timeouts (e.g. 120s max per task) with deterministic fallbacks.

3. **Memory Leaks during Long Reruns:**
   - Running 240 tasks sequentially easily triggers VRAM fragmentation or CPU memory exhaustion.
   - Mandatory `gc.collect()` and accelerator cache clearing between tasks.

4. **Multi-Test Task Ordering:**
   - While most tasks have 1 test input, 49 of 120 evaluation tasks (and ~30-40% of test tasks) have 2 or more test inputs.
   - Mismatched ordering or output array length results in complete failure for that task.

---

## 6. Strategic Roadmap & Baseline Plan

### Phase 1: Foundation & Baseline Solver
- Implement exact metric calculation and submission verification in `src/metrics.py` and `src/validator.py`.
- Implement a baseline solver (`src/models/baseline_heuristics.py`):
  - Grid dimension predictor (identity, factor scaling, bounding crop).
  - Trivial transforms: Identity / Copy, color permutation matching, spatial rotations/flips.
  - Fallback grid generator ensuring 100% valid submission schema.
- Run baseline against local evaluation set and establish baseline CV score.

### Phase 2: Neural-Symbolic & Program Synthesis Approaches
- Investigate ARC Domain Specific Languages (DSL) and combinatorial program search:
  - Primitives: object segmentation, connected components, geometric transforms, flood fill, border expansion, pattern repetition.
  - Execution engine with fast pruning on demonstration pairs.

### Phase 3: Test-Time Training (TTT) & LLMs
- Study the winning NVARC 2025 approach and open-source models:
  - Few-shot prompting / fine-tuning with open models (e.g., Qwen 2.5, DeepSeek, Llama).
  - Test-time fine-tuning / self-consistency voting across multiple candidate generations.
- Ensemble DSL candidates and LLM predictions to maximize 2-attempt coverage.
