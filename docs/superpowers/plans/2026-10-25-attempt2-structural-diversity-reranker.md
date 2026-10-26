# Attempt-2 Structural Diversity Reranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and integrate a Structural Diversity Reranker to maximize ARC-AGI-2 Attempt 1 / Attempt 2 coverage by filtering near-duplicate predictions and selecting candidates from distinct structural hypotheses.

**Architecture:** Create a standalone diversity engine in `src/models/diversity_reranker.py` with dimensional, relative Hamming, and color palette Jaccard distance metrics. Integrate the diversity selection policy into `HybridSolver` and `src/tools/build_submission_notebook.py` to ensure Attempt 2 selects the highest-scoring structurally diverse candidate with safe progressive fallbacks.

**Tech Stack:** Python 3.12, NumPy, Pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-attempt2-structural-diversity-reranker-design.md`

## Global Constraints

- **Python Version:** 3.12 compatible standard library and NumPy only.
- **Grid Schema:** Every grid must be a 2D list of integers with $1 \le H, W \le 30$ and pixel values in $[0, 9]$.
- **Execution Performance:** Distance calculation and candidate reranking must take $< 1\text{ms}$ per candidate pair on CPU.
- **Backward Compatibility:** All existing 124 unit tests must remain 100% passing.

## Review Focus

1. **Dimensional Mismatches:** Candidates with different row or column counts must never trigger `IndexError` or shape mismatch exceptions in NumPy.
2. **Degenerate Pools:** Empty candidate pools or pools containing a single candidate must never produce `None` or raise exceptions.
3. **Identical Duplicates:** Pools containing identical duplicate grids must never emit identical grids for Attempt 1 and Attempt 2.
4. **Single-Pixel Noise Threshold:** Two candidates differing by only 1 pixel must be rejected as diverse so Attempt 2 is not wasted on noise.
5. **Color Hypothesis Diversity:** Candidates with identical shape but distinct color palettes (e.g. inverted or re-indexed colors) must be recognized as diverse even if Hamming distance is borderline.

---

### Task 1: Core Distance Metrics (`src/models/diversity_reranker.py`)

**Files:**
- Create: `src/models/diversity_reranker.py`
- Test: `tests/test_diversity_reranker.py`

**Interfaces:**
- Produces:
  - `compute_dimensional_distance(g1: Sequence[Sequence[int]] | np.ndarray, g2: Sequence[Sequence[int]] | np.ndarray) -> float`
  - `compute_hamming_distance(g1: Sequence[Sequence[int]] | np.ndarray, g2: Sequence[Sequence[int]] | np.ndarray) -> tuple[int, float]`
  - `compute_palette_distance(g1: Sequence[Sequence[int]] | np.ndarray, g2: Sequence[Sequence[int]] | np.ndarray) -> float`
  - `compute_structural_distance(g1: Sequence[Sequence[int]] | np.ndarray, g2: Sequence[Sequence[int]] | np.ndarray) -> float`

- [ ] **Step 1: Write failing unit tests for core distance metrics**

Create `tests/test_diversity_reranker.py`:
```python
from __future__ import annotations

import numpy as np
import pytest

from src.models.diversity_reranker import (
    compute_dimensional_distance,
    compute_hamming_distance,
    compute_palette_distance,
    compute_structural_distance,
)


def test_compute_dimensional_distance():
    g1 = [[1, 2], [3, 4]]
    g2 = [[1, 2, 0], [3, 4, 0]]
    g3 = [[1, 2], [3, 4]]
    assert compute_dimensional_distance(g1, g2) == 1.0
    assert compute_dimensional_distance(g1, g3) == 0.0


def test_compute_hamming_distance():
    g1 = [[1, 1], [1, 1]]
    g2 = [[1, 1], [1, 2]]  # 1 pixel diff out of 4 (ratio 0.25)
    g3 = [[2, 2], [2, 2]]  # 4 pixels diff out of 4 (ratio 1.0)
    
    diff_pixels, ratio = compute_hamming_distance(g1, g2)
    assert diff_pixels == 1
    assert ratio == 0.25

    diff_pixels_full, ratio_full = compute_hamming_distance(g1, g3)
    assert diff_pixels_full == 4
    assert ratio_full == 1.0


def test_compute_palette_distance():
    g1 = [[1, 1], [1, 2]]  # colors {1, 2}
    g2 = [[1, 1], [1, 2]]  # colors {1, 2} -> jaccard distance 0.0
    g3 = [[3, 3], [3, 4]]  # colors {3, 4} -> disjoint -> jaccard distance 1.0
    g4 = [[1, 1], [1, 3]]  # colors {1, 3} -> inter=1, union=3 -> 1 - 1/3 = 0.667

    assert compute_palette_distance(g1, g2) == 0.0
    assert compute_palette_distance(g1, g3) == 1.0
    assert pytest.approx(compute_palette_distance(g1, g4), rel=1e-2) == 0.667


def test_compute_structural_distance_different_dimensions():
    g1 = [[1, 1]]
    g2 = [[1], [1]]
    assert compute_structural_distance(g1, g2) == 1.0


def test_compute_structural_distance_identical():
    g1 = [[1, 2], [3, 4]]
    assert compute_structural_distance(g1, g1) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_diversity_reranker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.diversity_reranker'`

- [ ] **Step 3: Implement core distance metrics**

Create `src/models/diversity_reranker.py`:
```python
from __future__ import annotations

from typing import Sequence
import numpy as np

GridType = Sequence[Sequence[int]] | np.ndarray


def to_np_array(grid: GridType) -> np.ndarray:
    """Converts a grid to an integer NumPy array."""
    if isinstance(grid, np.ndarray):
        return grid.astype(int)
    return np.asarray(grid, dtype=int)


def compute_dimensional_distance(g1: GridType, g2: GridType) -> float:
    """Returns 1.0 if shapes differ, 0.0 if shapes match."""
    a1 = to_np_array(g1)
    a2 = to_np_array(g2)
    return 0.0 if a1.shape == a2.shape else 1.0


def compute_hamming_distance(g1: GridType, g2: GridType) -> tuple[int, float]:
    """Computes (diff_pixels, relative_hamming_ratio) for grids of identical shape.

    If shapes differ, returns (max_size, 1.0).
    """
    a1 = to_np_array(g1)
    a2 = to_np_array(g2)
    if a1.shape != a2.shape:
        return max(a1.size, a2.size), 1.0
    if a1.size == 0:
        return 0, 0.0
    diff_count = int(np.sum(a1 != a2))
    ratio = float(diff_count / a1.size)
    return diff_count, ratio


def compute_palette_distance(g1: GridType, g2: GridType) -> float:
    """Computes Jaccard distance between unique color palettes in [0.0, 1.0]."""
    a1 = to_np_array(g1)
    a2 = to_np_array(g2)
    p1 = set(np.unique(a1).tolist())
    p2 = set(np.unique(a2).tolist())
    union = p1 | p2
    if not union:
        return 0.0
    inter = p1 & p2
    return float(1.0 - len(inter) / len(union))


def compute_structural_distance(g1: GridType, g2: GridType) -> float:
    """Computes composite structural distance in [0.0, 1.0]."""
    if compute_dimensional_distance(g1, g2) > 0.0:
        return 1.0
    _, hamming_ratio = compute_hamming_distance(g1, g2)
    palette_dist = compute_palette_distance(g1, g2)
    return float(0.7 * hamming_ratio + 0.3 * palette_dist)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_diversity_reranker.py -v`
Expected: PASS (5/5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/models/diversity_reranker.py tests/test_diversity_reranker.py
git commit -m "feat(models): add core structural distance metrics in diversity_reranker"
```

---

### Task 2: Diversity Classifier & Attempt Pair Selection (`src/models/diversity_reranker.py`)

**Files:**
- Modify: `src/models/diversity_reranker.py`
- Test: `tests/test_diversity_reranker.py`

**Interfaces:**
- Consumes: `compute_dimensional_distance`, `compute_hamming_distance`, `compute_palette_distance` from Task 1.
- Produces:
  - `is_structurally_diverse(g1: GridType, g2: GridType, min_pixels: int = 2, min_hamming_ratio: float = 0.05, min_palette_distance: float = 0.25) -> bool`
  - `select_diverse_attempt_pair(candidates: Sequence[GridType], fallback_input: GridType | None = None, min_pixels: int = 2, min_hamming_ratio: float = 0.05, min_palette_distance: float = 0.25) -> tuple[list[list[int]], list[list[int]]]`

- [ ] **Step 1: Write failing unit tests for diverse pair selection**

Append to `tests/test_diversity_reranker.py`:
```python
from src.models.diversity_reranker import (
    is_structurally_diverse,
    select_diverse_attempt_pair,
)


def test_is_structurally_diverse_rejects_single_pixel_noise():
    # 10x10 canvas, differing by exactly 1 pixel (1% Hamming)
    g1 = np.zeros((10, 10), dtype=int)
    g2 = g1.copy()
    g2[0, 0] = 1
    assert not is_structurally_diverse(g1, g2, min_pixels=2, min_hamming_ratio=0.05)


def test_is_structurally_diverse_accepts_different_dimensions():
    g1 = np.zeros((5, 5), dtype=int)
    g2 = np.zeros((5, 6), dtype=int)
    assert is_structurally_diverse(g1, g2)


def test_is_structurally_diverse_accepts_palette_shift():
    # 4x4 canvas with identical geometry but distinct colors
    g1 = np.full((4, 4), 1, dtype=int)
    g2 = np.full((4, 4), 2, dtype=int)
    assert is_structurally_diverse(g1, g2)


def test_select_diverse_attempt_pair_filters_near_duplicate():
    cand1 = [[1, 1, 1], [1, 1, 1]]
    cand2_noise = [[1, 1, 1], [1, 1, 2]]  # 1 pixel noise: rejected
    cand3_diverse = [[2, 2, 2], [2, 2, 2]]  # structurally distinct: accepted
    
    a1, a2 = select_diverse_attempt_pair([cand1, cand2_noise, cand3_diverse])
    assert a1 == cand1
    assert a2 == cand3_diverse


def test_select_diverse_attempt_pair_empty_and_single_pool():
    fallback = [[0, 0], [0, 0]]
    # Empty pool
    a1, a2 = select_diverse_attempt_pair([], fallback_input=fallback)
    assert a1 == fallback
    assert a2 != a1  # Geometric transformation fallback

    # Single candidate pool
    c1 = [[1, 2], [3, 4]]
    a1, a2 = select_diverse_attempt_pair([c1], fallback_input=fallback)
    assert a1 == c1
    assert a2 != a1


def test_select_diverse_attempt_pair_all_identical():
    c1 = [[1, 2], [3, 4]]
    a1, a2 = select_diverse_attempt_pair([c1, c1, c1])
    assert a1 == c1
    assert a2 != a1  # Fallback produces distinct transformation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_diversity_reranker.py -k test_select_diverse_attempt_pair`
Expected: FAIL with `ImportError: cannot import name 'is_structurally_diverse'`

- [ ] **Step 3: Implement diversity classifier & pair selection**

Add to `src/models/diversity_reranker.py`:
```python
def is_structurally_diverse(
    g1: GridType,
    g2: GridType,
    min_pixels: int = 2,
    min_hamming_ratio: float = 0.05,
    min_palette_distance: float = 0.25,
) -> bool:
    """Checks whether g2 represents an alternative structural hypothesis from g1."""
    if compute_dimensional_distance(g1, g2) > 0.0:
        return True

    diff_pixels, hamming_ratio = compute_hamming_distance(g1, g2)
    if diff_pixels >= min_pixels and hamming_ratio >= min_hamming_ratio:
        return True

    if compute_palette_distance(g1, g2) >= min_palette_distance:
        return True

    return False


def _safe_geometric_fallback(grid: GridType) -> list[list[int]]:
    """Generates a distinct ARC-valid attempt when candidate pool is exhausted."""
    arr = to_np_array(grid)
    if arr.size == 0:
        return [[0]]
    
    # Try horizontal reflection
    refl = np.fliplr(arr)
    if not np.array_equal(refl, arr):
        return refl.tolist()
        
    # Try vertical reflection
    refl_v = np.flipud(arr)
    if not np.array_equal(refl_v, arr):
        return refl_v.tolist()
        
    # If completely symmetric, invert non-zero colors: (c % 9) + 1
    mod = np.where(arr != 0, (arr % 9) + 1, 1)
    if not np.array_equal(mod, arr):
        return mod.tolist()
        
    return arr.tolist()


def select_diverse_attempt_pair(
    candidates: Sequence[GridType],
    fallback_input: GridType | None = None,
    min_pixels: int = 2,
    min_hamming_ratio: float = 0.05,
    min_palette_distance: float = 0.25,
) -> tuple[list[list[int]], list[list[int]]]:
    """Selects (attempt_1, attempt_2) maximizing structural hypothesis coverage."""
    clean_candidates = [
        to_np_array(c).tolist() for c in candidates if c is not None and len(c) > 0
    ]

    # Handle empty pool
    if not clean_candidates:
        default_grid = to_np_array(fallback_input).tolist() if fallback_input is not None else [[0]]
        return default_grid, _safe_geometric_fallback(default_grid)

    attempt_1 = clean_candidates[0]

    # Scan for first candidate meeting diversity threshold
    for cand in clean_candidates[1:]:
        if is_structurally_diverse(
            attempt_1,
            cand,
            min_pixels=min_pixels,
            min_hamming_ratio=min_hamming_ratio,
            min_palette_distance=min_palette_distance,
        ):
            return attempt_1, cand

    # Progressive relaxation 1: any candidate that is not bitwise identical to attempt_1
    for cand in clean_candidates[1:]:
        if not np.array_equal(np.asarray(cand), np.asarray(attempt_1)):
            return attempt_1, cand

    # Progressive relaxation 2: generate geometric fallback distinct from attempt_1
    attempt_2 = _safe_geometric_fallback(attempt_1)
    return attempt_1, attempt_2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_diversity_reranker.py -v`
Expected: PASS (all 11 tests passed)

- [ ] **Step 5: Commit**

```bash
git add src/models/diversity_reranker.py tests/test_diversity_reranker.py
git commit -m "feat(models): implement is_structurally_diverse and select_diverse_attempt_pair"
```

---

### Task 3: Integration into HybridSolver (`src/models/hybrid_solver.py`)

**Files:**
- Modify: `src/models/hybrid_solver.py:80-165`
- Test: `tests/test_dsl.py`

**Interfaces:**
- Consumes: `select_diverse_attempt_pair` from `src.models.diversity_reranker`.
- Produces: Updated `HybridSolver._generate_from_programs` and `_solve_tier2` returning structurally diverse pairs.

- [ ] **Step 1: Write test verifying diverse selection in HybridSolver**

Append to `tests/test_dsl.py`:
```python
def test_hybrid_solver_generates_diverse_attempts():
    from src.models.hybrid_solver import HybridSolver

    solver = HybridSolver()
    # Task with identity input
    task = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}
        ],
        "test": [
            {"input": [[5, 6], [7, 8]]}
        ]
    }
    preds = solver.solve_task(task)
    assert len(preds) == 1
    assert "attempt_1" in preds[0]
    assert "attempt_2" in preds[0]
    # Attempt 1 and Attempt 2 must never be bitwise identical
    assert preds[0]["attempt_1"] != preds[0]["attempt_2"]
```

- [ ] **Step 2: Run test to check current behavior**

Run: `python3 -m pytest tests/test_dsl.py -k test_hybrid_solver_generates_diverse_attempts`
Expected: PASS or FAIL depending on fallback logic.

- [ ] **Step 3: Update `src/models/hybrid_solver.py` to use `select_diverse_attempt_pair`**

Update `src/models/hybrid_solver.py`:
```python
from src.models.diversity_reranker import select_diverse_attempt_pair
```
And refactor `_generate_from_programs`:
```python
    def _generate_from_programs(
        self,
        perfect_programs: list[tuple[Program, float]],
        test_pairs: list[dict[str, Any]],
    ) -> list[dict[str, Grid]]:
        results: list[dict[str, Grid]] = []

        for item in test_pairs:
            test_in = item["input"]
            candidates: list[Grid] = []

            for prog, _ in perfect_programs:
                try:
                    cand = prog.execute(test_in)
                    candidates.append(cand)
                except Exception:
                    continue

            a1, a2 = select_diverse_attempt_pair(candidates, fallback_input=test_in)
            results.append({"attempt_1": a1, "attempt_2": a2})
        return results
```
And refactor `_solve_tier2`:
```python
    def _solve_tier2(self, task: dict[str, Any]) -> list[dict[str, Grid]]:
        train_pairs = task.get("train", [])
        test_pairs = task.get("test", [])

        # Candidate transforms
        transforms = [
            ("identity", lambda g: [row[:] for row in g]),
            ("rot90", rotate_90),
            ("rot180", rotate_180),
            ("refl_h", reflect_horizontal),
            ("refl_v", reflect_vertical),
        ]

        results: list[dict[str, Grid]] = []
        for item in test_pairs:
            test_in = item["input"]
            cands = []
            for _, fn in transforms:
                try:
                    cands.append(fn(test_in))
                except Exception:
                    continue
            a1, a2 = select_diverse_attempt_pair(cands, fallback_input=test_in)
            results.append({"attempt_1": a1, "attempt_2": a2})
        return results
```

- [ ] **Step 4: Run full test suite and evaluate**

Run: `python3 -m pytest tests/`
Expected: PASS (all 125 tests passed)

Run: `python3 -m src.evaluate --solver hybrid`
Expected: Successful benchmark evaluation with verified schema.

- [ ] **Step 5: Commit**

```bash
git add src/models/hybrid_solver.py tests/test_dsl.py
git commit -m "feat(hybrid): integrate select_diverse_attempt_pair into HybridSolver"
```

---

### Task 4: Integration into Submission Notebook Builder (`src/tools/build_submission_notebook.py`)

**Files:**
- Modify: `src/tools/build_submission_notebook.py`
- Test: `tests/test_submission_notebook_builder.py`

**Interfaces:**
- Injects diversity selection logic into Cell 9 of generated Kaggle submission notebooks.

- [ ] **Step 1: Write test verifying diverse attempt selection in builder**

Add test to `tests/test_submission_notebook_builder.py`:
```python
def test_submission_builder_injects_diversity_selection(tmp_path: Path):
    output_path = tmp_path / "submission_diversity.ipynb"
    build(output_path=output_path, validation_mode="submission")
    nb = json.loads(output_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    merge_source = next(source for source in code_cells if "decoder.load_decoded_results" in source)
    assert "select_diverse_attempt_pair" in merge_source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_submission_notebook_builder.py -k test_submission_builder_injects_diversity_selection`
Expected: FAIL

- [ ] **Step 3: Update `build_submission_notebook.py`**

In `src/tools/build_submission_notebook.py`, inject the diversity selector into the `arc_decoder.py` cell or the merge code cell:
```python
def select_diverse_pair(candidates, fallback=None):
    clean = [c for c in candidates if c is not None]
    if not clean:
        return fallback, fallback
    a1 = clean[0]
    for c in clean[1:]:
        # Different dimensions
        if np.shape(c) != np.shape(a1):
            return a1, c
        diff = np.sum(np.array(c) != np.array(a1))
        # At least 2 pixels diff and 5% Hamming ratio
        if diff >= 2 and (diff / np.size(a1)) >= 0.05:
            return a1, c
    # Distinct fallback
    for c in clean[1:]:
        if not np.array_equal(c, a1):
            return a1, c
    return a1, a1
```
And update the assignment of `submission[task_id][i]["attempt_2"]` to use `select_diverse_pair`.

- [ ] **Step 4: Run tests and rebuild submission kernel**

Run: `python3 -m pytest tests/`
Expected: All tests pass.

Run: `python3 -m src.tools.build_submission_notebook`
Expected: `notebooks/submissions/hybrid_sota_v1/submission_kernel.ipynb` built cleanly with zero syntax/indentation errors.

- [ ] **Step 5: Commit**

```bash
git add src/tools/build_submission_notebook.py tests/test_submission_notebook_builder.py notebooks/submissions/hybrid_sota_v1/submission_kernel.ipynb
git commit -m "feat(tools): inject structural diversity reranking into Kaggle submission builder"
```

---

### Task 5: Full System Verification & Benchmark Evaluation

**Files:**
- Run: `python3 -m src.evaluate --solver hybrid`
- Run: `python3 -m pytest tests/`

- [ ] **Step 1: Run complete test suite**
- [ ] **Step 2: Run local evaluation benchmark**
- [ ] **Step 3: Confirm schema compliance via `src.validator`**
- [ ] **Step 4: Final verification summary**
