# Technical Design Specification: Attempt-2 Structural Diversity Reranker

**Date:** 2026-10-25  
**Status:** Draft / Pending Review  
**Target:** ARC-AGI-2 (Hybrid SOTA Pipeline)  
**Author:** Gemini CLI & Aleix López  

---

## 1. Executive Summary & Purpose

In the ARC-AGI-2 competition, submissions evaluate two predicted attempts per test output: Attempt 1 and Attempt 2 ($A_1 \lor A_2$). A puzzle is scored as solved if **either** attempt matches the ground truth exactly.

Under standard beam search and greedy ensemble ranking, candidate #2 is selected purely as the second highest-scoring candidate. In practice, candidate #2 is often an unhelpful, near-duplicate variant of candidate #1 (differing by only 1–2 noisy pixels in an otherwise identical canvas). If candidate #1 fails because its core hypothesis (dimensions, symmetry, orientation, or underlying rule) is incorrect, candidate #2 fails simultaneously, effectively forfeiting the second attempt.

This specification introduces a dedicated **Structural Diversity Reranker** (`src/models/diversity_reranker.py`) to maximize the coverage of the Attempt 1 / Attempt 2 union by ensuring Attempt 2 represents the highest-scoring candidate from an alternative structural hypothesis.

---

## 2. Distance Metrics & Structural Divergence

For two grids $G_1$ and $G_2$:

### 2.1 Dimensional Distance
If $\text{shape}(G_1) \ne \text{shape}(G_2)$, the candidates represent fundamentally distinct spatial hypotheses (e.g. cropping vs. expansion vs. full preservation). 
$$\text{dist}_{\text{dim}}(G_1, G_2) = 1.0$$

### 2.2 Relative Hamming Distance
For grids of identical dimensions $H \times W$:
$$\text{dist}_{\text{hamming}}(G_1, G_2) = \frac{\sum_{r, c} \mathbb{I}[G_1[r, c] \ne G_2[r, c]]}{H \times W}$$
We also measure absolute pixel differences $\Delta_{\text{pixels}} = \sum_{r, c} \mathbb{I}[G_1[r, c] \ne G_2[r, c]]$.

### 2.3 Color Palette Jaccard Distance
Let $P_1 = \text{unique\_colors}(G_1)$ and $P_2 = \text{unique\_colors}(G_2)$:
$$\text{dist}_{\text{palette}}(G_1, G_2) = 1.0 - \frac{|P_1 \cap P_2|}{|P_1 \cup P_2|}$$

### 2.4 Composite Diversity Criterion
A candidate $C_i$ is accepted as structurally diverse from $A_1$ if:
1. $\text{shape}(C_i) \ne \text{shape}(A_1)$, **OR**
2. $\Delta_{\text{pixels}}(C_i, A_1) \ge \tau_{\text{min\_pixels}}$ **AND** $\text{dist}_{\text{hamming}}(C_i, A_1) \ge \tau_{\text{hamming}}$, **OR**
3. $\text{dist}_{\text{palette}}(C_i, A_1) \ge \tau_{\text{palette}}$.

**Default Hyperparameters:**
- $\tau_{\text{min\_pixels}} = 2$ (differ by at least 2 pixels to reject single-pixel noise)
- $\tau_{\text{hamming}} = 0.05$ (at least 5% pixel difference for larger grids)
- $\tau_{\text{palette}} = 0.25$ (distinct color palette hypothesis)

---

## 3. Algorithm & Selection Policy

Given a ranked list of candidate grids $[C_1, C_2, C_3, \dots, C_k]$ (sorted by confidence descending):

1. **Attempt 1 Assignment:**
   $$A_1 = C_1$$
2. **Attempt 2 Scan:**
   Iterate through $C_i$ for $i \in [2, k]$:
   - If $C_i$ satisfies the **Composite Diversity Criterion** relative to $A_1$, select $A_2 = C_i$ and terminate search.
3. **Progressive Relaxation Fallback:**
   - If no candidate satisfies the strict diversity threshold, pick the first candidate $C_j$ ($j \ge 2$) such that $C_j \ne A_1$ (any distinct grid).
   - If all candidates in the pool are bitwise identical to $A_1$, generate a deterministic geometric fallback (e.g. horizontal reflection or identity copy) compliant with ARC schema.

---

## 4. Architecture & Component Placement

```
src/
├── models/
│   ├── diversity_reranker.py   <-- [NEW] Core distance metrics & selection logic
│   └── hybrid_solver.py        <-- [UPDATED] Uses diversity reranker in Tier 1 & 2
├── tools/
│   └── build_submission_notebook.py <-- [UPDATED] Injects diverse selection in Cell 9
└── tests/
    └── test_diversity_reranker.py <-- [NEW] 100% unit test coverage
```

### 4.1 Interface Specification (`src/models/diversity_reranker.py`)

```python
def compute_structural_distance(
    grid_a: Grid | np.ndarray,
    grid_b: Grid | np.ndarray,
) -> float:
    """Computes normalized structural distance in [0.0, 1.0]."""
    ...

def is_structurally_diverse(
    grid_a: Grid | np.ndarray,
    grid_b: Grid | np.ndarray,
    min_pixels: int = 2,
    min_hamming_ratio: float = 0.05,
    min_palette_distance: float = 0.25,
) -> bool:
    """Checks whether grid_b represents a distinct structural hypothesis from grid_a."""
    ...

def select_diverse_attempt_pair(
    candidates: Sequence[Grid | np.ndarray],
    fallback_input: Grid | np.ndarray | None = None,
    min_pixels: int = 2,
    min_hamming_ratio: float = 0.05,
    min_palette_distance: float = 0.25,
) -> tuple[Grid, Grid]:
    """Returns (attempt_1, attempt_2) maximizing hypothesis diversity."""
    ...
```

---

## 5. Edge Cases & Robustness Handling

| Edge Case | Behavior |
| :--- | :--- |
| **Empty Candidate Pool** | Fallback to `fallback_input` (or empty grid), with Attempt 2 derived from geometric reflection. |
| **Single Candidate in Pool** | Attempt 1 = candidate; Attempt 2 derived via safe fallback (reflection if distinct, else identity). |
| **All Candidates Identical** | Attempt 1 = candidate; Attempt 2 = fallback distinct transformation. Never outputs duplicate identical attempts. |
| **Different Dimensions** | Evaluated safely without indexing exceptions or shape mismatch crashes. |
| **Out-of-Bounds Shapes** | Retains strict clipping to $30 \times 30$ and integers in $[0, 9]$. |

---

## 6. Testing & Verification Plan

1. **Unit Tests (`tests/test_diversity_reranker.py`):**
   - Distance computation between identical grids (returns 0.0).
   - Distance computation between grids of different dimensions (returns 1.0).
   - Distance computation for small noise (rejected) vs. structural shift (accepted).
   - Color palette Jaccard distance verification.
   - Pair selection with candidate pool of diverse vs. near-duplicate candidates.
   - Extreme edge cases: pool size 0, 1, and duplicate pools.
2. **Integration Verification:**
   - Run `HybridSolver` evaluation across 120 official ARC-AGI-2 evaluation puzzles (`python3 -m src.evaluate --solver hybrid`).
   - Confirm output schema passes `src/validator.py` with 100% compliance.
   - Pre-compile `submission_kernel.ipynb` with `src.tools.build_submission_notebook` to guarantee clean remote notebook generation.
