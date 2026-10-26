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
    if diff_pixels < min_pixels:
        return False

    if hamming_ratio >= min_hamming_ratio:
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

