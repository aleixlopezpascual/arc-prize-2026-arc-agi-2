from __future__ import annotations

import numpy as np
import pytest

from src.models.diversity_reranker import (
    compute_dimensional_distance,
    compute_hamming_distance,
    compute_palette_distance,
    compute_structural_distance,
    is_structurally_diverse,
    select_diverse_attempt_pair,
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

