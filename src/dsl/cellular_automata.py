from __future__ import annotations

from typing import Any, Sequence
import numpy as np

Grid = list[list[int]]


def extract_3x3_local_rules(
    train_pairs: list[dict[str, Any]],
) -> dict[tuple[int, ...], int] | None:
    """Extracts deterministic 3x3 local neighborhood transition rules.

    Maps a flattened 9-tuple representing a 3x3 window around cell (r, c)
    to the target color at (r, c). Returns None if there is any contradictory
    mapping (i.e. identical 3x3 context maps to different target colors).
    """
    rules: dict[tuple[int, ...], int] = {}

    for pair in train_pairs:
        inp = np.asarray(pair["input"], dtype=int)
        out = np.asarray(pair["output"], dtype=int)
        if inp.shape != out.shape:
            return None

        h, w = inp.shape
        # Pad with -1 representing boundary / out of bounds
        padded = np.pad(inp, 1, mode="constant", constant_values=-1)

        for r in range(h):
            for c in range(w):
                patch = tuple(padded[r : r + 3, c : c + 3].flatten())
                target = int(out[r, c])
                if patch in rules and rules[patch] != target:
                    return None
                rules[patch] = target

    return rules if rules else None


def apply_3x3_local_rules(
    grid: Grid | np.ndarray,
    rules: dict[tuple[int, ...], int],
    fallback_to_input: bool = True,
) -> Grid | None:
    """Applies 3x3 local neighborhood transition rules to an input grid.

    If fallback_to_input is False, returns None if any patch was unseen.
    """
    arr = np.asarray(grid, dtype=int)
    h, w = arr.shape
    padded = np.pad(arr, 1, mode="constant", constant_values=-1)
    out = np.zeros_like(arr)

    for r in range(h):
        for c in range(w):
            patch = tuple(padded[r : r + 3, c : c + 3].flatten())
            if patch in rules:
                out[r, c] = rules[patch]
            elif fallback_to_input:
                out[r, c] = arr[r, c]
            else:
                return None

    return out.tolist()
