from __future__ import annotations

import copy
from typing import Any, Callable
import numpy as np

from src.metrics import grid_equal


# Transform signature: takes 2D grid (list of lists of ints) and returns transformed 2D grid
TransformFn = Callable[[list[list[int]]], list[list[int]]]


def transform_identity(grid: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in grid]


def transform_rot90(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    return np.rot90(arr, k=-1).tolist()


def transform_rot180(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    return np.rot90(arr, k=2).tolist()


def transform_rot270(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    return np.rot90(arr, k=1).tolist()


def transform_flip_lr(grid: list[list[int]]) -> list[list[int]]:
    return [row[::-1] for row in grid]


def transform_flip_ud(grid: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in grid[::-1]]


def transform_transpose(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    return arr.T.tolist()


BASIC_GEOMETRIC_TRANSFORMS: list[tuple[str, TransformFn]] = [
    ("identity", transform_identity),
    ("rot90", transform_rot90),
    ("rot180", transform_rot180),
    ("rot270", transform_rot270),
    ("flip_lr", transform_flip_lr),
    ("flip_ud", transform_flip_ud),
    ("transpose", transform_transpose),
]


def find_consistent_color_map(
    train_pairs: list[dict[str, list[list[int]]]],
) -> dict[int, int] | None:
    """Finds if there is a consistent 1-to-1 or many-to-one color mapping across all train pairs."""
    mapping: dict[int, int] = {}
    for pair in train_pairs:
        inp = np.asarray(pair["input"], dtype=int)
        out = np.asarray(pair["output"], dtype=int)
        if inp.shape != out.shape:
            return None
        for in_val, out_val in zip(inp.flat, out.flat):
            if in_val in mapping and mapping[in_val] != out_val:
                return None
            mapping[in_val] = int(out_val)
    return mapping


def apply_color_map(grid: list[list[int]], mapping: dict[int, int]) -> list[list[int]]:
    return [[mapping.get(val, val) for val in row] for row in grid]


class BaselineHeuristicSolver:
    """Baseline solver evaluating simple geometric and color-mapping transforms."""

    def __init__(self) -> None:
        pass

    def solve_task(self, task: dict[str, Any]) -> list[dict[str, list[list[int]]]]:
        """Solves a single ARC task, returning predictions for all test inputs."""
        train_pairs = task.get("train", [])
        test_pairs = task.get("test", [])

        # Check candidate transforms against train pairs
        candidate_transforms: list[tuple[str, TransformFn, float]] = []

        # 1. Test geometric transforms
        for name, fn in BASIC_GEOMETRIC_TRANSFORMS:
            correct_count = 0
            possible = True
            for pair in train_pairs:
                try:
                    pred = fn(pair["input"])
                    if grid_equal(pred, pair["output"]):
                        correct_count += 1
                except Exception:
                    possible = False
                    break
            if possible and len(train_pairs) > 0:
                score = correct_count / len(train_pairs)
                if score > 0:
                    candidate_transforms.append((name, fn, score))

        # 2. Test color mapping
        cmap = find_consistent_color_map(train_pairs)
        if cmap is not None and len(train_pairs) > 0:
            candidate_transforms.append(
                ("color_map", lambda g, m=cmap: apply_color_map(g, m), 1.0)
            )

        # Sort candidate transforms by train score descending
        candidate_transforms.sort(key=lambda x: x[2], reverse=True)

        results: list[dict[str, list[list[int]]]] = []

        for test_item in test_pairs:
            test_in = test_item["input"]

            attempt_1: list[list[int]] | None = None
            attempt_2: list[list[int]] | None = None

            # Pick best candidate for attempt_1
            if candidate_transforms and candidate_transforms[0][2] == 1.0:
                try:
                    attempt_1 = candidate_transforms[0][1](test_in)
                except Exception:
                    attempt_1 = None

            if attempt_1 is None:
                attempt_1 = transform_identity(test_in)

            # Pick candidate for attempt_2
            if len(candidate_transforms) > 1 and candidate_transforms[1][2] == 1.0:
                try:
                    cand2 = candidate_transforms[1][1](test_in)
                    if not grid_equal(cand2, attempt_1):
                        attempt_2 = cand2
                except Exception:
                    attempt_2 = None

            if attempt_2 is None:
                # Fallback: inverted or transposed grid
                try:
                    alt = transform_flip_lr(test_in)
                    if not grid_equal(alt, attempt_1):
                        attempt_2 = alt
                except Exception:
                    pass

            if attempt_2 is None:
                attempt_2 = transform_identity(test_in)

            results.append({"attempt_1": attempt_1, "attempt_2": attempt_2})

        return results

    def predict_all(
        self, challenges: dict[str, Any]
    ) -> dict[str, list[dict[str, list[list[int]]]]]:
        """Runs the solver across all tasks in challenges dictionary."""
        all_preds: dict[str, list[dict[str, list[list[int]]]]] = {}
        for task_id, task in challenges.items():
            all_preds[task_id] = self.solve_task(task)
        return all_preds
