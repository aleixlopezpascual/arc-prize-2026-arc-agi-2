from __future__ import annotations

from typing import Any, Callable
import numpy as np

Grid = list[list[int]]


def apply_d4(arr: np.ndarray, transform_idx: int) -> np.ndarray:
    """Applies one of 8 dihedral symmetries (D4 group)."""
    if transform_idx == 0:
        return arr
    elif transform_idx == 1:
        return np.rot90(arr, -1)  # 90 deg clockwise
    elif transform_idx == 2:
        return np.rot90(arr, 2)  # 180 deg
    elif transform_idx == 3:
        return np.rot90(arr, 1)  # 270 deg clockwise
    elif transform_idx == 4:
        return np.fliplr(arr)  # vertical axis reflection
    elif transform_idx == 5:
        return np.flipud(arr)  # horizontal axis reflection
    elif transform_idx == 6:
        return arr.T  # transpose (main diagonal)
    elif transform_idx == 7:
        return np.rot90(arr.T, 2)  # anti-diagonal reflection
    else:
        raise ValueError(f"Invalid transform_idx: {transform_idx}")


def invert_d4(arr: np.ndarray, transform_idx: int) -> np.ndarray:
    """Inverts the D4 dihedral transformation."""
    inverse_indices = {
        0: 0,
        1: 3,  # rot270 inverts rot90
        2: 2,  # rot180 inverts rot180
        3: 1,  # rot90 inverts rot270
        4: 4,  # fliplr is self-inverse
        5: 5,  # flipud is self-inverse
        6: 6,  # transpose is self-inverse
        7: 7,  # anti-diag is self-inverse
    }
    return apply_d4(arr, inverse_indices[transform_idx])


def apply_color_perm(arr: np.ndarray, perm: Sequence[int]) -> np.ndarray:
    """Permutes color integers according to permutation array."""
    perm_arr = np.asarray(perm, dtype=int)
    return perm_arr[arr]


def invert_color_perm(arr: np.ndarray, perm: Sequence[int]) -> np.ndarray:
    """Inverts the color permutation."""
    inv_perm = np.zeros(len(perm), dtype=int)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return inv_perm[arr]


class TaskAugmenter:
    """Manages geometric and color augmentations with exact inverse un-mapping."""

    def __init__(self, rng_seed: int = 42) -> None:
        self.rng = np.random.default_rng(rng_seed)

    def augment_grid(
        self,
        grid: Grid | np.ndarray,
        d4_idx: int,
        color_perm: Sequence[int] | None = None,
    ) -> Grid:
        arr = np.asarray(grid, dtype=int)
        if color_perm is not None:
            arr = apply_color_perm(arr, color_perm)
        arr = apply_d4(arr, d4_idx)
        return arr.tolist()

    def create_inverter(
        self,
        d4_idx: int,
        color_perm: Sequence[int] | None = None,
    ) -> Callable[[Grid | np.ndarray], Grid]:
        """Returns a callable that inverts the augmentation on predicted grids."""

        def invert(grid: Grid | np.ndarray) -> Grid:
            arr = np.asarray(grid, dtype=int)
            arr = invert_d4(arr, d4_idx)
            if color_perm is not None:
                arr = invert_color_perm(arr, color_perm)
            return arr.tolist()

        return invert

    def augment_task(
        self,
        task: dict[str, Any],
        d4_idx: int,
        color_perm: Sequence[int] | None = None,
    ) -> tuple[dict[str, Any], Callable[[Grid | np.ndarray], Grid]]:
        """Augments an entire task (train demonstrations and test inputs)."""
        aug_train = []
        for pair in task.get("train", []):
            aug_train.append(
                {
                    "input": self.augment_grid(pair["input"], d4_idx, color_perm),
                    "output": self.augment_grid(pair["output"], d4_idx, color_perm),
                }
            )

        aug_test = []
        for pair in task.get("test", []):
            item = {"input": self.augment_grid(pair["input"], d4_idx, color_perm)}
            if "output" in pair:
                item["output"] = self.augment_grid(pair["output"], d4_idx, color_perm)
            aug_test.append(item)

        aug_task = {"train": aug_train, "test": aug_test}
        inverter = self.create_inverter(d4_idx, color_perm)

        return aug_task, inverter

    def generate_balanced_augmentations(
        self,
        task: dict[str, Any],
        num_augs: int = 8,
    ) -> list[tuple[dict[str, Any], Callable[[Grid | np.ndarray], Grid]]]:
        """Generates a balanced batch of D4 and color augmentations for a task."""
        augs = []
        for i in range(min(num_augs, 8)):
            perm = self.rng.permutation(10).tolist()
            aug_task, inv_fn = self.augment_task(task, d4_idx=i, color_perm=perm)
            augs.append((aug_task, inv_fn))
        return augs
