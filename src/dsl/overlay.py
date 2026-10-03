from __future__ import annotations

from typing import Any
import numpy as np

Grid = list[list[int]]


def split_and_overlay(grid: Grid | np.ndarray, op: str, background: int = 0) -> Grid | None:
    """Splits a grid into halves (across a divider or midpoint) and applies logical overlay."""
    arr = np.asarray(grid, dtype=int)
    h, w = arr.shape

    g1: np.ndarray | None = None
    g2: np.ndarray | None = None

    if op.startswith("vsplit"):
        if w % 2 == 1 and "_div" in op:
            mid = w // 2
            g1 = arr[:, :mid]
            g2 = arr[:, mid + 1 :]
        elif w % 2 == 0 and "_div" not in op:
            mid = w // 2
            g1 = arr[:, :mid]
            g2 = arr[:, mid:]
        if g1 is None or g2 is None or g1.shape != g2.shape:
            return None
        if "mirror" in op:
            g2 = np.fliplr(g2)
    elif op.startswith("hsplit"):
        if h % 2 == 1 and "_div" in op:
            mid = h // 2
            g1 = arr[:mid, :]
            g2 = arr[mid + 1 :, :]
        elif h % 2 == 0 and "_div" not in op:
            mid = h // 2
            g1 = arr[:mid, :]
            g2 = arr[mid:, :]
        if g1 is None or g2 is None or g1.shape != g2.shape:
            return None
        if "mirror" in op:
            g2 = np.flipud(g2)
    else:
        return None

    if g1 is None or g2 is None:
        return None

    if op.endswith("_or"):
        res = np.where(g1 != background, g1, g2)
    elif op.endswith("_and"):
        res = np.where((g1 != background) & (g2 != background), g1, background)
    elif op.endswith("_xor"):
        res = np.where(
            (g1 != background) ^ (g2 != background),
            np.where(g1 != background, g1, g2),
            background,
        )
    else:
        return None

    return res.tolist()


SPLIT_OVERLAY_OPERATIONS = [
    "vsplit_div_or",
    "vsplit_div_and",
    "vsplit_div_xor",
    "vsplit_or",
    "vsplit_and",
    "vsplit_xor",
    "vsplit_mirror_or",
    "vsplit_mirror_and",
    "vsplit_mirror_xor",
    "hsplit_div_or",
    "hsplit_div_and",
    "hsplit_div_xor",
    "hsplit_or",
    "hsplit_and",
    "hsplit_xor",
    "hsplit_mirror_or",
    "hsplit_mirror_and",
    "hsplit_mirror_xor",
]
