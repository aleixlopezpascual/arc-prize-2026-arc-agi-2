#!/usr/bin/env python3
"""
ARC Prize 2026 (ARC-AGI-2) — Factored Object-Centric & Topological Program Synthesizer (v2.0)
Author: Vigneshwara Kireeti Alugunulla

Implements a 44-primitive Domain-Specific Language (DSL) over 2D integer grids (values 0..9)
spanning 6 algebraic families:
  1. Dihedral D4 & Framing Operators
  2. Connected-Component Object Graph Operators (4/8-connectivity, odd-one-out, symmetry, noise removal)
  3. Divider-Line Grid Sub-Partitioning & Panel Boolean Algebra (XOR, AND, OR, NOR)
  4. Topological Flood-Fill, Enclosure Cavity Filling & Morphological Boundary Operators
  5. 4-Directional Physical Sedimentation (Gravity) & Collinear Ray-Casting / Line Completion
  6. Symmetry Completion, Multi-Scale Tiling & Functional Color Morphism Induction

Uses Minimum Description Length (MDL) ranking across Depth-1, Depth-2, and Heuristic-Guided
Depth-3 Beam Search with Test-Time Transduction fallback to emit diverse Pass@2 predictions
(`attempt_1`, `attempt_2`) compliant with the official Kaggle ARC Prize 2026 schema.
"""

import os
import json
import time
from collections import deque
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Callable, Tuple, Optional, Any


# ============================================================================
# 0. Background Estimation & Connected-Component Object Graph Extraction
# ============================================================================

def get_background_color(g: np.ndarray) -> int:
    """Returns 0 if present as significant background, otherwise the mode color."""
    if g.size == 0:
        return 0
    zero_ratio = np.sum(g == 0) / g.size
    if zero_ratio >= 0.25:
        return 0
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(counts)])


@dataclass
class ARCObject:
    color: int
    area: int
    bbox: Tuple[int, int, int, int]  # (y0, y1_incl, x0, x1_incl)
    subgrid: np.ndarray
    mask: np.ndarray
    symmetry_score: float


def extract_connected_objects(
    g: np.ndarray,
    bg_color: int = 0,
    multicolor: bool = False,
    connectivity: int = 4,
) -> List[ARCObject]:
    """
    Extracts connected components from grid `g` using BFS.
    Supports single-color components (`multicolor=False`) or arbitrary non-bg shapes (`multicolor=True`).
    """
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: List[ARCObject] = []

    if connectivity == 8:
        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(h):
        for c in range(w):
            val = int(g[r, c])
            if val == bg_color or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        nval = int(g[nr, nc])
                        if nval != bg_color and (multicolor or nval == val):
                            visited[nr, nc] = True
                            q.append((nr, nc))

            ys = [pt[0] for pt in coords]
            xs = [pt[1] for pt in coords]
            y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
            mask = np.zeros((h, w), dtype=bool)
            for pr, pc in coords:
                mask[pr, pc] = True
            sub = np.where(mask[y0:y1+1, x0:x1+1], g[y0:y1+1, x0:x1+1], bg_color)

            # Compute horizontal + vertical symmetry score in [0, 1]
            sym_h = float(np.mean(sub == np.fliplr(sub)))
            sym_v = float(np.mean(sub == np.flipud(sub)))
            sym_score = 0.5 * (sym_h + sym_v)

            vals, counts = np.unique(sub[sub != bg_color], return_counts=True)
            dom_color = int(vals[np.argmax(counts)]) if len(vals) > 0 else val

            objects.append(
                ARCObject(
                    color=dom_color,
                    area=len(coords),
                    bbox=(y0, y1, x0, x1),
                    subgrid=sub,
                    mask=mask,
                    symmetry_score=sym_score,
                )
            )
    return objects


# ============================================================================
# 1. Family I: Dihedral D4 & Spatial Framing Primitives
# ============================================================================

def op_identity(g: np.ndarray) -> np.ndarray:
    return g.copy()

def op_rot90(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, 1).copy()

def op_rot180(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, 2).copy()

def op_rot270(g: np.ndarray) -> np.ndarray:
    return np.rot90(g, 3).copy()

def op_flip_h(g: np.ndarray) -> np.ndarray:
    return np.fliplr(g).copy()

def op_flip_v(g: np.ndarray) -> np.ndarray:
    return np.flipud(g).copy()

def op_transpose(g: np.ndarray) -> np.ndarray:
    return g.T.copy()

def op_anti_transpose(g: np.ndarray) -> np.ndarray:
    return np.rot90(np.fliplr(g), 1).copy()

def op_crop_nonzero(g: np.ndarray) -> np.ndarray:
    ys, xs = np.where(g != 0)
    if len(ys) == 0:
        return g.copy()
    return g[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()

def op_crop_non_background(g: np.ndarray) -> np.ndarray:
    bg = get_background_color(g)
    ys, xs = np.where(g != bg)
    if len(ys) == 0:
        return g.copy()
    return g[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()

def op_pad_1px_zero(g: np.ndarray) -> np.ndarray:
    if g.shape[0] >= 28 or g.shape[1] >= 28:
        return g.copy()
    return np.pad(g, pad_width=1, mode="constant", constant_values=0)


# ============================================================================
# 2. Family II: Connected-Component Object Graph Primitives
# ============================================================================

def op_isolate_largest_object(g: np.ndarray) -> np.ndarray:
    objs = extract_connected_objects(g, bg_color=0, multicolor=True, connectivity=8)
    if not objs:
        return g.copy()
    best = max(objs, key=lambda o: (o.area, -o.bbox[0], -o.bbox[2]))
    return best.subgrid.copy()

def op_isolate_smallest_object(g: np.ndarray) -> np.ndarray:
    objs = extract_connected_objects(g, bg_color=0, multicolor=True, connectivity=8)
    if not objs:
        return g.copy()
    best = min(objs, key=lambda o: (o.area, o.bbox[0], o.bbox[2]))
    return best.subgrid.copy()

def op_keep_largest_object_in_place(g: np.ndarray) -> np.ndarray:
    objs = extract_connected_objects(g, bg_color=0, multicolor=True, connectivity=8)
    if not objs:
        return g.copy()
    best = max(objs, key=lambda o: o.area)
    out = np.zeros_like(g)
    out[best.mask] = g[best.mask]
    return out

def op_isolate_unique_color_object(g: np.ndarray) -> np.ndarray:
    """Finds the object whose dominant color appears in only a single object (odd-one-out)."""
    objs = extract_connected_objects(g, bg_color=0, multicolor=False, connectivity=8)
    if not objs:
        return g.copy()
    color_counts: Dict[int, int] = {}
    for o in objs:
        color_counts[o.color] = color_counts.get(o.color, 0) + 1
    unique_objs = [o for o in objs if color_counts[o.color] == 1]
    if unique_objs:
        best = max(unique_objs, key=lambda o: o.area)
        return best.subgrid.copy()
    best = min(objs, key=lambda o: color_counts[o.color])
    return best.subgrid.copy()

def op_isolate_most_symmetric_object(g: np.ndarray) -> np.ndarray:
    objs = extract_connected_objects(g, bg_color=0, multicolor=True, connectivity=8)
    if not objs:
        return g.copy()
    best = max(objs, key=lambda o: (round(o.symmetry_score, 4), o.area))
    return best.subgrid.copy()

def op_remove_isolated_noise_1px(g: np.ndarray) -> np.ndarray:
    """Removes 1-pixel isolated speckles (4-connectivity area == 1) while keeping larger shapes."""
    objs = extract_connected_objects(g, bg_color=0, multicolor=True, connectivity=4)
    if not objs:
        return g.copy()
    out = np.zeros_like(g)
    kept = False
    for o in objs:
        if o.area > 1:
            out[o.mask] = g[o.mask]
            kept = True
    return out if kept else g.copy()

def op_recolor_objects_to_largest_color(g: np.ndarray) -> np.ndarray:
    objs = extract_connected_objects(g, bg_color=0, multicolor=False, connectivity=4)
    if len(objs) < 2:
        return g.copy()
    largest = max(objs, key=lambda o: o.area)
    out = g.copy()
    out[out != 0] = largest.color
    return out


# ============================================================================
# 3. Family III: Divider-Line Sub-Partitioning & Panel Boolean Algebra
# ============================================================================

def _split_grid_into_panels(g: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Detects if `g` is split into two equal-sized sub-panels by a full uniform divider row
    or column (or exact even halves if no explicit separator exists).
    Returns (panel_A, panel_B, dominant_fg_color) or None.
    """
    h, w = g.shape
    # Check vertical divider column at mid
    if w >= 3 and w % 2 == 1:
        mid_c = w // 2
        col = g[:, mid_c]
        if len(np.unique(col)) == 1 and int(col[0]) != 0:
            p1, p2 = g[:, :mid_c], g[:, mid_c+1:]
            vals, counts = np.unique(np.concatenate([p1[p1 != 0], p2[p2 != 0]]), return_counts=True)
            fg = int(vals[np.argmax(counts)]) if len(vals) > 0 else 1
            return p1, p2, fg
    # Check horizontal divider row at mid
    if h >= 3 and h % 2 == 1:
        mid_r = h // 2
        row = g[mid_r, :]
        if len(np.unique(row)) == 1 and int(row[0]) != 0:
            p1, p2 = g[:mid_r, :], g[mid_r+1:, :]
            vals, counts = np.unique(np.concatenate([p1[p1 != 0], p2[p2 != 0]]), return_counts=True)
            fg = int(vals[np.argmax(counts)]) if len(vals) > 0 else 1
            return p1, p2, fg
    # Check exact even split (left/right)
    if w >= 4 and w % 2 == 0:
        mid_c = w // 2
        p1, p2 = g[:, :mid_c], g[:, mid_c:]
        vals, counts = np.unique(g[g != 0], return_counts=True)
        fg = int(vals[np.argmax(counts)]) if len(vals) > 0 else 1
        return p1, p2, fg
    return None

def op_panel_boolean_xor(g: np.ndarray) -> np.ndarray:
    sp = _split_grid_into_panels(g)
    if sp is None:
        return g.copy()
    p1, p2, fg = sp
    mask = (p1 != 0) ^ (p2 != 0)
    return np.where(mask, fg, 0).astype(g.dtype)

def op_panel_boolean_and(g: np.ndarray) -> np.ndarray:
    sp = _split_grid_into_panels(g)
    if sp is None:
        return g.copy()
    p1, p2, fg = sp
    mask = (p1 != 0) & (p2 != 0)
    return np.where(mask, fg, 0).astype(g.dtype)

def op_panel_boolean_or(g: np.ndarray) -> np.ndarray:
    sp = _split_grid_into_panels(g)
    if sp is None:
        return g.copy()
    p1, p2, _ = sp
    return np.where(p1 != 0, p1, p2).copy()

def op_panel_boolean_nor(g: np.ndarray) -> np.ndarray:
    sp = _split_grid_into_panels(g)
    if sp is None:
        return g.copy()
    p1, p2, fg = sp
    mask = (p1 == 0) & (p2 == 0)
    return np.where(mask, fg, 0).astype(g.dtype)


# ============================================================================
# 4. Family IV: Topological Enclosure, Flood-Fill & Morphology Primitives
# ============================================================================

def op_fill_bounding_box_interior(g: np.ndarray) -> np.ndarray:
    """Fills 0-cells strictly inside the bounding box of non-zero pixels with color 5."""
    out = g.copy()
    ys, xs = np.where(g != 0)
    if len(ys) < 4:
        return out
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    if y1 - y0 >= 2 and x1 - x0 >= 2:
        sub = out[y0+1:y1, x0+1:x1]
        sub[sub == 0] = 5
    return out

def op_fill_closed_holes(g: np.ndarray) -> np.ndarray:
    """
    True topological flood-fill from all 4 borders to identify exterior 0-cells;
    any unvisited 0-cell is a topologically enclosed hole and is filled with color 2 (or 5).
    """
    h, w = g.shape
    if h < 3 or w < 3:
        return g.copy()
    exterior = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if g[r, c] == 0 and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if g[r, c] == 0 and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    while q:
        cr, cc = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not exterior[nr, nc] and g[nr, nc] == 0:
                exterior[nr, nc] = True
                q.append((nr, nc))
    out = g.copy()
    holes = (g == 0) & (~exterior)
    if np.any(holes):
        out[holes] = 2
    return out

def op_hollow_interior(g: np.ndarray) -> np.ndarray:
    """Morphological boundary extraction: zeroes out pixels whose 4-neighbors are all non-zero."""
    h, w = g.shape
    if h < 3 or w < 3:
        return g.copy()
    out = g.copy()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if g[r, c] != 0:
                if g[r-1, c] != 0 and g[r+1, c] != 0 and g[r, c-1] != 0 and g[r, c+1] != 0:
                    out[r, c] = 0
    return out

def op_outline_nonzero_1px(g: np.ndarray) -> np.ndarray:
    """Draws a 1-pixel halo of color 8 around existing non-zero pixels into adjacent 0-cells."""
    h, w = g.shape
    out = g.copy()
    nz_r, nz_c = np.where(g != 0)
    for r, c in zip(nz_r, nz_c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 0:
                    out[nr, nc] = 8
    return out


# ============================================================================
# 5. Family V: Physical Sedimentation (4D Gravity) & Collinear Ray-Casting
# ============================================================================

def op_gravity_down(g: np.ndarray) -> np.ndarray:
    out = np.zeros_like(g)
    h, w = g.shape
    for c in range(w):
        nz = g[:, c][g[:, c] != 0]
        if len(nz) > 0:
            out[h - len(nz):, c] = nz
    return out

def op_gravity_up(g: np.ndarray) -> np.ndarray:
    out = np.zeros_like(g)
    h, w = g.shape
    for c in range(w):
        nz = g[:, c][g[:, c] != 0]
        if len(nz) > 0:
            out[:len(nz), c] = nz
    return out

def op_gravity_left(g: np.ndarray) -> np.ndarray:
    out = np.zeros_like(g)
    h, _ = g.shape
    for r in range(h):
        nz = g[r, :][g[r, :] != 0]
        if len(nz) > 0:
            out[r, :len(nz)] = nz
    return out

def op_gravity_right(g: np.ndarray) -> np.ndarray:
    out = np.zeros_like(g)
    h, w = g.shape
    for r in range(h):
        nz = g[r, :][g[r, :] != 0]
        if len(nz) > 0:
            out[r, w - len(nz):] = nz
    return out

def op_connect_collinear_anchors(g: np.ndarray) -> np.ndarray:
    """
    Connects pairs of same-colored non-zero pixels sharing the same row or column
    with straight lines of that color across intervening 0-cells.
    """
    h, w = g.shape
    out = g.copy()
    # Horizontal ray connections
    for r in range(h):
        row = g[r, :]
        colors = [int(c) for c in np.unique(row) if c != 0]
        for col_val in colors:
            xs = np.where(row == col_val)[0]
            if len(xs) >= 2:
                for x in range(xs.min(), xs.max() + 1):
                    if out[r, x] == 0:
                        out[r, x] = col_val
    # Vertical ray connections
    for c in range(w):
        col = g[:, c]
        colors = [int(v) for v in np.unique(col) if v != 0]
        for col_val in colors:
            ys = np.where(col == col_val)[0]
            if len(ys) >= 2:
                for y in range(ys.min(), ys.max() + 1):
                    if out[y, c] == 0:
                        out[y, c] = col_val
    return out

def op_crosshair_from_single_pixels(g: np.ndarray) -> np.ndarray:
    """Projects full-row and full-column rays from isolated non-zero pixels."""
    h, w = g.shape
    out = g.copy()
    ys, xs = np.where(g != 0)
    if len(ys) == 0 or len(ys) > 6:
        return out
    for r, c in zip(ys, xs):
        val = int(g[r, c])
        out[r, out[r, :] == 0] = val
        out[out[:, c] == 0, c] = val
    return out


# ============================================================================
# 6. Family VI: Symmetry Completion, Multi-Scale Tiling & Color Algebra
# ============================================================================

def op_complete_symmetry_h(g: np.ndarray) -> np.ndarray:
    flipped = np.fliplr(g)
    return np.where(g != 0, g, flipped).copy()

def op_complete_symmetry_v(g: np.ndarray) -> np.ndarray:
    flipped = np.flipud(g)
    return np.where(g != 0, g, flipped).copy()

def op_complete_symmetry_rot180(g: np.ndarray) -> np.ndarray:
    rot = np.rot90(g, 2)
    return np.where(g != 0, g, rot).copy()

def op_complete_symmetry_diag(g: np.ndarray) -> np.ndarray:
    if g.shape[0] != g.shape[1]:
        return g.copy()
    tr = g.T
    return np.where(g != 0, g, tr).copy()

def op_tile_2x2(g: np.ndarray) -> np.ndarray:
    if g.shape[0] > 15 or g.shape[1] > 15:
        return g.copy()
    return np.tile(g, (2, 2))

def op_tile_3x3(g: np.ndarray) -> np.ndarray:
    if g.shape[0] > 10 or g.shape[1] > 10:
        return g.copy()
    return np.tile(g, (3, 3))

def op_mirror_quad_tile(g: np.ndarray) -> np.ndarray:
    if g.shape[0] > 15 or g.shape[1] > 15:
        return g.copy()
    top = np.hstack([g, np.fliplr(g)])
    bot = np.vstack([top, np.flipud(top)])
    return bot

def op_upscale_2x(g: np.ndarray) -> np.ndarray:
    if g.shape[0] > 15 or g.shape[1] > 15:
        return g.copy()
    return np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)

def op_upscale_3x(g: np.ndarray) -> np.ndarray:
    if g.shape[0] > 10 or g.shape[1] > 10:
        return g.copy()
    return np.repeat(np.repeat(g, 3, axis=0), 3, axis=1)

def op_downsample_2x(g: np.ndarray) -> np.ndarray:
    if g.shape[0] < 2 or g.shape[1] < 2 or g.shape[0] % 2 != 0 or g.shape[1] % 2 != 0:
        return g.copy()
    return g[::2, ::2].copy()

def op_majority_color_recolor(g: np.ndarray) -> np.ndarray:
    vals, counts = np.unique(g[g != 0], return_counts=True)
    if len(vals) == 0:
        return g.copy()
    maj = int(vals[np.argmax(counts)])
    out = g.copy()
    out[out != 0] = maj
    return out

def op_minority_color_recolor(g: np.ndarray) -> np.ndarray:
    vals, counts = np.unique(g[g != 0], return_counts=True)
    if len(vals) == 0:
        return g.copy()
    min_c = int(vals[np.argmin(counts)])
    out = g.copy()
    out[out != 0] = min_c
    return out

def op_swap_top2_colors(g: np.ndarray) -> np.ndarray:
    vals, counts = np.unique(g[g != 0], return_counts=True)
    if len(vals) < 2:
        return g.copy()
    order = np.argsort(-counts)
    c1, c2 = int(vals[order[0]]), int(vals[order[1]])
    out = g.copy()
    out[g == c1] = c2
    out[g == c2] = c1
    return out


# ============================================================================
# Master Registry of 44 Composable ARC-AGI-2 Primitives
# ============================================================================

PRIMITIVES: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    # Family I: Dihedral & Framing
    ("identity", op_identity),
    ("rot90", op_rot90),
    ("rot180", op_rot180),
    ("rot270", op_rot270),
    ("flip_h", op_flip_h),
    ("flip_v", op_flip_v),
    ("transpose", op_transpose),
    ("anti_transpose", op_anti_transpose),
    ("crop_nonzero", op_crop_nonzero),
    ("crop_non_background", op_crop_non_background),
    ("pad_1px_zero", op_pad_1px_zero),
    # Family II: Object Graph
    ("isolate_largest_object", op_isolate_largest_object),
    ("isolate_smallest_object", op_isolate_smallest_object),
    ("keep_largest_object_in_place", op_keep_largest_object_in_place),
    ("isolate_unique_color_object", op_isolate_unique_color_object),
    ("isolate_most_symmetric_object", op_isolate_most_symmetric_object),
    ("remove_isolated_noise_1px", op_remove_isolated_noise_1px),
    ("recolor_objects_to_largest", op_recolor_objects_to_largest_color),
    # Family III: Divider Panel Boolean Algebra
    ("panel_boolean_xor", op_panel_boolean_xor),
    ("panel_boolean_and", op_panel_boolean_and),
    ("panel_boolean_or", op_panel_boolean_or),
    ("panel_boolean_nor", op_panel_boolean_nor),
    # Family IV: Topological Enclosure & Morphology
    ("fill_bbox_interior", op_fill_bounding_box_interior),
    ("fill_closed_holes", op_fill_closed_holes),
    ("hollow_interior", op_hollow_interior),
    ("outline_nonzero_1px", op_outline_nonzero_1px),
    # Family V: Physical Sedimentation & Ray-Casting
    ("gravity_down", op_gravity_down),
    ("gravity_up", op_gravity_up),
    ("gravity_left", op_gravity_left),
    ("gravity_right", op_gravity_right),
    ("connect_collinear_anchors", op_connect_collinear_anchors),
    ("crosshair_from_single_pixels", op_crosshair_from_single_pixels),
    # Family VI: Symmetry, Multi-Scale Tiling & Color Algebra
    ("complete_symmetry_h", op_complete_symmetry_h),
    ("complete_symmetry_v", op_complete_symmetry_v),
    ("complete_symmetry_rot180", op_complete_symmetry_rot180),
    ("complete_symmetry_diag", op_complete_symmetry_diag),
    ("tile_2x2", op_tile_2x2),
    ("tile_3x3", op_tile_3x3),
    ("mirror_quad_tile", op_mirror_quad_tile),
    ("upscale_2x", op_upscale_2x),
    ("upscale_3x", op_upscale_3x),
    ("downsample_2x", op_downsample_2x),
    ("majority_recolor", op_majority_color_recolor),
    ("minority_recolor", op_minority_color_recolor),
    ("swap_top2_colors", op_swap_top2_colors),
]


# ============================================================================
# Factored Color Morphism Induction & Heuristic Program Scoring
# ============================================================================

def infer_color_mapping(train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Dict[int, int]]:
    """
    Verifies in O(HWK) linear time whether outputs are obtained by a consistent
    functional color morphism M: {0..9} -> {0..9} after spatial transformation.
    """
    mapping: Dict[int, int] = {}
    for inp, out in train_pairs:
        if inp.shape != out.shape:
            return None
        for a, b in zip(inp.flat, out.flat):
            a_i, b_i = int(a), int(b)
            if a_i in mapping and mapping[a_i] != b_i:
                return None
            mapping[a_i] = b_i
    return mapping


def apply_color_mapping(g: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    out = g.copy()
    for k, v in mapping.items():
        out[g == k] = v
    return out


def score_candidate_on_pairs(
    transformed_pairs: List[Tuple[np.ndarray, np.ndarray]]
) -> float:
    """
    Computes a continuous heuristic alignment score in [0.0, 1.0] for beam search guidance:
      - 0.40 weight: Exact grid shape match
      - 0.45 weight: Pixel-wise accuracy (when shapes match)
      - 0.15 weight: Foreground non-zero pixel count ratio similarity
    """
    total = 0.0
    for pred, gt in transformed_pairs:
        if pred.shape == gt.shape:
            s_shape = 0.40
            s_pix = 0.45 * float(np.mean(pred == gt))
        else:
            hr = min(pred.shape[0], gt.shape[0]) / max(pred.shape[0], gt.shape[0], 1)
            wr = min(pred.shape[1], gt.shape[1]) / max(pred.shape[1], gt.shape[1], 1)
            s_shape = 0.20 * (hr * wr)
            s_pix = 0.0
        nz_p = int(np.sum(pred != 0))
        nz_g = int(np.sum(gt != 0))
        s_cnt = 0.15 * (min(nz_p, nz_g) + 1.0) / (max(nz_p, nz_g) + 1.0)
        total += (s_shape + s_pix + s_cnt)
    return total / max(len(transformed_pairs), 1)


# ============================================================================
# Main Synthesizer Engine: MDL Depth-1/2/3 Beam Search + Pass@2 Transduction
# ============================================================================

class ARCSymbolicSynthesizer:
    """
    Synthesizes Depth-1, Depth-2, and Heuristic-Guided Depth-3 DSL programs + learned
    color morphisms that satisfy demonstration pairs, emitting top-2 distinct predictions.
    """

    def __init__(self, beam_width: int = 12, enable_depth3: bool = True):
        self.beam_width = beam_width
        self.enable_depth3 = enable_depth3

    def solve_task(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> Tuple[List[np.ndarray], str]:
        exact_candidates: List[Tuple[int, str, Callable[[np.ndarray], np.ndarray]]] = []
        scored_beam: List[Tuple[float, str, Callable[[np.ndarray], np.ndarray], List[np.ndarray]]] = []

        # --- Stage 1: Depth-1 Primitives + Factored Color Morphism ---
        for name1, fn1 in PRIMITIVES:
            try:
                t_inputs = [fn1(inp) for inp, _ in train_pairs]
                t_pairs = list(zip(t_inputs, [out for _, out in train_pairs]))
                if all(tp.shape == out.shape and np.array_equal(tp, out) for tp, out in t_pairs):
                    mdl_cost = 1 if name1 == "identity" else 2
                    exact_candidates.append((mdl_cost, name1, fn1))
                else:
                    cmap = infer_color_mapping(t_pairs)
                    if cmap is not None:
                        def make_d1_cmap(f1=fn1, cm=cmap):
                            return lambda x: apply_color_mapping(f1(x), cm)
                        exact_candidates.append((3, f"{name1}+color_map", make_d1_cmap()))
                if name1 != "identity":
                    h_score = score_candidate_on_pairs(t_pairs)
                    scored_beam.append((h_score, name1, fn1, t_inputs))
            except Exception:
                continue

        # --- Stage 2: Depth-2 Compositions + Factored Color Morphism ---
        slow_bfs_ops = {
            "isolate_largest_object",
            "isolate_smallest_object",
            "keep_largest_object_in_place",
            "isolate_unique_color_object",
            "isolate_most_symmetric_object",
            "remove_isolated_noise_1px",
            "recolor_objects_to_largest",
            "fill_closed_holes",
            "hollow_interior",
            "outline_nonzero_1px",
        }
        if len(exact_candidates) < 2:
            for name1, fn1 in PRIMITIVES:
                if name1 == "identity":
                    continue
                try:
                    mid_inputs = [fn1(inp) for inp, _ in train_pairs]
                except Exception:
                    continue
                for name2, fn2 in PRIMITIVES:
                    if name2 == "identity" or name1 == name2:
                        continue
                    # Avoid running slow Python BFS inside inner loop unless name1 is a framing/crop op
                    if name2 in slow_bfs_ops and name1 not in ("crop_nonzero", "crop_non_background", "rot90", "flip_h", "flip_v"):
                        continue
                    try:
                        t_inputs = [fn2(m) for m in mid_inputs]
                        t_pairs = list(zip(t_inputs, [out for _, out in train_pairs]))
                        if all(tp.shape == out.shape and np.array_equal(tp, out) for tp, out in t_pairs):
                            def make_d2(f1=fn1, f2=fn2):
                                return lambda x: f2(f1(x))
                            exact_candidates.append((4, f"{name1}->{name2}", make_d2()))
                        else:
                            cmap = infer_color_mapping(t_pairs)
                            if cmap is not None:
                                def make_d2_cmap(f1=fn1, f2=fn2, cm=cmap):
                                    return lambda x: apply_color_mapping(f2(f1(x)), cm)
                                exact_candidates.append((5, f"{name1}->{name2}+color_map", make_d2_cmap()))
                        h_score = score_candidate_on_pairs(t_pairs)
                        if h_score > 0.55:
                            def make_d2_beam(f1=fn1, f2=fn2):
                                return lambda x: f2(f1(x))
                            scored_beam.append((h_score, f"{name1}->{name2}", make_d2_beam(), t_inputs))
                    except Exception:
                        continue

        # --- Stage 3: Heuristic-Guided Depth-3 Beam Search ---
        if len(exact_candidates) < 2 and self.enable_depth3 and scored_beam:
            scored_beam.sort(key=lambda item: item[0], reverse=True)
            top_beam = scored_beam[: self.beam_width]
            for _, prefix_name, prefix_fn, mid_inputs in top_beam:
                if "->" not in prefix_name:
                    continue  # Expand depth-2 states to depth-3
                for name3, fn3 in PRIMITIVES:
                    if name3 == "identity" or name3 in slow_bfs_ops:
                        continue
                    try:
                        t_inputs = [fn3(m) for m in mid_inputs]
                        t_pairs = list(zip(t_inputs, [out for _, out in train_pairs]))
                        if all(tp.shape == out.shape and np.array_equal(tp, out) for tp, out in t_pairs):
                            def make_d3(pf=prefix_fn, f3=fn3):
                                return lambda x: f3(pf(x))
                            exact_candidates.append((6, f"{prefix_name}->{name3}", make_d3()))
                        else:
                            cmap = infer_color_mapping(t_pairs)
                            if cmap is not None:
                                def make_d3_cmap(pf=prefix_fn, f3=fn3, cm=cmap):
                                    return lambda x: apply_color_mapping(f3(pf(x)), cm)
                                exact_candidates.append((7, f"{prefix_name}->{name3}+color_map", make_d3_cmap()))
                    except Exception:
                        continue

        # Sort exact candidates by Minimum Description Length (MDL cost)
        exact_candidates.sort(key=lambda item: item[0])

        preds: List[np.ndarray] = []
        used_programs: List[str] = []

        for _, prog_name, prog_fn in exact_candidates:
            try:
                pred = np.asarray(prog_fn(test_input), dtype=int)
                if pred.ndim == 2 and 1 <= pred.shape[0] <= 30 and 1 <= pred.shape[1] <= 30:
                    pred = np.clip(pred, 0, 9)
                    if not any(p.shape == pred.shape and np.array_equal(p, pred) for p in preds):
                        preds.append(pred)
                        used_programs.append(prog_name)
                    if len(preds) == 2:
                        break
            except Exception:
                continue

        # --- Stage 4: Test-Time Transduction Fallback (Highest-Scoring Approximate Programs) ---
        if len(preds) < 2 and scored_beam:
            scored_beam.sort(key=lambda item: item[0], reverse=True)
            for h_score, prog_name, prog_fn, t_inputs in scored_beam[:16]:
                try:
                    t_pairs = list(zip(t_inputs, [out for _, out in train_pairs]))
                    cmap = infer_color_mapping(t_pairs)
                    raw_pred = prog_fn(test_input)
                    if cmap is not None:
                        raw_pred = apply_color_mapping(raw_pred, cmap)
                    pred = np.clip(np.asarray(raw_pred, dtype=int), 0, 9)
                    if pred.ndim == 2 and 1 <= pred.shape[0] <= 30 and 1 <= pred.shape[1] <= 30:
                        if not any(p.shape == pred.shape and np.array_equal(p, pred) for p in preds):
                            preds.append(pred)
                            used_programs.append(f"transduction({prog_name},score={h_score:.2f})")
                        if len(preds) == 2:
                            break
                except Exception:
                    continue

        # Final deterministic safety net so output always has exactly 2 valid 2D grids
        fallback_ops = [op_crop_nonzero, op_complete_symmetry_h, op_identity]
        for fb_fn in fallback_ops:
            if len(preds) >= 2:
                break
            try:
                fb_pred = np.clip(np.asarray(fb_fn(test_input), dtype=int), 0, 9)
                if fb_pred.ndim == 2 and 1 <= fb_pred.shape[0] <= 30 and 1 <= fb_pred.shape[1] <= 30:
                    if not any(p.shape == fb_pred.shape and np.array_equal(p, fb_pred) for p in preds):
                        preds.append(fb_pred)
                        used_programs.append(f"fallback({fb_fn.__name__})")
            except Exception:
                continue
        while len(preds) < 2:
            preds.append(np.clip(test_input.copy(), 0, 9))
            used_programs.append("fallback_identity")

        return preds, " | ".join(used_programs)


# ============================================================================
# Official Kaggle ARC Prize 2026 Challenge File Runner
# ============================================================================

def solve_kaggle_challenges_file(
    challenges_json_path: str,
    output_submission_path: str,
    solutions_json_path: Optional[str] = None,
    max_tasks: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Loads an official ARC-AGI JSON file (e.g. `/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json`),
    solves every test instance per task, and writes a strictly valid Kaggle `submission.json`.
    """
    t0 = time.perf_counter()
    with open(challenges_json_path, "r", encoding="utf-8") as f:
        challenges = json.load(f)

    solutions = None
    if solutions_json_path and os.path.exists(solutions_json_path):
        with open(solutions_json_path, "r", encoding="utf-8") as f:
            solutions = json.load(f)

    synth = ARCSymbolicSynthesizer(beam_width=10, enable_depth3=True)
    submission: Dict[str, List[Dict[str, List[List[int]]]]] = {}

    task_ids = list(challenges.keys())
    if max_tasks is not None:
        task_ids = task_ids[:max_tasks]

    exact_solved = 0
    total_test_items = 0

    for tid in task_ids:
        task = challenges[tid]
        train_pairs = [
            (np.array(pair["input"], dtype=int), np.array(pair["output"], dtype=int))
            for pair in task["train"]
        ]
        task_preds = []
        for idx, test_pair in enumerate(task["test"]):
            test_inp = np.array(test_pair["input"], dtype=int)
            preds, _ = synth.solve_task(train_pairs, test_inp)
            task_preds.append({
                "attempt_1": preds[0].tolist(),
                "attempt_2": preds[1].tolist(),
            })
            total_test_items += 1
            if solutions and tid in solutions and idx < len(solutions[tid]):
                gt = np.array(solutions[tid][idx], dtype=int)
                if (preds[0].shape == gt.shape and np.array_equal(preds[0], gt)) or (
                    preds[1].shape == gt.shape and np.array_equal(preds[1], gt)
                ):
                    exact_solved += 1
        submission[tid] = task_preds

    with open(output_submission_path, "w", encoding="utf-8") as f:
        json.dump(submission, f)

    return {
        "challenges_file": challenges_json_path,
        "tasks_processed": len(task_ids),
        "test_instances_processed": total_test_items,
        "exact_solved_if_solutions_provided": exact_solved if solutions else None,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "output_submission_path": output_submission_path,
    }


if __name__ == "__main__":
    # Auto-detect official Kaggle runtime paths recursively under /kaggle/input, then local data/ fallback
    found_path = None
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "arc-agi_test_challenges.json" in files:
                found_path = os.path.join(root, "arc-agi_test_challenges.json")
                break
    if found_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        local_candidate = os.path.join(base_dir, "data", "arc-agi_test_challenges.json")
        if os.path.exists(local_candidate):
            found_path = local_candidate

    out_path = "/kaggle/working/submission.json" if os.path.exists("/kaggle/working") else "submission.json"
    if found_path is not None:
        stats = solve_kaggle_challenges_file(found_path, out_path)
        print(json.dumps(stats, indent=2))
    else:
        print("Warning: arc-agi_test_challenges.json not found.")
