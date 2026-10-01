"""Domain-Specific Language (DSL) primitives and object perception engine for ARC-AGI."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

Grid = list[list[int]]


def to_grid(grid: Grid | np.ndarray) -> Grid:
    """Converts a NumPy array or list-based grid to standard 2D list of ints."""
    if isinstance(grid, np.ndarray):
        return grid.astype(int).tolist()
    return [list(row) for row in grid]


def to_array(grid: Grid | np.ndarray) -> np.ndarray:
    """Converts a 2D grid to a NumPy int32 array."""
    if isinstance(grid, np.ndarray):
        return grid.astype(np.int32)
    return np.array(grid, dtype=np.int32)


@dataclass
class Object:
    """Structured representation of a connected component or object in an ARC grid."""

    coords: set[tuple[int, int]]
    color: int
    bbox: tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    height: int
    width: int
    size: int
    colors: set[int] = field(default_factory=set)

    def crop_from(self, grid: Grid | np.ndarray) -> Grid:
        """Crops this object's bounding box from the given grid."""
        return crop_to_bounding_box(grid, self.bbox)


def find_connected_components(
    grid: Grid | np.ndarray,
    background: int | None = 0,
    connectivity: int = 4,
    monochromatic: bool = True,
) -> list[Object]:
    """Extracts connected components from a grid.

    Args:
        grid: 2D grid of integers.
        background: Pixel value considered background to ignore (None to include all).
        connectivity: 4 for orthogonal neighbors, 8 to include diagonals.
        monochromatic: If True, components are partitioned by identical color.
                       If False, adjacent non-background pixels form a multi-color object.

    Returns:
        List of Object instances sorted predictably by (min_r, min_c).
    """
    arr = to_array(grid)
    h, w = arr.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[Object] = []

    if connectivity == 8:
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(h):
        for c in range(w):
            val = int(arr[r, c])
            if visited[r, c] or (background is not None and val == background):
                continue

            comp_coords: set[tuple[int, int]] = set()
            comp_colors: set[int] = set()
            queue = deque([(r, c)])
            visited[r, c] = True

            while queue:
                curr_r, curr_c = queue.popleft()
                curr_val = int(arr[curr_r, curr_c])
                comp_coords.add((curr_r, curr_c))
                comp_colors.add(curr_val)

                for dr, dc in deltas:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        nval = int(arr[nr, nc])
                        if background is not None and nval == background:
                            continue
                        if monochromatic and nval != val:
                            continue
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            min_r = min(cr for cr, _ in comp_coords)
            max_r = max(cr for cr, _ in comp_coords)
            min_c = min(cc for _, cc in comp_coords)
            max_c = max(cc for _, cc in comp_coords)
            obj_h = max_r - min_r + 1
            obj_w = max_c - min_c + 1

            dominant_color = val if monochromatic else Counter(
                arr[cr, cc] for cr, cc in comp_coords
            ).most_common(1)[0][0]

            objects.append(
                Object(
                    coords=comp_coords,
                    color=int(dominant_color),
                    bbox=(min_r, min_c, max_r, max_c),
                    height=obj_h,
                    width=obj_w,
                    size=len(comp_coords),
                    colors=comp_colors,
                )
            )

    objects.sort(key=lambda o: (o.bbox[0], o.bbox[1]))
    return objects


def find_multicolor_objects(
    grid: Grid | np.ndarray,
    background: int | None = 0,
    connectivity: int = 4,
) -> list[Object]:
    """Finds contiguous multi-colored components excluding background."""
    return find_connected_components(
        grid, background=background, connectivity=connectivity, monochromatic=False
    )


def crop_to_bounding_box(
    grid: Grid | np.ndarray,
    bbox: tuple[int, int, int, int],
) -> Grid:
    """Crops the grid to the specified bounding box (min_r, min_c, max_r, max_c)."""
    arr = to_array(grid)
    min_r, min_c, max_r, max_c = bbox
    min_r = max(0, min_r)
    min_c = max(0, min_c)
    max_r = min(arr.shape[0] - 1, max_r)
    max_c = min(arr.shape[1] - 1, max_c)
    if min_r > max_r or min_c > max_c:
        return [[]]
    return arr[min_r : max_r + 1, min_c : max_c + 1].tolist()


def crop_to_content(
    grid: Grid | np.ndarray,
    background: int = 0,
) -> Grid:
    """Crops to the bounding box containing all non-background pixels."""
    arr = to_array(grid)
    mask = arr != background
    if not np.any(mask):
        return [[int(arr[0, 0])]] if arr.size > 0 else [[]]
    rows, cols = np.where(mask)
    bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
    return crop_to_bounding_box(arr, bbox)


def pad_grid(
    grid: Grid | np.ndarray,
    target_h: int,
    target_w: int,
    pad_val: int = 0,
    align: str = "top_left",
) -> Grid:
    """Pads grid with pad_val to reach target_h x target_w."""
    arr = to_array(grid)
    curr_h, curr_w = arr.shape
    new_h = max(curr_h, target_h)
    new_w = max(curr_w, target_w)
    out = np.full((new_h, new_w), pad_val, dtype=np.int32)

    if align == "center":
        start_r = (new_h - curr_h) // 2
        start_c = (new_w - curr_w) // 2
    else:  # top_left
        start_r = 0
        start_c = 0

    out[start_r : start_r + curr_h, start_c : start_c + curr_w] = arr
    return out.tolist()


def scale_grid(grid: Grid | np.ndarray, factor: int) -> Grid:
    """Scales grid by integer factor using nearest-neighbor / block repeat."""
    if factor <= 1:
        return to_grid(grid)
    arr = to_array(grid)
    repeated = np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)
    return repeated.tolist()


def gravity(
    grid: Grid | np.ndarray,
    direction: str = "down",
    obstacle_colors: Sequence[int] | set[int] | None = None,
    background: int = 0,
) -> Grid:
    """Simulates gravity drop in specified direction ('down', 'up', 'left', 'right').

    Non-background cells move in the given direction until blocked by grid boundary,
    an obstacle pixel, or another settled pixel.
    """
    arr = to_array(grid)
    obstacles = set(obstacle_colors) if obstacle_colors is not None else set()

    if direction == "up":
        flipped = np.flipud(arr)
        res = gravity(flipped, direction="down", obstacle_colors=obstacles, background=background)
        return np.flipud(to_array(res)).tolist()

    if direction == "right":
        trans = np.transpose(arr)
        res = gravity(trans, direction="down", obstacle_colors=obstacles, background=background)
        return np.transpose(to_array(res)).tolist()

    if direction == "left":
        trans = np.transpose(arr)
        flipped = np.flipud(trans)
        res = gravity(flipped, direction="down", obstacle_colors=obstacles, background=background)
        unflipped = np.flipud(to_array(res))
        return np.transpose(unflipped).tolist()

    # direction == 'down'
    h, w = arr.shape
    out = np.full((h, w), background, dtype=np.int32)

    for c in range(w):
        col = arr[:, c].tolist()
        for r_idx, val in enumerate(col):
            if val in obstacles:
                out[r_idx, c] = val

        write_r = h - 1
        for r_idx in range(h - 1, -1, -1):
            val = col[r_idx]
            if val in obstacles:
                write_r = r_idx - 1
            elif val != background:
                while write_r >= 0 and out[write_r, c] != background:
                    write_r -= 1
                if write_r >= 0:
                    out[write_r, c] = val
                    write_r -= 1

    return out.tolist()


def flood_fill(
    grid: Grid | np.ndarray,
    r: int,
    c: int,
    new_color: int,
    connectivity: int = 4,
) -> Grid:
    """Flood-fills from (r, c) replacing contiguous target color with new_color."""
    arr = to_array(grid).copy()
    h, w = arr.shape
    if not (0 <= r < h and 0 <= c < w):
        return arr.tolist()

    target_color = int(arr[r, c])
    if target_color == new_color:
        return arr.tolist()

    if connectivity == 8:
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    queue = deque([(r, c)])
    arr[r, c] = new_color

    while queue:
        curr_r, curr_c = queue.popleft()
        for dr, dc in deltas:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < h and 0 <= nc < w and arr[nr, nc] == target_color:
                arr[nr, nc] = new_color
                queue.append((nr, nc))

    return arr.tolist()


def rotate_90(grid: Grid | np.ndarray) -> Grid:
    """Rotates grid 90 degrees clockwise."""
    return np.rot90(to_array(grid), -1).tolist()


def rotate_180(grid: Grid | np.ndarray) -> Grid:
    """Rotates grid 180 degrees."""
    return np.rot90(to_array(grid), 2).tolist()


def rotate_270(grid: Grid | np.ndarray) -> Grid:
    """Rotates grid 270 degrees clockwise (90 counter-clockwise)."""
    return np.rot90(to_array(grid), 1).tolist()


def reflect_horizontal(grid: Grid | np.ndarray) -> Grid:
    """Reflects grid across the horizontal axis (vertical flip)."""
    return np.flipud(to_array(grid)).tolist()


def reflect_vertical(grid: Grid | np.ndarray) -> Grid:
    """Reflects grid across the vertical axis (horizontal mirror)."""
    return np.fliplr(to_array(grid)).tolist()


def transpose(grid: Grid | np.ndarray) -> Grid:
    """Transposes the grid (rows become columns)."""
    return np.transpose(to_array(grid)).tolist()


def detect_symmetry(grid: Grid | np.ndarray) -> dict[str, bool]:
    """Detects various axes of symmetry in the grid."""
    arr = to_array(grid)
    h, w = arr.shape
    horiz = bool(np.array_equal(arr, np.flipud(arr)))
    vert = bool(np.array_equal(arr, np.fliplr(arr)))
    diag = bool(np.array_equal(arr, arr.T)) if h == w else False
    anti_diag = bool(np.array_equal(arr, np.rot90(arr.T, 2))) if h == w else False

    return {
        "horizontal": horiz,
        "vertical": vert,
        "diagonal": diag,
        "anti_diagonal": anti_diag,
    }


def mirror_append(
    grid: Grid | np.ndarray,
    axis: str = "vertical",
) -> Grid:
    """Doubles the grid by appending its mirror reflection along the axis."""
    arr = to_array(grid)
    if axis == "vertical":
        flipped = np.fliplr(arr)
        return np.concatenate([arr, flipped], axis=1).tolist()
    if axis == "horizontal":
        flipped = np.flipud(arr)
        return np.concatenate([arr, flipped], axis=0).tolist()
    if axis == "both":
        top = np.concatenate([arr, np.fliplr(arr)], axis=1)
        bottom = np.flipud(top)
        return np.concatenate([top, bottom], axis=0).tolist()
    raise ValueError(f"Unknown axis: {axis}")


def complete_symmetry(
    grid: Grid | np.ndarray,
    axis: str = "vertical",
    source: str = "left",
) -> Grid:
    """Completes symmetry by copying one half across the symmetry axis."""
    arr = to_array(grid).copy()
    h, w = arr.shape

    if axis == "vertical":
        mid = w // 2
        for c in range(mid):
            opp = w - 1 - c
            if source == "left":
                arr[:, opp] = arr[:, c]
            else:
                arr[:, c] = arr[:, opp]
    elif axis == "horizontal":
        mid = h // 2
        for r in range(mid):
            opp = h - 1 - r
            if source == "top":
                arr[opp, :] = arr[r, :]
            else:
                arr[r, :] = arr[opp, :]
    return arr.tolist()


def tile_grid(
    grid: Grid | np.ndarray,
    repeats_h: int = 1,
    repeats_w: int = 1,
) -> Grid:
    """Repeats grid periodically along rows and columns."""
    arr = to_array(grid)
    return np.tile(arr, (repeats_h, repeats_w)).tolist()


def replace_color(
    grid: Grid | np.ndarray,
    old_c: int,
    new_c: int,
) -> Grid:
    """Replaces all occurrences of old_c with new_c."""
    arr = to_array(grid).copy()
    arr[arr == old_c] = new_c
    return arr.tolist()


def remap_colors(
    grid: Grid | np.ndarray,
    color_map: dict[int, int],
) -> Grid:
    """Remaps multiple colors simultaneously according to mapping dictionary."""
    arr = to_array(grid).copy()
    out = arr.copy()
    for old_c, new_c in color_map.items():
        out[arr == old_c] = new_c
    return out.tolist()


def color_counts(
    grid: Grid | np.ndarray,
    ignore_background: bool = False,
    background: int = 0,
) -> dict[int, int]:
    """Returns frequency counts of all colors present in the grid."""
    arr = to_array(grid)
    counts = dict(Counter(int(x) for x in arr.flatten()))
    if ignore_background and background in counts:
        del counts[background]
    return counts


def most_frequent_color(
    grid: Grid | np.ndarray,
    ignore_background: bool = True,
    background: int = 0,
) -> int | None:
    """Returns the most frequent color in the grid."""
    counts = color_counts(grid, ignore_background=ignore_background, background=background)
    if not counts:
        return None
    return max(counts.items(), key=lambda x: (x[1], -x[0]))[0]


def least_frequent_color(
    grid: Grid | np.ndarray,
    ignore_background: bool = True,
    background: int = 0,
) -> int | None:
    """Returns the least frequent color in the grid."""
    counts = color_counts(grid, ignore_background=ignore_background, background=background)
    if not counts:
        return None
    return min(counts.items(), key=lambda x: (x[1], x[0]))[0]
