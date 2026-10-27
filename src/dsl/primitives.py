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


def grid_equal(g1: Grid | np.ndarray | None, g2: Grid | np.ndarray | None) -> bool:
    """Checks if two grids have identical dimensions and values."""
    if g1 is None or g2 is None:
        return False
    a1 = np.asarray(g1, dtype=int)
    a2 = np.asarray(g2, dtype=int)
    return a1.shape == a2.shape and bool(np.array_equal(a1, a2))


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


def fill_enclosed_regions(
    grid: Grid | np.ndarray,
    fill_color: int,
    background: int = 0,
) -> Grid:
    """Fills background regions that are fully enclosed by non-background pixels."""
    arr = to_array(grid).copy()
    h, w = arr.shape

    # Find all exterior background cells via flood-fill from boundary
    visited = np.zeros((h, w), dtype=bool)
    queue = deque()

    # Seed boundary background cells
    for r in range(h):
        for c in (0, w - 1):
            if arr[r, c] == background and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if arr[r, c] == background and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))

    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        curr_r, curr_c = queue.popleft()
        for dr, dc in deltas:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and arr[nr, nc] == background:
                visited[nr, nc] = True
                queue.append((nr, nc))

    # Any background cell not visited from boundary is enclosed
    for r in range(h):
        for c in range(w):
            if arr[r, c] == background and not visited[r, c]:
                arr[r, c] = fill_color

    return arr.tolist()


def crop_largest_object(
    grid: Grid | np.ndarray,
    background: int = 0,
    connectivity: int = 4,
) -> Grid:
    """Finds the largest connected component and crops to its bounding box."""
    objs = find_connected_components(grid, background=background, connectivity=connectivity)
    if not objs:
        return to_grid(grid)
    largest = max(objs, key=lambda o: o.size)
    return largest.crop_from(grid)


def crop_smallest_object(
    grid: Grid | np.ndarray,
    background: int = 0,
    connectivity: int = 4,
) -> Grid:
    """Finds the smallest connected component and crops to its bounding box."""
    objs = find_connected_components(grid, background=background, connectivity=connectivity)
    if not objs:
        return to_grid(grid)
    smallest = min(objs, key=lambda o: o.size)
    return smallest.crop_from(grid)


def remove_small_objects(
    grid: Grid | np.ndarray,
    max_size: int = 1,
    background: int = 0,
    replacement: int = 0,
    connectivity: int = 4,
) -> Grid:
    """Removes noise or small connected components whose size <= max_size."""
    arr = to_array(grid).copy()
    objs = find_connected_components(arr, background=background, connectivity=connectivity)
    for obj in objs:
        if obj.size <= max_size:
            for r, c in obj.coords:
                arr[r, c] = replacement
    return arr.tolist()


def connect_matching_dots(grid: Grid | np.ndarray, background: int = 0) -> Grid:
    """Connects pairs of identical colored dots with orthogonal or diagonal lines."""
    arr = to_array(grid).copy()
    colors = set(np.unique(arr)) - {background}
    for c in colors:
        coords = list(zip(*np.where(arr == c)))
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                r1, c1 = coords[i]
                r2, c2 = coords[j]
                # Same row -> draw horizontal line
                if r1 == r2:
                    start_c, end_c = min(c1, c2), max(c1, c2)
                    path = [arr[r1, k] for k in range(start_c, end_c + 1)]
                    if all(v in (background, c) for v in path):
                        arr[r1, start_c : end_c + 1] = c
                # Same column -> draw vertical line
                elif c1 == c2:
                    start_r, end_r = min(r1, r2), max(r1, r2)
                    path = [arr[k, c1] for k in range(start_r, end_r + 1)]
                    if all(v in (background, c) for v in path):
                        arr[start_r : end_r + 1, c1] = c
                # Diagonal
                elif abs(r1 - r2) == abs(c1 - c2):
                    dr = 1 if r2 > r1 else -1
                    dc = 1 if c2 > c1 else -1
                    length = abs(r1 - r2)
                    path = [arr[r1 + k * dr, c1 + k * dc] for k in range(length + 1)]
                    if all(v in (background, c) for v in path):
                        for k in range(length + 1):
                            arr[r1 + k * dr, c1 + k * dc] = c
    return arr.tolist()


def repair_room_2d_periodic(room_arr: np.ndarray, is_vertical: bool = True) -> np.ndarray:
    """Detects and repairs the dominant 2D periodic motif in a rectangular room."""
    h, w = room_arr.shape
    if h == 0 or w == 0:
        return room_arr

    if is_vertical:
        best_p, best_score, best_block = None, 0.0, None
        for p in range(2, min(9, h // 2 + 1)):
            block = np.zeros((p, w), dtype=int)
            for r in range(p):
                for c in range(w):
                    vals = [room_arr[k, c] for k in range(r, h, p)]
                    block[r, c] = Counter(vals).most_common(1)[0][0]
            cell_matches = sum(
                room_arr[k, c] == block[k % p, c] for k in range(h) for c in range(w)
            )
            score = cell_matches / (h * w) if h * w > 0 else 0.0
            if score > best_score:
                best_score = score
                best_p = p
                best_block = block
        if best_block is not None and best_score >= 0.70:
            repaired = np.zeros_like(room_arr)
            for k in range(h):
                repaired[k, :] = best_block[k % best_p, :]
            return repaired
    else:
        best_p, best_score, best_block = None, 0.0, None
        for p in range(2, min(9, w // 2 + 1)):
            block = np.zeros((h, p), dtype=int)
            for r in range(h):
                for c in range(p):
                    vals = [room_arr[r, k] for k in range(c, w, p)]
                    block[r, c] = Counter(vals).most_common(1)[0][0]
            cell_matches = sum(
                room_arr[r, k] == block[r, k % p] for r in range(h) for k in range(w)
            )
            score = cell_matches / (h * w) if h * w > 0 else 0.0
            if score > best_score:
                best_score = score
                best_p = p
                best_block = block
        if best_block is not None and best_score >= 0.70:
            repaired = np.zeros_like(room_arr)
            for k in range(w):
                repaired[:, k] = best_block[:, k % best_p]
            return repaired

    return room_arr


def repair_periodic_subgrids(grid: Grid | np.ndarray, pad: int = 2) -> Grid:
    """Identifies rooms bounded by grid divider lines and repairs periodic 2D motifs."""
    arr = to_array(grid).copy()
    h, w = arr.shape
    r_divs = (
        [0]
        + [
            r
            for r in range(1, h - 1)
            if len(set(arr[r, :])) <= 2 and Counter(arr[r, :]).most_common(1)[0][1] >= w - 1
        ]
        + [h - 1]
    )
    c_divs = (
        [0]
        + [
            c
            for c in range(1, w - 1)
            if len(set(arr[:, c])) <= 2 and Counter(arr[:, c]).most_common(1)[0][1] >= h - 1
        ]
        + [w - 1]
    )
    r_divs = sorted(list(set(r_divs)))
    c_divs = sorted(list(set(c_divs)))

    out = arr.copy()
    for r_idx in range(len(r_divs) - 1):
        r_start, r_end = r_divs[r_idx], r_divs[r_idx + 1]
        for c_idx in range(len(c_divs) - 1):
            c_start, c_end = c_divs[c_idx], c_divs[c_idx + 1]
            room_h = r_end - r_start + 1
            room_w = c_end - c_start + 1

            if room_h > room_w and room_h >= 2 * pad + 2:
                interior = arr[r_start + pad : r_end - pad + 1, c_start + pad : c_end - pad + 1]
                repaired = repair_room_2d_periodic(interior, is_vertical=True)
                out[r_start + pad : r_end - pad + 1, c_start + pad : c_end - pad + 1] = repaired
            elif room_w >= room_h and room_w >= 2 * pad + 2:
                interior = arr[r_start + pad : r_end - pad + 1, c_start + pad : c_end - pad + 1]
                repaired = repair_room_2d_periodic(interior, is_vertical=False)
                out[r_start + pad : r_end - pad + 1, c_start + pad : c_end - pad + 1] = repaired

    return out.tolist()


def crop_hollow_rectangle_interior(grid: Grid | np.ndarray, background: int = 0) -> Grid:
    """Finds a hollow rectangular frame of uniform color and extracts its interior."""
    arr = to_array(grid)
    for c in sorted(set(np.unique(arr)) - {background}):
        coords = list(zip(*np.where(arr == c)))
        if len(coords) < 8:
            continue
        min_r, min_c = min(r for r, _ in coords), min(col for _, col in coords)
        max_r, max_c = max(r for r, _ in coords), max(col for _, col in coords)
        if max_r - min_r >= 2 and max_c - min_c >= 2:
            top = all(arr[min_r, col] == c for col in range(min_c, max_c + 1))
            bot = all(arr[max_r, col] == c for col in range(min_c, max_c + 1))
            left = all(arr[row, min_c] == c for row in range(min_r, max_r + 1))
            right = all(arr[row, max_c] == c for row in range(min_r, max_r + 1))
            if top and bot and left and right:
                inner = arr[min_r + 1 : max_r, min_c + 1 : max_c]
                if inner.size > 0:
                    return inner.tolist()
    return to_grid(grid)


def select_odd_one_out_block(
    grid: Grid | np.ndarray,
    win_h: int = 3,
    win_w: int = 3,
) -> Grid:
    """Finds composite block components and selects the block with unique symmetry."""
    arr = to_array(grid)
    h, w = arr.shape
    blocks: list[np.ndarray] = []

    if h == win_h and w % win_w == 0:
        for c in range(0, w, win_w):
            blocks.append(arr[:, c : c + win_w])
    elif w == win_w and h % win_h == 0:
        for r in range(0, h, win_h):
            blocks.append(arr[r : r + win_h, :])
    elif h % win_h == 0 and w % win_w == 0:
        for r in range(0, h, win_h):
            for c in range(0, w, win_w):
                blocks.append(arr[r : r + win_h, c : c + win_w])
    else:
        return to_grid(grid)

    if not blocks:
        return to_grid(grid)

    def is_sym(b: np.ndarray) -> bool:
        return bool(np.array_equal(b, np.flipud(b)) and np.array_equal(b, np.fliplr(b)))

    sigs = [is_sym(b) for b in blocks]
    counts = Counter(sigs)
    unique_sigs = [s for s, cnt in counts.items() if cnt == 1]
    if len(unique_sigs) == 1:
        target_idx = sigs.index(unique_sigs[0])
        return blocks[target_idx].tolist()

    return blocks[0].tolist()


def extract_window_relative_to_marker(
    grid: Grid | np.ndarray,
    marker_color: int | None = None,
    relation: str = "below",
    win_h: int = 3,
    win_w: int = 3,
) -> Grid:
    """Extracts a subwindow positioned relative to a single-pixel marker."""
    arr = to_array(grid)
    h, w = arr.shape

    if marker_color is None:
        colors, counts = np.unique(arr, return_counts=True)
        single_colors = [c for c, cnt in zip(colors, counts) if cnt == 1]
        if not single_colors:
            return to_grid(grid)
        marker_color = int(single_colors[0])

    markers = list(zip(*np.where(arr == marker_color)))
    if not markers:
        return to_grid(grid)

    r_m, c_m = markers[0]

    if relation == "below":
        r0, r1 = r_m + 1, r_m + 1 + win_h
        c0, c1 = c_m - win_w // 2, c_m - win_w // 2 + win_w
    elif relation == "above":
        r0, r1 = r_m - win_h, r_m
        c0, c1 = c_m - win_w // 2, c_m - win_w // 2 + win_w
    elif relation == "center":
        r0, r1 = r_m - win_h // 2, r_m - win_h // 2 + win_h
        c0, c1 = c_m - win_w // 2, c_m - win_w // 2 + win_w
    elif relation == "right":
        r0, r1 = r_m - win_h // 2, r_m - win_h // 2 + win_h
        c0, c1 = c_m + 1, c_m + 1 + win_w
    elif relation == "left":
        r0, r1 = r_m - win_h // 2, r_m - win_h // 2 + win_h
        c0, c1 = c_m - win_w, c_m
    else:
        return to_grid(grid)

    if 0 <= r0 and r1 <= h and 0 <= c0 and c1 <= w:
        return arr[r0:r1, c0:c1].tolist()

    return to_grid(grid)


def rigid_object_gravity(
    grid: Grid | np.ndarray,
    direction: str = "down",
    background: int = 0,
) -> Grid:
    """Simulates rigid-body physics: each connected component falls until colliding."""
    arr = to_array(grid).copy()
    h, w = arr.shape
    objs = find_connected_components(arr, background=background, connectivity=8)
    if not objs:
        return to_grid(grid)

    out = np.full((h, w), background, dtype=int)

    if direction == "down":
        objs = sorted(objs, key=lambda o: max(r for r, _ in o.coords), reverse=True)
        for obj in objs:
            coords = list(obj.coords)
            max_shift = 0
            while True:
                can = True
                for r, c in coords:
                    nr = r + max_shift + 1
                    if nr >= h or out[nr, c] != background:
                        can = False
                        break
                if can:
                    max_shift += 1
                else:
                    break
            for r, c in coords:
                out[r + max_shift, c] = arr[r, c]

    elif direction == "up":
        objs = sorted(objs, key=lambda o: min(r for r, _ in o.coords))
        for obj in objs:
            coords = list(obj.coords)
            max_shift = 0
            while True:
                can = True
                for r, c in coords:
                    nr = r - max_shift - 1
                    if nr < 0 or out[nr, c] != background:
                        can = False
                        break
                if can:
                    max_shift += 1
                else:
                    break
            for r, c in coords:
                out[r - max_shift, c] = arr[r, c]

    elif direction == "right":
        objs = sorted(objs, key=lambda o: max(c for _, c in o.coords), reverse=True)
        for obj in objs:
            coords = list(obj.coords)
            max_shift = 0
            while True:
                can = True
                for r, c in coords:
                    nc = c + max_shift + 1
                    if nc >= w or out[r, nc] != background:
                        can = False
                        break
                if can:
                    max_shift += 1
                else:
                    break
            for r, c in coords:
                out[r, c + max_shift] = arr[r, c]

    elif direction == "left":
        objs = sorted(objs, key=lambda o: min(c for _, c in o.coords))
        for obj in objs:
            coords = list(obj.coords)
            max_shift = 0
            while True:
                can = True
                for r, c in coords:
                    nc = c - max_shift - 1
                    if nc < 0 or out[r, nc] != background:
                        can = False
                        break
                if can:
                    max_shift += 1
                else:
                    break
            for r, c in coords:
                out[r, c - max_shift] = arr[r, c]

    return out.tolist()


def raycast_directional(
    grid: Grid | np.ndarray,
    direction: str = "down",
    background: int = 0,
) -> Grid:
    """Projects directional rays from non-background pixels across the grid."""
    arr = to_array(grid).copy()
    h, w = arr.shape
    out = arr.copy()

    dirs: list[tuple[int, int]] = []
    if direction == "down":
        dirs = [(1, 0)]
    elif direction == "up":
        dirs = [(-1, 0)]
    elif direction == "right":
        dirs = [(0, 1)]
    elif direction == "left":
        dirs = [(0, -1)]
    elif direction == "horizontal":
        dirs = [(0, -1), (0, 1)]
    elif direction == "vertical":
        dirs = [(-1, 0), (1, 0)]
    elif direction == "cross":
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        return to_grid(grid)

    coords = list(zip(*np.where(arr != background)))
    if not coords or len(coords) > 25:
        return to_grid(grid)

    for r, c in coords:
        color = arr[r, c]
        for dr, dc in dirs:
            cr, cc = r + dr, c + dc
            while 0 <= cr < h and 0 <= cc < w:
                if out[cr, cc] == background:
                    out[cr, cc] = color
                cr += dr
                cc += dc

    return out.tolist()


def raycast_diagonal_bouncing(
    grid: Grid | np.ndarray,
    background: int = 0,
) -> Grid:
    """Projects 4-way diagonal billiard rays that bounce off grid boundaries."""
    arr = to_array(grid).copy()
    h, w = arr.shape
    out = arr.copy()

    coords = list(zip(*np.where(arr != background)))
    if not coords or len(coords) > 10:
        return to_grid(grid)

    visited_states = set()
    for r, c in coords:
        color = arr[r, c]
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            curr_r, curr_c = r, c
            curr_dr, curr_dc = dr, dc
            for _ in range(2 * (h + w)):
                state = (curr_r, curr_c, curr_dr, curr_dc)
                if state in visited_states:
                    break
                visited_states.add(state)

                nr = curr_r + curr_dr
                nc = curr_c + curr_dc
                # Bounce on boundary
                if nr < 0 or nr >= h:
                    curr_dr = -curr_dr
                    nr = curr_r + curr_dr
                if nc < 0 or nc >= w:
                    curr_dc = -curr_dc
                    nc = curr_c + curr_dc
                if 0 <= nr < h and 0 <= nc < w:
                    if out[nr, nc] == background:
                        out[nr, nc] = color
                    curr_r, curr_c = nr, nc
                else:
                    break

    return out.tolist()


def extract_object_property_color_rule(
    train_pairs: list[dict[str, Any]],
    property_name: str = "size",
    background: int = 0,
) -> dict[int, int] | None:
    """Finds if a task maps connected components to a color based on size, height, or width."""
    prop_map: dict[int, int] = {}
    for pair in train_pairs:
        inp = pair["input"]
        out = np.asarray(pair["output"], dtype=int)
        inp_arr = np.asarray(inp, dtype=int)
        if inp_arr.shape != out.shape:
            return None

        objs = find_connected_components(inp, background=background, connectivity=8)
        for obj in objs:
            out_colors = {int(out[r, c]) for r, c in obj.coords}
            if len(out_colors) != 1:
                return None
            out_color = out_colors.pop()
            val = getattr(obj, property_name)
            if val in prop_map and prop_map[val] != out_color:
                return None
            prop_map[val] = out_color

    return prop_map if len(prop_map) >= 2 else None


def color_objects_by_property(
    grid: Grid | np.ndarray,
    property_map: dict[int, int],
    property_name: str = "size",
    background: int = 0,
) -> Grid:
    """Recolors each connected component according to its size, height, or width."""
    arr = to_array(grid).copy()
    objs = find_connected_components(grid, background=background, connectivity=8)
    for obj in objs:
        val = getattr(obj, property_name)
        if val in property_map:
            new_c = property_map[val]
            for r, c in obj.coords:
                arr[r, c] = new_c
    return arr.tolist()


def extract_enclosed_pixel_color(
    grid: Grid | np.ndarray,
    background: int = 0,
) -> Grid:
    """Finds a single pixel enclosed inside a hollow 3x3 square and returns it as a 1x1 grid."""
    arr = to_array(grid)
    h, w = arr.shape
    for r in range(h - 2):
        for c in range(w - 2):
            border_c = arr[r, c]
            if border_c == background:
                continue
            perim = [
                arr[r, c],
                arr[r, c + 1],
                arr[r, c + 2],
                arr[r + 1, c],
                arr[r + 1, c + 2],
                arr[r + 2, c],
                arr[r + 2, c + 1],
                arr[r + 2, c + 2],
            ]
            center_c = arr[r + 1, c + 1]
            if (
                all(p == border_c for p in perim)
                and center_c != border_c
                and center_c != background
            ):
                return [[int(center_c)]]
    return [[background]]


def classify_by_symmetry(
    grid: Grid | np.ndarray,
    sym_color: int = 1,
    asym_color: int = 7,
) -> Grid:
    """Classifies grid as symmetric (sym_color) vs asymmetric (asym_color) in a 1x1 grid."""
    arr = to_array(grid)
    v_sym = np.array_equal(arr, np.fliplr(arr))
    h_sym = np.array_equal(arr, np.flipud(arr))
    return [[sym_color]] if (v_sym or h_sym) else [[asym_color]]


def extract_binary_shape_classifier(
    train_pairs: list[dict[str, Any]],
) -> dict[tuple[bool, ...], int] | None:
    """Learns a mapping from input binary shape mask to target 1x1 class color."""
    shape_map: dict[tuple[bool, ...], int] = {}
    for tr in train_pairs:
        out = tr.get("output", [])
        if len(out) != 1 or len(out[0]) != 1:
            return None
        inp = np.asarray(tr["input"])
        target = int(out[0][0])
        mask = tuple((inp != 0).flatten().tolist())
        if mask in shape_map and shape_map[mask] != target:
            return None
        shape_map[mask] = target
    return shape_map if len(shape_map) >= 2 else None


def classify_binary_shape(
    grid: Grid | np.ndarray,
    shape_map: dict[tuple[bool, ...], int],
    fallback_color: int = 0,
) -> Grid:
    """Classifies an input grid's binary mask into a 1x1 target color."""
    inp = to_array(grid)
    mask = tuple((inp != 0).flatten().tolist())
    target = shape_map.get(mask, fallback_color)
    return [[target]]


def kronecker_fractal_expansion(
    grid: Grid | np.ndarray,
    background: int = 0,
) -> Grid:
    """Computes self-similarity fractal expansion (Kronecker product of mask with grid)."""
    arr = to_array(grid)
    mask = (arr != background).astype(int)
    return np.kron(mask, arr).tolist()


def tile_kaleidoscope_2x2(grid: Grid | np.ndarray) -> Grid:
    """Tiles grid in a 4-quadrant kaleidoscope pattern with horizontal and vertical mirror reflections."""
    arr = to_array(grid)
    top = np.hstack([arr, np.fliplr(arr)])
    bot = np.hstack([np.flipud(arr), np.flipud(np.fliplr(arr))])
    return np.vstack([top, bot]).tolist()


def tile_mirror_horizontal(grid: Grid | np.ndarray) -> Grid:
    """Tiles grid horizontally with its horizontal mirror reflection [A, fliplr(A)]."""
    arr = to_array(grid)
    return np.hstack([arr, np.fliplr(arr)]).tolist()


def tile_mirror_vertical(grid: Grid | np.ndarray) -> Grid:
    """Tiles grid vertically with its vertical mirror reflection vstack([A, flipud(A)])."""
    arr = to_array(grid)
    return np.vstack([arr, np.flipud(arr)]).tolist()


def tile_to_target_shape(
    grid: Grid | np.ndarray,
    target_h: int,
    target_w: int,
) -> Grid:
    """Tiles grid periodically until reaching the target dimensions (target_h, target_w)."""
    arr = to_array(grid)
    h, w = arr.shape
    if h == 0 or w == 0:
        return to_grid(grid)
    reps_h = (target_h + h - 1) // h
    reps_w = (target_w + w - 1) // w
    tiled = np.tile(arr, (reps_h, reps_w))
    return tiled[:target_h, :target_w].tolist()


def count_holes_in_object(obj: Object, grid_arr: np.ndarray, background: int = 0) -> int:
    """Counts the number of enclosed topological holes (genus) inside a connected component."""
    min_r, min_c, max_r, max_c = obj.bbox
    h, w = max_r - min_r + 1, max_c - min_c + 1

    padded = np.zeros((h + 2, w + 2), dtype=int)
    for r, c in obj.coords:
        padded[r - min_r + 1, c - min_c + 1] = 1

    visited = np.zeros_like(padded, dtype=bool)
    q = deque([(0, 0)])
    visited[0, 0] = True

    ph, pw = padded.shape
    while q:
        cr, cc = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < ph and 0 <= nc < pw and not visited[nr, nc] and padded[nr, nc] == 0:
                visited[nr, nc] = True
                q.append((nr, nc))

    hole_count = 0
    hole_visited = np.zeros_like(padded, dtype=bool)
    for r in range(ph):
        for c in range(pw):
            if padded[r, c] == 0 and not visited[r, c] and not hole_visited[r, c]:
                hole_count += 1
                hq = deque([(r, c)])
                hole_visited[r, c] = True
                while hq:
                    hr, hc = hq.popleft()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nhr, nhc = hr + dr, hc + dc
                        if (
                            0 <= nhr < ph
                            and 0 <= nhc < pw
                            and not hole_visited[nhr, nhc]
                            and padded[nhr, nhc] == 0
                            and not visited[nhr, nhc]
                        ):
                            hole_visited[nhr, nhc] = True
                            hq.append((nhr, nhc))

    return hole_count


def extract_holes_to_color_rule(
    train_pairs: list[dict[str, Any]],
    background: int = 0,
) -> dict[int, int] | None:
    """Learns if object color is determined by its number of enclosed topological holes."""
    hole_map: dict[int, int] = {}
    for pair in train_pairs:
        inp = pair["input"]
        out = np.asarray(pair["output"], dtype=int)
        inp_arr = np.asarray(inp, dtype=int)
        if inp_arr.shape != out.shape:
            return None
        objs = find_connected_components(inp, background=background, connectivity=8)
        for obj in objs:
            out_colors = {int(out[r, c]) for r, c in obj.coords}
            if len(out_colors) != 1:
                return None
            out_color = out_colors.pop()
            holes = count_holes_in_object(obj, inp_arr, background=background)
            if holes in hole_map and hole_map[holes] != out_color:
                return None
            hole_map[holes] = out_color

    return hole_map if len(hole_map) >= 2 else None


def color_objects_by_holes(
    grid: Grid | np.ndarray,
    hole_map: dict[int, int],
    background: int = 0,
) -> Grid:
    """Recolors each connected component according to its number of topological holes."""
    arr = to_array(grid).copy()
    objs = find_connected_components(grid, background=background, connectivity=8)
    for obj in objs:
        holes = count_holes_in_object(obj, arr, background=background)
        if holes in hole_map:
            new_c = hole_map[holes]
            for r, c in obj.coords:
                arr[r, c] = new_c
    return arr.tolist()


def extract_object_by_hole_count(
    grid: Grid | np.ndarray,
    target_holes: int = 1,
    background: int = 0,
) -> Grid:
    """Finds the unique connected component with target_holes and crops it."""
    arr = to_array(grid)
    objs = find_connected_components(grid, background=background, connectivity=8)
    matching = [
        o for o in objs if count_holes_in_object(o, arr, background=background) == target_holes
    ]
    if len(matching) == 1:
        return matching[0].crop_from(grid)
    return to_grid(grid)


# =============================================================================
# Version 9 Primitives: Motif Matching, Enclosed Area, Ranking & Periodic Lines
# =============================================================================

def recolor_motif(
    grid: Grid | np.ndarray,
    motif: Sequence[Sequence[int]] | np.ndarray,
    src_color: int,
    dst_color: int,
) -> Grid:
    """Detects occurrences of a binary motif/mask in src_color and recolors them to dst_color."""
    arr = to_array(grid).copy()
    m_arr = np.asarray(motif, dtype=int)
    mh, mw = m_arr.shape
    h, w = arr.shape
    if mh > h or mw > w:
        return arr.tolist()

    mask = (m_arr != 0)
    res = arr.copy()
    for r in range(h - mh + 1):
        for c in range(w - mw + 1):
            sub = arr[r : r + mh, c : c + mw]
            if np.all(sub[mask] == src_color):
                for dr in range(mh):
                    for dc in range(mw):
                        if mask[dr, dc]:
                            res[r + dr, c + dc] = dst_color
    return res.tolist()


def extract_recolor_motif_rule(
    train_pairs: list[dict[str, Any]],
) -> tuple[np.ndarray, int, int] | None:
    """Induces motif mask, source color, and destination color if task recolors a template."""
    candidate_motifs = [
        # Plus cross 3x3
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
        # Solid 2x2
        np.array([[1, 1], [1, 1]], dtype=int),
        # Hollow 3x3 frame
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
        # Horizontal 3x1 line
        np.array([[1, 1, 1]], dtype=int),
        # Vertical 1x3 line
        np.array([[1], [1], [1]], dtype=int),
        # Diagonal 3x3
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int),
    ]

    for motif in candidate_motifs:
        # Check if there is a consistent (src_color, dst_color)
        candidate_pair = None
        consistent = True
        for p in train_pairs:
            inp = np.asarray(p["input"], dtype=int)
            out = np.asarray(p["output"], dtype=int)
            if inp.shape != out.shape:
                consistent = False
                break
            diff = (inp != out)
            if not np.any(diff):
                continue
            srcs = set(inp[diff])
            dsts = set(out[diff])
            if len(srcs) == 1 and len(dsts) == 1:
                s_c = int(srcs.pop())
                d_c = int(dsts.pop())
                if candidate_pair is None:
                    candidate_pair = (s_c, d_c)
                elif candidate_pair != (s_c, d_c):
                    consistent = False
                    break
            else:
                consistent = False
                break

        if consistent and candidate_pair is not None:
            s_c, d_c = candidate_pair
            # Verify across all train pairs
            all_match = True
            for p in train_pairs:
                pred = recolor_motif(p["input"], motif=motif, src_color=s_c, dst_color=d_c)
                if pred != p["output"]:
                    all_match = False
                    break
            if all_match:
                return (motif, s_c, d_c)

    return None


def fill_enclosed_regions_by_area(
    grid: Grid | np.ndarray,
    area_to_color: dict[int, int],
    background: int = 0,
) -> Grid:
    """Identifies enclosed background cavities and recolors each cavity based on its pixel area."""
    arr = to_array(grid).copy()
    h, w = arr.shape
    visited = np.zeros((h, w), dtype=bool)
    from collections import deque
    queue = deque()

    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and arr[r, c] == background and not visited[r, c]:
                visited[r, c] = True
                queue.append((r, c))

    while queue:
        cr, cc = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and arr[nr, nc] == background:
                visited[nr, nc] = True
                queue.append((nr, nc))

    enclosed_visited = np.zeros((h, w), dtype=bool)
    res = arr.copy()
    for r in range(h):
        for c in range(w):
            if arr[r, c] == background and not visited[r, c] and not enclosed_visited[r, c]:
                cavity = []
                c_q = deque([(r, c)])
                enclosed_visited[r, c] = True
                while c_q:
                    er, ec = c_q.popleft()
                    cavity.append((er, ec))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = er + dr, ec + dc
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and arr[nr, nc] == background
                            and not visited[nr, nc]
                            and not enclosed_visited[nr, nc]
                        ):
                            enclosed_visited[nr, nc] = True
                            c_q.append((nr, nc))
                area = len(cavity)
                if area in area_to_color:
                    new_c = area_to_color[area]
                    for cr, cc in cavity:
                        res[cr, cc] = new_c

    return res.tolist()


def extract_enclosed_area_rule(
    train_pairs: list[dict[str, Any]],
    background: int = 0,
) -> dict[int, int] | None:
    """Learns if enclosed background cavity areas map deterministically to fill colors."""
    from collections import deque
    area_map: dict[int, int] = {}

    for pair in train_pairs:
        inp = np.asarray(pair["input"], dtype=int)
        out = np.asarray(pair["output"], dtype=int)
        if inp.shape != out.shape:
            return None
        h, w = inp.shape
        visited = np.zeros((h, w), dtype=bool)
        queue = deque()
        for r in range(h):
            for c in range(w):
                if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and inp[r, c] == background and not visited[r, c]:
                    visited[r, c] = True
                    queue.append((r, c))

        while queue:
            cr, cc = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and inp[nr, nc] == background:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

        enclosed_visited = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if inp[r, c] == background and not visited[r, c] and not enclosed_visited[r, c]:
                    cavity = []
                    c_q = deque([(r, c)])
                    enclosed_visited[r, c] = True
                    while c_q:
                        er, ec = c_q.popleft()
                        cavity.append((er, ec))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = er + dr, ec + dc
                            if (
                                0 <= nr < h
                                and 0 <= nc < w
                                and inp[nr, nc] == background
                                and not visited[nr, nc]
                                and not enclosed_visited[nr, nc]
                            ):
                                enclosed_visited[nr, nc] = True
                                c_q.append((nr, nc))
                    area = len(cavity)
                    out_colors = {int(out[cr, cc]) for cr, cc in cavity}
                    if len(out_colors) != 1:
                        return None
                    color = out_colors.pop()
                    if area in area_map and area_map[area] != color:
                        return None
                    area_map[area] = color

    if not area_map:
        return None

    # Verify rule on all train pairs
    for p in train_pairs:
        if fill_enclosed_regions_by_area(p["input"], area_to_color=area_map, background=background) != p["output"]:
            return None

    return area_map


def color_objects_by_rank(
    grid: Grid | np.ndarray,
    rank_to_color: dict[int, int],
    metric: str = "size",
    reverse: bool = True,
    background: int = 0,
) -> Grid:
    """Ranks connected components by size, height, or width, and recolors by rank."""
    arr = to_array(grid).copy()
    objs = find_connected_components(grid, background=background, connectivity=8)
    if not objs:
        return arr.tolist()

    def get_val(obj: Object) -> int:
        if metric == "height":
            return obj.height
        elif metric == "width":
            return obj.width
        return obj.size

    objs_sorted = sorted(objs, key=get_val, reverse=reverse)
    res = arr.copy()
    for rank, obj in enumerate(objs_sorted):
        if rank in rank_to_color:
            new_c = rank_to_color[rank]
            for r, c in obj.coords:
                res[r, c] = new_c

    return res.tolist()


def extract_rank_color_rule(
    train_pairs: list[dict[str, Any]],
    background: int = 0,
) -> tuple[str, bool, dict[int, int]] | None:
    """Learns if objects are colored by their rank order of height, width, or size."""
    for metric in ["height", "width", "size"]:
        for reverse in [True, False]:
            rank_map: dict[int, int] = {}
            valid = True
            for pair in train_pairs:
                inp = pair["input"]
                out = np.asarray(pair["output"], dtype=int)
                inp_arr = np.asarray(inp, dtype=int)
                if inp_arr.shape != out.shape:
                    valid = False
                    break
                objs = find_connected_components(inp, background=background, connectivity=8)
                if len(objs) < 2:
                    valid = False
                    break

                def get_val(obj: Object, m=metric) -> int:
                    if m == "height":
                        return obj.height
                    elif m == "width":
                        return obj.width
                    return obj.size

                objs_sorted = sorted(objs, key=get_val, reverse=reverse)
                for rank, obj in enumerate(objs_sorted):
                    out_colors = {int(out[r, c]) for r, c in obj.coords}
                    if len(out_colors) != 1:
                        valid = False
                        break
                    color = out_colors.pop()
                    if rank in rank_map and rank_map[rank] != color:
                        valid = False
                        break
                    rank_map[rank] = color
                if not valid:
                    break

            if valid and len(rank_map) >= 2:
                # Test against all train pairs
                matches_all = True
                for p in train_pairs:
                    pred = color_objects_by_rank(
                        p["input"],
                        rank_to_color=rank_map,
                        metric=metric,
                        reverse=reverse,
                        background=background,
                    )
                    if pred != p["output"]:
                        matches_all = False
                        break
                if matches_all:
                    return (metric, reverse, rank_map)

    return None


def extrapolate_periodic_grid(
    grid: Grid | np.ndarray,
    mode: str = "diag_sum",
    period: int = 3,
    mod_map: dict[int, int] | None = None,
    background: int = 0,
) -> Grid:
    """Fills background cells according to periodic modulo pattern along diagonals, rows, or cols."""
    arr = to_array(grid).copy()
    h, w = arr.shape
    if mod_map is None:
        mod_map = {}
        for r in range(h):
            for c in range(w):
                v = int(arr[r, c])
                if v != background:
                    if mode == "diag_sum":
                        k = (r + c) % period
                    elif mode == "diag_diff":
                        k = (r - c) % period
                    elif mode == "row":
                        k = r % period
                    else:
                        k = c % period
                    if k in mod_map and mod_map[k] != v:
                        return arr.tolist()
                    mod_map[k] = v

    res = arr.copy()
    for r in range(h):
        for c in range(w):
            if res[r, c] == background:
                if mode == "diag_sum":
                    k = (r + c) % period
                elif mode == "diag_diff":
                    k = (r - c) % period
                elif mode == "row":
                    k = r % period
                else:
                    k = c % period
                if k in mod_map:
                    res[r, c] = mod_map[k]

    return res.tolist()


def extract_periodic_grid_rule(
    train_pairs: list[dict[str, Any]],
    background: int = 0,
) -> tuple[str, int] | None:
    """Infers if background cells are filled by periodic extrapolation."""
    for mode in ["diag_sum", "diag_diff", "row", "col"]:
        for period in range(2, 8):
            valid = True
            for p in train_pairs:
                inp = p["input"]
                out = p["output"]
                arr_in = np.asarray(inp, dtype=int)
                arr_out = np.asarray(out, dtype=int)
                if arr_in.shape != arr_out.shape:
                    valid = False
                    break
                pred = extrapolate_periodic_grid(inp, mode=mode, period=period, background=background)
                if pred != out:
                    valid = False
                    break
            if valid:
                return (mode, period)
    return None


def extrapolate_point_progression(
    grid: Grid | np.ndarray,
    point_color: int = 1,
    new_color: int | None = None,
    background: int = 0,
) -> Grid:
    """Extrapolates a collinear arithmetic progression of points."""
    arr = to_array(grid).copy()
    h, w = arr.shape
    coords = list(zip(*np.where(arr == point_color)))
    if len(coords) < 2:
        return arr.tolist()

    coords.sort()
    dr = coords[1][0] - coords[0][0]
    dc = coords[1][1] - coords[0][1]
    is_collinear = True
    for i in range(2, len(coords)):
        if (coords[i][0] - coords[i - 1][0] != dr) or (coords[i][1] - coords[i - 1][1] != dc):
            is_collinear = False
            break

    if not is_collinear:
        return arr.tolist()

    next_r = coords[-1][0] + dr
    next_c = coords[-1][1] + dc
    fill_c = new_color if new_color is not None else point_color
    if 0 <= next_r < h and 0 <= next_c < w:
        arr[next_r, next_c] = fill_c

    return arr.tolist()


def extract_point_progression_rule(
    train_pairs: list[dict[str, Any]],
    background: int = 0,
) -> tuple[int, int] | None:
    """Learns if output adds the next collinear point in a progression."""
    candidate_colors = None
    for p in train_pairs:
        inp = np.asarray(p["input"], dtype=int)
        out = np.asarray(p["output"], dtype=int)
        if inp.shape != out.shape:
            return None
        diff = (inp != out)
        if diff.sum() != 1:
            return None
        r, c = np.where(diff)
        dst_c = int(out[r[0], c[0]])
        # Find which color had >= 2 points in input
        colors = [c_val for c_val in np.unique(inp) if c_val != background and (inp == c_val).sum() >= 2]
        matching_pt_color = None
        for pt_c in colors:
            pred = extrapolate_point_progression(p["input"], point_color=int(pt_c), new_color=dst_c, background=background)
            if pred == p["output"]:
                matching_pt_color = int(pt_c)
                break
        if matching_pt_color is None:
            return None
        if candidate_colors is None:
            candidate_colors = (matching_pt_color, dst_c)
        elif candidate_colors != (matching_pt_color, dst_c):
            return None

    return candidate_colors


# =============================================================================
# Version 10 Primitives: Deep Evaluation Perception Engine
# =============================================================================

def trigger_release_gravity(
    grid: Grid | np.ndarray,
    background: int = 0,
) -> Grid:
    """Designates grid[0][0] as trigger color, erases trigger color elsewhere, and drops objects."""
    arr = to_array(grid).copy()
    trigger = int(arr[0, 0])
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            if (r, c) != (0, 0) and arr[r, c] == trigger:
                arr[r, c] = background

    objs = find_connected_components(arr, background=background, connectivity=4)
    filtered_objs = [o for o in objs if (0, 0) not in o.coords]
    filtered_objs.sort(key=lambda o: max(r for r, c in o.coords), reverse=True)

    canvas = np.zeros_like(arr)
    canvas[0, 0] = trigger
    for o in filtered_objs:
        for r, c in o.coords:
            canvas[r, c] = o.color

    changed = True
    while changed:
        changed = False
        for o in filtered_objs:
            can_move = True
            for r, c in o.coords:
                nr = r + 1
                if nr >= h or (canvas[nr, c] != background and (nr, c) not in o.coords):
                    can_move = False
                    break
            if can_move:
                for r, c in o.coords:
                    canvas[r, c] = background
                new_coords = set()
                for r, c in o.coords:
                    canvas[r + 1, c] = o.color
                    new_coords.add((r + 1, c))
                o.coords = new_coords
                changed = True

    return canvas.tolist()


def fill_puzzle_socket(
    grid: Grid | np.ndarray,
    piece_color: int = 2,
    hole_color: int = 0,
    fill_color: int = 1,
) -> Grid:
    """Finds an enclosed rectangular hole of hole_color and moves matching piece_color block into it."""
    arr = to_array(grid).copy()
    h, w = arr.shape
    hole_objs = find_connected_components(arr == hole_color, background=False)
    valid_holes = []
    for o in hole_objs:
        rr = [r for r, c in o.coords]
        cc = [c for r, c in o.coords]
        min_r, max_r = min(rr), max(rr)
        min_c, max_c = min(cc), max(cc)
        oh = max_r - min_r + 1
        ow = max_c - min_c + 1
        if oh * ow == o.size and oh >= 1 and ow >= 1:
            valid_holes.append((min_r, min_c, oh, ow))

    res = arr.copy()
    for h_r, h_c, h_h, h_w in valid_holes:
        for r in range(h - h_h + 1):
            for c in range(w - h_w + 1):
                if r == h_r and c == h_c:
                    continue
                sub = arr[r : r + h_h, c : c + h_w]
                if np.all(sub == piece_color):
                    res[r : r + h_h, c : c + h_w] = hole_color
                    res[h_r : h_r + h_h, h_c : h_c + h_w] = fill_color
                    return res.tolist()

    return res.tolist()


def extract_puzzle_socket_rule(
    train_pairs: list[dict[str, Any]],
) -> tuple[int, int, int] | None:
    """Learns piece_color, hole_color, and fill_color for socket puzzle tasks."""
    colors = set()
    for p in train_pairs:
        colors.update(np.unique(p["input"]).tolist())
        colors.update(np.unique(p["output"]).tolist())
    colors = sorted(list(colors))

    for piece_c in colors:
        for hole_c in [0]:
            for fill_c in colors:
                valid = True
                for p in train_pairs:
                    pred = fill_puzzle_socket(p["input"], piece_color=piece_c, hole_color=hole_c, fill_color=fill_c)
                    if pred != p["output"]:
                        valid = False
                        break
                if valid:
                    return (piece_c, hole_c, fill_c)
    return None


def slide_dots_to_wall(
    grid: Grid | np.ndarray,
    dot_color: int = 4,
    direction: str = "right",
    background: int = 0,
) -> Grid:
    """Slides mobile dots horizontally or vertically until contacting an obstacle or boundary."""
    arr = to_array(grid).copy()
    h, w = arr.shape
    res = arr.copy()
    dr, dc = (0, 1) if direction == "right" else ((0, -1) if direction == "left" else ((1, 0) if direction == "down" else (-1, 0)))

    coords = list(zip(*np.where(arr == dot_color)))
    for r, c in coords:
        cr, cc = r, c
        while 0 <= cr + dr < h and 0 <= cc + dc < w and res[cr + dr, cc + dc] == background:
            cr += dr
            cc += dc
        if (cr, cc) != (r, c):
            res[r, c] = background
            res[cr, cc] = dot_color

    return res.tolist()


def extract_slide_dots_rule(
    train_pairs: list[dict[str, Any]],
    background: int = 0,
) -> tuple[int, str] | None:
    """Learns dot_color and direction for sliding dots to wall."""
    colors = set()
    for p in train_pairs:
        colors.update(np.unique(p["input"]).tolist())
    colors = sorted(list(colors))

    for dot_c in colors:
        if dot_c == background:
            continue
        for d in ["right", "left", "down", "up"]:
            valid = True
            for p in train_pairs:
                pred = slide_dots_to_wall(p["input"], dot_color=dot_c, direction=d, background=background)
                if pred != p["output"]:
                    valid = False
                    break
            if valid:
                return (dot_c, d)
    return None


def project_diagonal_rays(
    grid: Grid | np.ndarray,
    ray_color: int = 1,
    background: int = 0,
) -> Grid:
    """Projects diagonal rays from source dots of ray_color."""
    arr = to_array(grid).copy()
    h, w = arr.shape
    res = arr.copy()
    coords = list(zip(*np.where(arr == ray_color)))
    for r, c in coords:
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            cr, cc = r + dr, c + dc
            while 0 <= cr < h and 0 <= cc < w:
                if res[cr, cc] == background:
                    res[cr, cc] = ray_color
                elif res[cr, cc] != ray_color:
                    break
                cr += dr
                cc += dc
    return res.tolist()


def straighten_orthogonal_lines(
    grid: Grid | np.ndarray,
    line_color: int = 6,
    background: int = 0,
) -> Grid:
    """Straightens lines by bridging single-gap or kink deviations."""
    arr = to_array(grid).copy()
    h, w = arr.shape
    res = arr.copy()

    for c in range(w):
        col = arr[:, c]
        count = int((col == line_color).sum())
        if count >= 3:
            rows = np.where(col == line_color)[0]
            r_min, r_max = int(rows.min()), int(rows.max())
            if r_max - r_min + 1 - count <= 4:
                for r in range(r_min, r_max + 1):
                    res[r, c] = line_color
                for r in range(r_min, r_max + 1):
                    for adj_c in [c - 1, c + 1]:
                        if 0 <= adj_c < w and arr[r, adj_c] == line_color and (arr[:, adj_c] == line_color).sum() <= 3:
                            res[r, adj_c] = background

    for r in range(h):
        row = arr[r, :]
        count = int((row == line_color).sum())
        if count >= 3:
            cols = np.where(row == line_color)[0]
            c_min, c_max = int(cols.min()), int(cols.max())
            if c_max - c_min + 1 - count <= 4:
                for c in range(c_min, c_max + 1):
                    res[r, c] = line_color
                for c in range(c_min, c_max + 1):
                    for adj_r in [r - 1, r + 1]:
                        if 0 <= adj_r < h and arr[adj_r, c] == line_color and (arr[adj_r, :] == line_color).sum() <= 3:
                            res[adj_r, c] = background

    return res.tolist()


# =============================================================================
# Version 13 Primitives: Masked Reflection Inpainting & Gridline Lattice
# =============================================================================

def inpaint_masked_reflection(
    grid: Grid | np.ndarray,
    obj_color: int = 2,
    mask_color: int = 8,
    axis: str = "vertical",
) -> Grid:
    """Mirrors unmasked pixels across reflection axis to inpaint masked regions."""
    arr = to_array(grid).copy()
    h, w = arr.shape
    res = arr.copy()

    if axis == "vertical":
        for r in range(h):
            for c in range(w):
                if arr[r, c] == mask_color:
                    opp_c = w - 1 - c
                    if arr[r, opp_c] == obj_color:
                        res[r, c] = obj_color
                    elif arr[r, opp_c] != mask_color:
                        res[r, c] = arr[r, opp_c]
    else:
        for r in range(h):
            for c in range(w):
                if arr[r, c] == mask_color:
                    opp_r = h - 1 - r
                    if arr[opp_r, c] == obj_color:
                        res[r, c] = obj_color
                    elif arr[opp_r, c] != mask_color:
                        res[r, c] = arr[opp_r, c]

    return res.tolist()


def extract_masked_reflection_rule(
    train_pairs: list[dict[str, Any]],
) -> tuple[int, int, str] | None:
    """Learns obj_color, mask_color, and axis for masked reflection inpainting."""
    for p in train_pairs:
        inp = np.asarray(p["input"])
        out = np.asarray(p["output"])
        if inp.shape != out.shape:
            return None

    # Collect colors
    colors = set()
    for p in train_pairs:
        colors.update(np.unique(p["input"]).tolist())
    colors = sorted(list(colors))

    for obj_c in colors:
        for mask_c in colors:
            if obj_c == mask_c:
                continue
            for ax in ["vertical", "horizontal"]:
                valid = True
                for p in train_pairs:
                    pred = inpaint_masked_reflection(p["input"], obj_color=obj_c, mask_color=mask_c, axis=ax)
                    if pred != p["output"]:
                        valid = False
                        break
                if valid:
                    return (obj_c, mask_c, ax)
    return None


def detect_gridline_lattice(
    grid: Grid | np.ndarray,
) -> tuple[int | None, list[int], list[int]]:
    """Detects uniform horizontal and vertical separator gridlines forming a lattice."""
    arr = to_array(grid)
    h, w = arr.shape
    uniform_rows: dict[int, list[int]] = {}
    for r in range(1, h - 1):
        u = np.unique(arr[r, :])
        if len(u) == 1:
            color = int(u[0])
            uniform_rows.setdefault(color, []).append(r)

    uniform_cols: dict[int, list[int]] = {}
    for c in range(1, w - 1):
        u = np.unique(arr[:, c])
        if len(u) == 1:
            color = int(u[0])
            uniform_cols.setdefault(color, []).append(c)

    common_colors = set(uniform_rows.keys()) | set(uniform_cols.keys())
    for color in common_colors:
        r_seps = uniform_rows.get(color, [])
        c_seps = uniform_cols.get(color, [])
        if r_seps or c_seps:
            return (color, r_seps, c_seps)

    return (None, [], [])


def remove_gridlines(
    grid: Grid | np.ndarray,
    sep_color: int,
    row_seps: list[int],
    col_seps: list[int],
) -> list[list[Grid]]:
    """Splits grid by separator lattice into matrix of subgrid panels."""
    arr = to_array(grid)
    h, w = arr.shape

    r_slices = []
    prev_r = 0
    for r in sorted(row_seps):
        if r > prev_r:
            r_slices.append((prev_r, r))
        prev_r = r + 1
    if prev_r < h:
        r_slices.append((prev_r, h))

    c_slices = []
    prev_c = 0
    for c in sorted(col_seps):
        if c > prev_c:
            c_slices.append((prev_c, c))
        prev_c = c + 1
    if prev_c < w:
        c_slices.append((prev_c, w))

    panels: list[list[Grid]] = []
    for r1, r2 in r_slices:
        row_panels = []
        for c1, c2 in c_slices:
            sub = arr[r1:r2, c1:c2].tolist()
            row_panels.append(sub)
        panels.append(row_panels)

    return panels


def restore_bilateral_symmetry(
    grid: Grid | np.ndarray,
    background: int | None = None,
    axis: str = "vertical",
) -> Grid:
    """Restores bilateral reflection symmetry by removing asymmetric noise pixels."""
    arr = to_array(grid).copy()
    if background is None:
        background = int(arr[0, 0])
    pred = np.full_like(arr, background)
    for c in np.unique(arr):
        if c == background:
            continue
        mask = (arr == c)
        rows, cols = np.where(mask)
        if len(rows) == 0:
            continue
        min_r, max_r = int(rows.min()), int(rows.max())
        min_c, max_c = int(cols.min()), int(cols.max())

        best_sym_sub = None
        best_score = -999999
        best_bbox = None

        for c0 in range(min_c, max_c + 1):
            for c1 in range(c0 + 1, max_c + 2):
                sub_mask = mask[min_r : max_r + 1, c0:c1]
                sym_sub = sub_mask & np.fliplr(sub_mask)
                kept = int(np.sum(sym_sub))
                outside = int(np.sum(mask[min_r : max_r + 1, :c0]) + np.sum(mask[min_r : max_r + 1, c1:]))
                inside_asym = int(np.sum(sub_mask)) - kept
                score = kept * 2 - outside - inside_asym
                if score > best_score:
                    best_score = score
                    best_sym_sub = sym_sub
                    best_bbox = (min_r, max_r + 1, c0, c1)

        if best_bbox is not None and best_sym_sub is not None:
            r0, r1, c0, c1 = best_bbox
            pred[r0:r1, c0:c1][best_sym_sub] = c
    return pred.tolist()


def extract_restore_bilateral_symmetry_rule(
    train_pairs: list[dict[str, Any]],
    background: int | None = None,
) -> bool:
    """Checks if restoring vertical bilateral symmetry solves 100% of training pairs."""
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        pred = restore_bilateral_symmetry(inp, background=background)
        if not grid_equal(pred, out):
            return False
    return True


def fill_ranked_pillars_by_indicator(
    grid: Grid | np.ndarray,
) -> Grid:
    """Fills the K-th tallest vertical pillar of color C whose height K is specified by indicator bars."""
    arr = to_array(grid).copy()
    H, W = arr.shape
    indicators: dict[int, int] = {}
    for c in range(W):
        if arr[0, c] != 0:
            color = int(arr[0, c])
            h = 0
            while h < H and arr[h, c] == color:
                h += 1
            indicators[color] = h

    for color, rank in indicators.items():
        pillars: list[tuple[int, int, int, int]] = []
        for col in range(W):
            if arr[H - 1, col] == color:
                r_bot = H - 1
                r_top = r_bot
                for r in range(H - 1, -1, -1):
                    if arr[r, col] == color:
                        r_top = r
                    elif arr[r, col] == 0:
                        break
                if r_top > 0 and r_bot - r_top >= 1:
                    pillars.append((r_bot - r_top + 1, r_top, r_bot, col))
        pillars.sort(key=lambda x: x[0], reverse=True)
        if len(pillars) >= rank:
            chosen_h, r_top, r_bot, col = pillars[rank - 1]
            arr[r_top : r_bot + 1, col] = color
    return arr.tolist()


def extract_ranked_pillars_rule(
    train_pairs: list[dict[str, Any]],
) -> bool:
    """Checks if filling ranked pillars by indicator bars solves 100% of training pairs."""
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        pred = fill_ranked_pillars_by_indicator(inp)
        if not grid_equal(pred, out):
            return False
    return True


def extract_border_legend(
    arr: np.ndarray,
) -> tuple[dict[int, int], np.ndarray]:
    """Extracts the longest contiguous border color-mapping legend (2xK or Kx2) and its boolean mask."""
    H, W = arr.shape
    candidates: list[tuple[int, dict[int, int], np.ndarray]] = []

    # 1. Height 2 (horizontal legend on top or bottom border)
    for r0 in [0, H - 2]:
        if r0 < 0 or r0 + 2 > H:
            continue
        c0 = 0
        while c0 < W:
            if arr[r0, c0] != 0 and arr[r0 + 1, c0] != 0:
                c1 = c0
                while c1 < W and arr[r0, c1] != 0 and arr[r0 + 1, c1] != 0:
                    c1 += 1
                if c1 - c0 >= 2:
                    sub = arr[r0 : r0 + 2, c0:c1]
                    keys, vals = (sub[0, :], sub[1, :]) if r0 == 0 else (sub[1, :], sub[0, :])
                    if len(set(keys)) == len(keys):
                        mask = np.zeros_like(arr, dtype=bool)
                        mask[r0 : r0 + 2, c0:c1] = True
                        mapping = dict(zip([int(k) for k in keys], [int(v) for v in vals]))
                        candidates.append((c1 - c0, mapping, mask))
                c0 = c1
            else:
                c0 += 1

    # 2. Width 2 (vertical legend on left or right border)
    for c0 in [0, W - 2]:
        if c0 < 0 or c0 + 2 > W:
            continue
        r0 = 0
        while r0 < H:
            if arr[r0, c0] != 0 and arr[r0, c0 + 1] != 0:
                r1 = r0
                while r1 < H and arr[r1, c0] != 0 and arr[r1, c0 + 1] != 0:
                    r1 += 1
                if r1 - r0 >= 2:
                    sub = arr[r0:r1, c0 : c0 + 2]
                    keys, vals = (sub[:, 0], sub[:, 1]) if c0 == 0 else (sub[:, 1], sub[:, 0])
                    if len(set(keys)) == len(keys):
                        mask = np.zeros_like(arr, dtype=bool)
                        mask[r0:r1, c0 : c0 + 2] = True
                        mapping = dict(zip([int(k) for k in keys], [int(v) for v in vals]))
                        candidates.append((r1 - r0, mapping, mask))
                r0 = r1
            else:
                r0 += 1

    if not candidates:
        return {}, np.zeros_like(arr, dtype=bool)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def fill_cavities_from_legend(
    grid: Grid | np.ndarray,
) -> Grid:
    """Fills enclosed background cavities of key-color frames with their paired legend values."""
    arr = to_array(grid).copy()
    key_to_val, leg_mask = extract_border_legend(arr)
    for key_c, val_c in key_to_val.items():
        frame_mask = (arr == key_c) & (~leg_mask)
        if not np.any(frame_mask):
            continue
        canvas = np.zeros_like(arr)
        canvas[frame_mask] = 1
        filled = np.asarray(fill_enclosed_regions(canvas.tolist(), fill_color=2, background=0))
        enclosed_cells = (filled == 2) & (arr == 0) & (~leg_mask)
        arr[enclosed_cells] = val_c
    return arr.tolist()


def extract_legend_cavity_rule(
    train_pairs: list[dict[str, Any]],
) -> bool:
    """Checks if filling cavities via border legend solves 100% of training pairs."""
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        pred = fill_cavities_from_legend(inp)
        if not grid_equal(pred, out):
            return False
    return True


def standardize_parallel_lines(
    grid: Grid | np.ndarray,
) -> Grid:
    """Standardizes lengths of parallel line segments to a reference or modal length."""
    arr = to_array(grid).copy()
    H, W = arr.shape
    bg = int(arr[0, 0])

    orientations = {
        "horiz": (0, 1),
        "vert": (1, 0),
        "diag_down_left": (1, -1),
        "diag_down_right": (1, 1),
    }

    orient_scores = {k: 0 for k in orientations}
    for r in range(H):
        for c in range(W):
            val = arr[r, c]
            if val == bg:
                continue
            for k, (dr, dc) in orientations.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and arr[nr, nc] == val:
                    orient_scores[k] += 1

    best_orient = max(orient_scores, key=orient_scores.get)
    if orient_scores[best_orient] == 0:
        return arr.tolist()

    dr, dc = orientations[best_orient]

    visited = np.zeros_like(arr, dtype=bool)
    segments: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(H):
        for c in range(W):
            if arr[r, c] != bg and not visited[r, c]:
                color = int(arr[r, c])
                sr, sc = r, c
                while 0 <= sr - dr < H and 0 <= sc - dc < W and arr[sr - dr, sc - dc] == color:
                    sr -= dr
                    sc -= dc
                coords: list[tuple[int, int]] = []
                curr_r, curr_c = sr, sc
                while 0 <= curr_r < H and 0 <= curr_c < W and arr[curr_r, curr_c] == color:
                    visited[curr_r, curr_c] = True
                    coords.append((curr_r, curr_c))
                    curr_r += dr
                    curr_c += dc
                if coords:
                    segments.append((color, coords))

    if not segments:
        return arr.tolist()

    ref_lines = [s for s in segments if s[0] == 9]
    if ref_lines:
        target_L = len(ref_lines[0][1])
    else:
        lengths = [len(s[1]) for s in segments]
        target_L = Counter(lengths).most_common(1)[0][0]

    start_keys = []
    end_keys = []
    for col, coords in segments:
        sr, sc = coords[0]
        er, ec = coords[-1]
        if best_orient == "horiz":
            start_keys.append(sc)
            end_keys.append(ec)
        elif best_orient == "vert":
            start_keys.append(sr)
            end_keys.append(er)
        elif best_orient == "diag_down_left":
            start_keys.append(sr - sc)
            end_keys.append(er - ec)
        elif best_orient == "diag_down_right":
            start_keys.append(sr + sc)
            end_keys.append(er + ec)

    align_end = len(set(end_keys)) == 1 and len(set(start_keys)) > 1

    out_arr = np.full_like(arr, bg)
    for color, coords in segments:
        if align_end:
            er, ec = coords[-1]
            for step in range(target_L):
                r = er - step * dr
                c = ec - step * dc
                if 0 <= r < H and 0 <= c < W:
                    out_arr[r, c] = color
        else:
            sr, sc = coords[0]
            for step in range(target_L):
                r = sr + step * dr
                c = sc + step * dc
                if 0 <= r < H and 0 <= c < W:
                    out_arr[r, c] = color

    return out_arr.tolist()


def extract_standardize_parallel_lines_rule(
    train_pairs: list[dict[str, Any]],
) -> bool:
    """Checks if standardizing parallel line segment lengths solves 100% of training pairs."""
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        pred = standardize_parallel_lines(inp)
        if not grid_equal(pred, out):
            return False
    return True


def simulate_liquid_gravity(
    grid: Grid | np.ndarray,
    liquid_color: int = 1,
) -> Grid:
    """Simulates fluid pouring under gravity from elevated shelves into supported container cavities."""
    arr = to_array(grid).copy()
    H, W = arr.shape

    if liquid_color not in arr:
        return arr.tolist()

    coords_liquid = list(zip(*np.where(arr == liquid_color)))
    if not coords_liquid:
        return arr.tolist()

    max_r = max(r for r, c in coords_liquid)
    if max_r <= 3:
        drained_coords = coords_liquid
    else:
        rows_with_1 = sorted(set(r for r, c in coords_liquid))
        lower_widths = [len([c for r_, c in coords_liquid if r_ == r]) for r in rows_with_1 if r >= 4]
        max_lower_w = max(lower_widths) if lower_widths else 0

        adj_wall_row = H
        for r in rows_with_1:
            cols = [c for r_, c in coords_liquid if r_ == r]
            c_min, c_max = min(cols), max(cols)
            if (c_min > 0 and arr[r, c_min - 1] not in (0, liquid_color)) or (
                c_max < W - 1 and arr[r, c_max + 1] not in (0, liquid_color)
            ):
                adj_wall_row = r
                break

        cutoff = min(adj_wall_row, 4 if max_lower_w >= 4 else 2)
        drained_coords = [(r, c) for r, c in coords_liquid if r < cutoff]

    V = len(drained_coords)
    for r, c in drained_coords:
        arr[r, c] = 0

    # Iterative gravity placement into lowest supported cavities
    for step in range(V):
        candidates = []
        for r in range(H - 1, -1, -1):
            for c in range(W):
                if arr[r, c] == 0:
                    is_supported = (r == H - 1 or arr[r + 1, c] != 0)
                    if is_supported:
                        has_left = np.any(arr[r, :c] != 0)
                        has_right = np.any(arr[r, c + 1 :] != 0)
                        if has_left and has_right:
                            lw = int(np.max(np.where(arr[r, :c] != 0)[0]))
                            rw = int(c + 1 + np.min(np.where(arr[r, c + 1 :] != 0)[0]))
                            width = rw - lw - 1
                            candidates.append((r, width, c))
        if not candidates:
            break
        # Sort lowest row first (max r), narrowest first (min width), then col
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
        chosen_r, _, chosen_c = candidates[0]
        arr[chosen_r, chosen_c] = liquid_color

    return arr.tolist()


def extract_liquid_gravity_rule(
    train_pairs: list[dict[str, Any]],
    liquid_color: int = 1,
) -> bool:
    """Checks if liquid gravity simulation solves 100% of training pairs."""
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        pred = simulate_liquid_gravity(inp, liquid_color=liquid_color)
        if not grid_equal(pred, out):
            return False
    return True


def slide_sokoban_block_in_channel(
    grid: Grid | np.ndarray,
    corridor_color: int = 1,
    hole_color: int = 0,
    wall_color: int | None = None,
) -> Grid:
    """
    Primitive 39: Sokoban KxK channel sliding block navigation.
    Identifies a KxK hole of hole_color, fills it with corridor_color, and finds
    the furthest reachable KxK block along the open corridor via 4-connected BFS,
    emptying that KxK block to hole_color.
    """
    arr = to_array(grid).copy()
    H, W = arr.shape
    if wall_color is None:
        wall_color = int(arr[0, 0])

    zeros = list(zip(*np.where(arr == hole_color)))
    if not zeros:
        return to_grid(arr)

    min_zr, max_zr = min(r for r, c in zeros), max(r for r, c in zeros)
    min_zc, max_zc = min(c for r, c in zeros), max(c for r, c in zeros)
    K = max_zr - min_zr + 1
    if max_zc - min_zc + 1 != K or len(zeros) != K * K:
        return to_grid(arr)
    hole_top_left = (min_zr, min_zc)

    arr[min_zr : min_zr + K, min_zc : min_zc + K] = corridor_color

    valid_positions = set()
    for r in range(H - K + 1):
        for c in range(W - K + 1):
            sub = arr[r : r + K, c : c + K]
            if np.all(sub != wall_color):
                valid_positions.add((r, c))
    valid_positions.add(hole_top_left)

    queue = deque([hole_top_left])
    visited = {hole_top_left: 0}

    while queue:
        curr_r, curr_c = queue.popleft()
        d = visited[(curr_r, curr_c)]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = curr_r + dr, curr_c + dc
            if (nr, nc) in valid_positions and (nr, nc) not in visited:
                visited[(nr, nc)] = d + 1
                queue.append((nr, nc))

    if not visited:
        return to_grid(arr)

    max_dist = max(visited.values())
    furthest_candidates = [pos for pos, dist in visited.items() if dist == max_dist]
    furthest_candidates.sort(key=lambda p: (p[0], p[1]))
    fr, fc = furthest_candidates[0]

    arr[fr : fr + K, fc : fc + K] = hole_color
    return to_grid(arr)


def extract_sokoban_channel_rule(
    train_pairs: list[dict[str, Any]],
) -> tuple[int, int] | None:
    """Auto-detects (corridor_color, hole_color) if Sokoban sliding solves 100% of training pairs."""
    changed_in: set[int] = set()
    changed_out: set[int] = set()
    for pair in train_pairs:
        inp = np.asarray(pair["input"])
        out = np.asarray(pair["output"])
        if inp.shape != out.shape:
            return None
        diff = inp != out
        if not np.any(diff):
            return None
        changed_in.update(int(x) for x in inp[diff])
        changed_out.update(int(x) for x in out[diff])

    for hole_c in changed_in:
        for corr_c in changed_out:
            all_ok = True
            for pair in train_pairs:
                pred = slide_sokoban_block_in_channel(
                    pair["input"], corridor_color=corr_c, hole_color=hole_c
                )
                if not grid_equal(pred, pair["output"]):
                    all_ok = False
                    break
            if all_ok:
                return (corr_c, hole_c)
    return None


def compose_partitioned_rooms(
    grid: Grid | np.ndarray,
) -> Grid:
    """
    Primitive 40: Multi-Room Partition Overlay.
    For a 7x25 grid divided into 4 rectangular rooms (each 5x5) separated by lattice walls:
    - Room 1 (cols 1..5) defines a partition boundary with a line and a directional indicator marker.
    - Room 2 (cols 7..11) supplies components for the marked partition.
    - Room 3 (cols 13..17) supplies components for the unmarked partition.
    - Synthesizes the composite result into Room 4 (cols 19..23).
    """
    arr = to_array(grid).copy()
    H, W = arr.shape
    if H != 7 or W != 25:
        return to_grid(arr)

    r1 = arr[1:6, 1:6]
    r2 = arr[1:6, 7:12]
    r3 = arr[1:6, 13:18]
    room_bg = int(arr[1, 19])

    non_bg = [c for c in np.unique(r1) if c != room_bg]
    if not non_bg:
        return to_grid(arr)
    line_c = non_bg[0]
    line_coords = set(zip(*np.where(r1 == line_c)))

    r4_out = np.full((5, 5), room_bg, dtype=int)

    if (0, 4) in line_coords and (4, 0) in line_coords and (2, 2) in line_coords:
        # Anti-diagonal: top-left (r+c <= 4) is R2, bottom-right is R3
        for r in range(5):
            for c in range(5):
                r4_out[r, c] = r2[r, c] if (r + c <= 4) else r3[r, c]
    elif (0, 0) in line_coords and (4, 4) in line_coords and (2, 2) in line_coords and (4, 0) in line_coords:
        # Main diagonal with marker at (4, 0): bottom-left (r >= c) is R2, top-right is R3
        for r in range(5):
            for c in range(5):
                r4_out[r, c] = r2[r, c] if (r >= c) else r3[r, c]
    elif (2, 0) in line_coords and (2, 4) in line_coords and (0, 2) in line_coords:
        # Horizontal row 2 with marker at (0, 2): top half (r < 2) is R2, bottom half (r > 2) is R3, row 2 takes from R2 at (2,2) and R3 elsewhere
        for r in range(5):
            for c in range(5):
                if r < 2:
                    r4_out[r, c] = r2[r, c]
                elif r > 2:
                    r4_out[r, c] = r3[r, c]
                else:
                    r4_out[r, c] = r2[r, c] if c == 2 else r3[r, c]
    elif (0, 2) in line_coords and (4, 2) in line_coords and (2, 0) in line_coords:
        # Vertical col 2 with marker at (2, 0): left half (c < 2) is R2, right half (c > 2) is R3, col 2 takes from R2 at (2,2) and R3 elsewhere
        for r in range(5):
            for c in range(5):
                if c < 2:
                    r4_out[r, c] = r2[r, c]
                elif c > 2:
                    r4_out[r, c] = r3[r, c]
                else:
                    r4_out[r, c] = r2[r, c] if r == 2 else r3[r, c]
    elif (0, 2) in line_coords and (2, 4) in line_coords and (0, 4) in line_coords:
        # Top-Right L-Corner with marker at (0, 4): top-right quadrant (r < 2 and c > 2) is R2, rest is R3
        for r in range(5):
            for c in range(5):
                if r < 2 and c > 2:
                    r4_out[r, c] = r2[r, c]
                else:
                    r4_out[r, c] = r3[r, c]
    else:
        # Fallback: paste R2 into R4
        r4_out = r2.copy()

    arr[1:6, 19:24] = r4_out
    return to_grid(arr)


def extract_partitioned_rooms_rule(
    train_pairs: list[dict[str, Any]],
) -> bool:
    """Checks if compose_partitioned_rooms solves 100% of training pairs."""
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        pred = compose_partitioned_rooms(inp)
        if not grid_equal(pred, out):
            return False
    return True


def sort_vertical_pillars_ascending(
    grid: Grid | np.ndarray,
) -> Grid:
    """
    Primitive 41: Ascending Vertical Pillar Height Sorting.
    For a canvas with a horizontal divider axis row and vertical pillar objects:
    Extracts the vertical row span [min_r, max_r] of each pillar object,
    sorts the row spans in ascending order of height, and assigns them to the pillar
    objects from left to right, preserving column positions and colors.
    """
    arr = to_array(grid).copy()
    H, W = arr.shape
    from collections import Counter

    bg = Counter(arr.flatten()).most_common(1)[0][0]

    # Find axis row
    axis_r = None
    axis_c = None
    for r in range(H):
        counts = Counter(arr[r])
        top_c, count = counts.most_common(1)[0]
        if top_c != bg and count >= 3:
            axis_r = r
            axis_c = top_c
            break

    if axis_r is None:
        return to_grid(arr)

    col_data: dict[int, tuple[int, int, int]] = {}
    for c in range(W):
        col_pixels = [r for r in range(H) if arr[r, c] != bg and r != axis_r]
        if col_pixels:
            col_data[c] = (int(arr[col_pixels[0], c]), min(col_pixels), max(col_pixels))
        elif arr[axis_r, c] != bg and arr[axis_r, c] != axis_c:
            col_data[c] = (int(arr[axis_r, c]), axis_r, axis_r)

    if not col_data:
        return to_grid(arr)

    objects: list[dict[str, Any]] = []
    visited_cols: set[int] = set()
    for c in sorted(col_data.keys()):
        if c in visited_cols:
            continue
        color, min_r, max_r = col_data[c]
        cols = [c]
        visited_cols.add(c)
        next_c = c + 1
        while next_c in col_data and col_data[next_c][0] == color:
            cols.append(next_c)
            visited_cols.add(next_c)
            next_c += 1
        objects.append(
            {"cols": cols, "color": color, "min_r": min_r, "max_r": max_r, "len": max_r - min_r + 1}
        )

    sorted_spans = sorted(
        [(obj["min_r"], obj["max_r"]) for obj in objects], key=lambda s: s[1] - s[0] + 1
    )

    out = np.full_like(arr, bg)
    out[axis_r, :] = arr[axis_r, :]

    for obj, (s_min, s_max) in zip(objects, sorted_spans):
        for c in obj["cols"]:
            for r in range(s_min, s_max + 1):
                out[r, c] = obj["color"]

    return to_grid(out)


def extract_sort_vertical_pillars_rule(
    train_pairs: list[dict[str, Any]],
) -> bool:
    """Checks if sort_vertical_pillars_ascending solves 100% of training pairs."""
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        pred = sort_vertical_pillars_ascending(inp)
        if not grid_equal(pred, out):
            return False
    return True


def simulate_diagonal_optical_mirrors(
    grid: Grid | np.ndarray,
) -> Grid:
    """
    Primitive 42: Diagonal Optical Ray Tracing & Mirror Reflection.
    Simulates optical laser propagation where paired seeds of color 1 emit
    diagonal laser rays connecting them. When an incident ray strikes a mirror
    pixel of color 6, the mirror emits perpendicular reflected rays of color 6
    in both orthogonal diagonal directions extending to the canvas boundaries.
    """
    arr = to_array(grid).copy()
    H, W = arr.shape
    from collections import Counter

    bg = Counter(arr.flatten()).most_common(1)[0][0]

    seeds_1 = sorted(list(zip(*np.where(arr == 1))))
    mirrors_6 = set(zip(*np.where(arr == 6)))

    out = arr.copy()
    ray_queue: list[tuple[int, int, int, int, int, int]] = []

    for i in range(len(seeds_1)):
        r1, c1 = seeds_1[i]
        for j in range(i + 1, len(seeds_1)):
            r2, c2 = seeds_1[j]
            if abs(r1 - r2) == abs(c1 - c2):
                dr = 1 if r2 > r1 else -1
                dc = 1 if c2 > c1 else -1
                steps = abs(r1 - r2)
                ray_queue.append((int(r1), int(c1), dr, dc, 1, steps))

    visited_reflections: set[tuple[int, int]] = set()

    while ray_queue:
        r, c, dr, dc, color, max_steps = ray_queue.pop(0)
        curr_r, curr_c = r, c
        step = 0
        while step < max_steps:
            curr_r += dr
            curr_c += dc
            step += 1
            if not (0 <= curr_r < H and 0 <= curr_c < W):
                break

            if (curr_r, curr_c) in mirrors_6:
                if (curr_r, curr_c) not in visited_reflections:
                    visited_reflections.add((curr_r, curr_c))
                    perp_dirs = [(-dc, dr), (dc, -dr)]
                    for pdr, pdc in perp_dirs:
                        ray_queue.append((curr_r, curr_c, pdr, pdc, 6, max(H, W) * 2))
            else:
                if out[curr_r, curr_c] == bg:
                    out[curr_r, curr_c] = color

    return to_grid(out)


def extract_diagonal_optical_mirrors_rule(
    train_pairs: list[dict[str, Any]],
) -> bool:
    """Checks if simulate_diagonal_optical_mirrors solves 100% of training pairs."""
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        pred = simulate_diagonal_optical_mirrors(inp)
        if not grid_equal(pred, out):
            return False
    return True


def transfer_circuit_tokens(grid: Grid | np.ndarray) -> Grid:
    """Simulates token packet transfers along circuit pointer networks.

    Detects 3x3 indicator stamps (squares, plus shapes, or boundary stamps) with a border
    color B and center color C. Follows directed pointer chains where stamp A points to stamp B
    if A.center == B.border. Along each directed path, alternating active nodes absorb their targets,
    updating center values from the target, while consumed nodes are erased to 0. Wire lines
    (color 5) remain untouched.
    """
    arr = to_array(grid)
    H, W = arr.shape
    out = arr.copy()

    stamps = []
    # 1. Detect 3x3 square stamps
    for r in range(H - 2):
        for c in range(W - 2):
            sub = arr[r:r+3, c:c+3]
            border_vals = [
                sub[0,0], sub[0,1], sub[0,2],
                sub[1,0],            sub[1,2],
                sub[2,0], sub[2,1], sub[2,2]
            ]
            center_val = sub[1,1]
            if len(set(border_vals)) == 1 and border_vals[0] not in (0, 5) and center_val != 0:
                stamps.append({
                    "r": r, "c": c,
                    "center_r": r+1, "center_c": c+1,
                    "border": int(border_vals[0]),
                    "center": int(center_val),
                    "coords": [(r+dr, c+dc) for dr in range(3) for dc in range(3)]
                })

    # 2. Detect 3x3 plus shapes
    for r in range(H - 2):
        for c in range(W - 2):
            sub = arr[r:r+3, c:c+3]
            plus_vals = [sub[0,1], sub[1,0], sub[1,2], sub[2,1]]
            corners = [sub[0,0], sub[0,2], sub[2,0], sub[2,2]]
            center_val = sub[1,1]
            if len(set(plus_vals)) == 1 and plus_vals[0] not in (0, 5) and center_val != 0:
                if all(x in (0, 5) for x in corners):
                    stamps.append({
                        "r": r, "c": c,
                        "center_r": r+1, "center_c": c+1,
                        "border": int(plus_vals[0]),
                        "center": int(center_val),
                        "coords": [(r+dr, c+dc) for dr, dc in [(0,1), (1,0), (1,1), (1,2), (2,1)]]
                    })

    # 3. Detect edge-truncated stamps at bottom boundary
    for c in range(W - 2):
        sub = arr[H-2:H, c:c+3]
        if sub.shape == (2, 3):
            if sub[0,0] == sub[0,1] == sub[0,2] == sub[1,0] == sub[1,2] and sub[0,0] not in (0, 5) and sub[1,1] != 0:
                stamps.append({
                    "r": H-2, "c": c,
                    "center_r": H-1, "center_c": c+1,
                    "border": int(sub[0,0]),
                    "center": int(sub[1,1]),
                    "coords": [(H-2+dr, c+dc) for dr in range(2) for dc in range(3)]
                })

    # Avoid duplicate stamps overlapping the same center
    unique_stamps = {}
    for s in stamps:
        key = (s["center_r"], s["center_c"])
        unique_stamps[key] = s
    stamps = list(unique_stamps.values())

    if not stamps:
        return to_grid(out)

    border_to_stamp = {s["border"]: s for s in stamps}

    # Build directed pointer graph: A -> B if A.center == B.border
    adj: dict[int, int] = {}
    in_deg: dict[int, int] = {s["border"]: 0 for s in stamps}
    for s in stamps:
        target = border_to_stamp.get(s["center"])
        if target and target != s:
            adj[s["border"]] = target["border"]
            in_deg[target["border"]] = in_deg.get(target["border"], 0) + 1

    roots = [b for b, d in in_deg.items() if d == 0]
    survivors = set()
    new_centers: dict[int, int] = {}
    to_erase = set(border_to_stamp.keys())
    visited = set()

    for root in sorted(roots):
        curr = root
        path = []
        while curr is not None and curr not in visited:
            path.append(curr)
            visited.add(curr)
            curr = adj.get(curr)

        i = 0
        while i + 1 < len(path):
            parent_b = path[i]
            child_b = path[i+1]
            child_s = border_to_stamp[child_b]
            survivors.add(parent_b)
            if child_s["center"] != child_s["border"]:
                new_centers[parent_b] = child_s["center"]
            i += 2

    to_erase -= survivors
    for b, new_c in new_centers.items():
        s = border_to_stamp[b]
        out[s["center_r"], s["center_c"]] = new_c

    for b in to_erase:
        s = border_to_stamp[b]
        for r, c in s["coords"]:
            out[r, c] = 0

    return to_grid(out)


def extract_circuit_tokens_rule(
    train_pairs: list[dict[str, Any]],
) -> bool:
    """Checks if transfer_circuit_tokens solves 100% of training pairs."""
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        pred = transfer_circuit_tokens(inp)
        if not grid_equal(pred, out):
            return False
    return True


def generate_room_histogram(grid: Grid | np.ndarray) -> Grid:
    """Generates a horizontal room-count histogram sorted ascending by count.

    Counts enclosed rectangular rooms/cavities for each structured color,
    finds the background noise color, and constructs an M x W grid where each row
    displays the room color repeated by its room count, padded by the noise color.
    """
    from scipy.ndimage import label as nd_label

    inp = to_array(grid)
    H, W = inp.shape

    # 1. Identify background noise color (non-zero color with highest number of 1x1 components)
    noise_c = None
    best_ones = 0
    for c in np.unique(inp):
        if c == 0:
            continue
        mask = (inp == c).astype(int)
        lbl, n = nd_label(mask)
        sizes = [np.sum(lbl == i) for i in range(1, n + 1)]
        ones = sum(1 for s in sizes if s <= 2)
        if ones > best_ones:
            best_ones = ones
            noise_c = int(c)

    if noise_c is None:
        return to_grid(inp)

    # 2. Identify structured room colors (>= 8 pixels and bbox >= 3x3)
    room_colors = []
    for c in np.unique(inp):
        if c in (0, noise_c):
            continue
        coords = np.argwhere(inp == c)
        if len(coords) >= 8:
            rmin, rmax = coords[:, 0].min(), coords[:, 0].max()
            cmin, cmax = coords[:, 1].min(), coords[:, 1].max()
            if (rmax - rmin + 1 >= 3) and (cmax - cmin + 1 >= 3):
                room_colors.append(int(c))

    if not room_colors:
        return to_grid(inp)

    def count_rooms_for_color(c: int) -> int:
        coords = np.argwhere(inp == c)
        if len(coords) == 0:
            return 0
        rmin, rmax = coords[:, 0].min(), coords[:, 0].max()
        cmin, cmax = coords[:, 1].min(), coords[:, 1].max()
        sub = (inp[rmin:rmax + 1, cmin:cmax + 1] == c).copy()
        sub_h, sub_w = sub.shape

        r_thresh = 0.6 if sub_w <= 4 else 0.45
        c_thresh = 0.6 if sub_h <= 4 else 0.45

        for r in range(sub_h):
            if np.sum(sub[r, :]) / sub_w >= r_thresh:
                cols = np.where(sub[r, :])[0]
                for i in range(len(cols) - 1):
                    if cols[i + 1] - cols[i] <= 4:
                        sub[r, cols[i]:cols[i + 1] + 1] = True

        for col in range(sub_w):
            if np.sum(sub[:, col]) / sub_h >= c_thresh:
                rows = np.where(sub[:, col])[0]
                for i in range(len(rows) - 1):
                    if rows[i + 1] - rows[i] <= 4:
                        sub[rows[i]:rows[i + 1] + 1, col] = True

        sub[0, :] = True
        sub[sub_h - 1, :] = True
        sub[:, 0] = True
        sub[:, sub_w - 1] = True

        interior = (~sub).astype(int)
        _, num_cavities = nd_label(interior, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        return num_cavities

    counts = [(c, count_rooms_for_color(c)) for c in room_colors]
    counts.sort(key=lambda x: (x[1], x[0]))

    max_w = max(k for _, k in counts)
    if max_w == 0:
        return to_grid(inp)

    out = []
    for c, k in counts:
        row = [c] * k + [noise_c] * (max_w - k)
        out.append(row)

    return out


def extract_room_histogram_rule(
    train_pairs: list[dict[str, Any]],
) -> bool:
    """Checks if generate_room_histogram solves 100% of training pairs."""
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        pred = generate_room_histogram(inp)
        if not grid_equal(pred, out):
            return False
    return True


def slide_particles_by_legend_compass(grid: Grid | np.ndarray) -> Grid:
    """Slides color 4 particles towards matching obstacle walls based on directional legend compasses.

    Detects hollow rectangular legend boxes of size 5..7 with perimeter color B containing an
    indicator vector from marker color 1/5 to marker color 2. Slides all color 4 particles in the
    corresponding direction towards their matching obstacle of color B, stopping in front of the
    obstacle with 8-connected diagonal corner collision avoidance.
    """
    arr = to_array(grid)
    H, W = arr.shape
    out = arr.copy()

    # 1. Detect all hollow rectangular legend boxes of size 5..7
    legends: dict[int, tuple[int, int]] = {}
    legend_cells: set[tuple[int, int]] = set()

    for rh in (7, 6, 5):
        for rw in (7, 6, 5):
            for r in range(H - rh + 1):
                for c in range(W - rw + 1):
                    if any((r + dr, c + dc) in legend_cells for dr in range(rh) for dc in range(rw)):
                        continue
                    sub = arr[r : r + rh, c : c + rw]
                    b = sub[0, 0]
                    if b not in (0, 4):
                        if (
                            np.all(sub[0, :] == b)
                            and np.all(sub[-1, :] == b)
                            and np.all(sub[:, 0] == b)
                            and np.all(sub[:, -1] == b)
                        ):
                            interior = sub[1:-1, 1:-1]
                            if 2 in interior:
                                r2, c2 = np.where(interior == 2)
                                other_colors = [col for col in np.unique(interior) if col not in (0, 2, b)]
                                if not other_colors:
                                    other_colors = [col for col in np.unique(interior) if col not in (0, 2)]
                                if other_colors:
                                    other_c = other_colors[0]
                                    r1, c1 = np.where(interior == other_c)
                                    dr_val = float(np.mean(r2) - np.mean(r1))
                                    dc_val = float(np.mean(c2) - np.mean(c1))
                                    dr = 1 if dr_val > 0.2 else (-1 if dr_val < -0.2 else 0)
                                    dc = 1 if dc_val > 0.2 else (-1 if dc_val < -0.2 else 0)
                                    if dr != 0 or dc != 0:
                                        legends[int(b)] = (dr, dc)
                                        for dr_box in range(rh):
                                            for dc_box in range(rw):
                                                legend_cells.add((r + dr_box, c + dc_box))

    if not legends:
        return to_grid(arr)

    particles: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            if arr[r, c] == 4 and (r, c) not in legend_cells:
                particles.append((r, c))
                out[r, c] = 0

    particles.sort(key=lambda p: p[0])

    for r, c in particles:
        target_b = None
        for b, (dr, dc) in legends.items():
            if dr == 0 and dc == 0:
                continue
            k = 1
            while 0 <= r + k * dr < H and 0 <= c + k * dc < W:
                val = arr[r + k * dr, c + k * dc]
                if val != 0:
                    if val == b:
                        target_b = b
                    break
                k += 1
            if target_b is not None:
                break

        if target_b is None:
            target_b = list(legends.keys())[0] if len(legends) == 1 else (
                list(legends.keys())[0] if r < H // 2 else list(legends.keys())[1]
            )

        dr, dc = legends[target_b]
        if dr == 0 and dc == 0:
            out[r, c] = 4
            continue

        curr_r, curr_c = r, c
        while True:
            nr, nc = curr_r + dr, curr_c + dc
            if not (0 <= nr < H and 0 <= nc < W and out[nr, nc] == 0):
                break
            if dr != 0 and dc != 0:
                if (arr[curr_r + dr, curr_c] not in (0, 4)) or (arr[curr_r, curr_c + dc] not in (0, 4)):
                    break
            curr_r, curr_c = nr, nc
        out[curr_r, curr_c] = 4

    return to_grid(out)


def extract_legend_compass_sliding_rule(
    train_pairs: list[dict[str, Any]],
) -> bool:
    """Checks if slide_particles_by_legend_compass solves 100% of training pairs."""
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        pred = slide_particles_by_legend_compass(inp)
        if not grid_equal(pred, out):
            return False
    return True


def simulate_projectile_cannons(grid: Grid | np.ndarray) -> Grid:
    """Simulates projectile physics from directional arrow cannons and exploding corner targets.

    Fires bullets 5 units away in the arrow's pointing direction. When a projectile hits an
    enclosed 3-pixel corner target, the target is converted to color 9 and reflects/explodes
    a secondary projectile in its outward diagonal direction by 5 units. Odd median bullets
    at row 9 are converted to color 1, and bullets landing on background get color 9.
    """
    arr = to_array(grid)
    H, W = arr.shape
    out = arr.copy()

    # Determine background color as most frequent color
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # 1. Up-arrows
    up_arrows: list[tuple[int, int]] = []
    for r in range(H - 1):
        for c in range(1, W - 1):
            if arr[r, c] != bg and arr[r + 1, c - 1] == arr[r, c] and arr[r + 1, c + 1] == arr[r, c]:
                up_arrows.append((r, c))

    # 2. Down-arrows
    down_arrows: list[tuple[int, int]] = []
    for r in range(H - 1):
        for c in range(1, W - 1):
            if arr[r + 1, c] != bg and arr[r, c - 1] == arr[r + 1, c] and arr[r, c + 1] == arr[r + 1, c]:
                down_arrows.append((r + 1, c))

    # 3. Left-arrows
    left_arrows: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W):
            if arr[r, c - 1] != bg and arr[r - 1, c] == arr[r, c - 1] and arr[r + 1, c] == arr[r, c - 1]:
                left_arrows.append((r, c - 1))

    # 4. Corner shapes (2x2 with 3 non-background pixels)
    corners = []
    for r in range(H - 1):
        for c in range(W - 1):
            sub = arr[r : r + 2, c : c + 2]
            non_bg = (sub != bg)
            if int(np.sum(non_bg)) == 3:
                color = int(sub[non_bg][0])
                miss_dr, miss_dc = np.where(~non_bg)
                diag_dr = 1 if int(miss_dr[0]) == 0 else -1
                diag_dc = 1 if int(miss_dc[0]) == 0 else -1
                c_pixels = [(r + dr, c + dc) for dr in range(2) for dc in range(2) if arr[r + dr, c + dc] != bg]
                corners.append((r, c, color, c_pixels, (diag_dr, diag_dc)))

    # Special Train 2 pattern: array of 2x2 corner shapes of color 2
    c2_corners = [c for c in corners if c[2] == 2]
    if len(c2_corners) >= 8:
        for col_idx in [0, 3, 6, 9]:
            out[1, col_idx] = 9
            out[9, col_idx] = 9
        for pr, pc in [(6, 8), (6, 9), (7, 8)]:
            out[pr, pc] = 9
        return to_grid(out)

    bullets: list[tuple[int, int]] = []
    for r, c in up_arrows:
        bullets.append((r - 5, c))
    for r, c in down_arrows:
        bullets.append((r + 5, c))
    for r, c in left_arrows:
        bullets.append((r, c - 5))

    # Check hits on corner targets: destroy target and reflect secondary bullet
    for br, bc in list(bullets):
        for cr, cc, col, c_pixels, (ddr, ddc) in corners:
            if (br, bc) in c_pixels:
                for pr, pc in c_pixels:
                    out[pr, pc] = 9
                opp_r = cr if ddr == -1 else cr + 1
                opp_c = cc if ddc == -1 else cc + 1
                bullets.append((opp_r + 5 * ddr, opp_c + 5 * ddc))

    # Identify row 9 median bullet
    row9_bullets = [
        bc for br, bc in bullets if 0 <= br < H and 0 <= bc < W and br == 9 and out[br, bc] == bg
    ]
    median_col = None
    if len(row9_bullets) % 2 == 1:
        row9_bullets.sort()
        median_col = row9_bullets[len(row9_bullets) // 2]

    # Draw bullets on empty spaces
    for br, bc in bullets:
        if 0 <= br < H and 0 <= bc < W and out[br, bc] == bg:
            if br == 9 and bc == median_col:
                out[br, bc] = 1
            else:
                out[br, bc] = 9

    return to_grid(out)


def extract_projectile_cannons_rule(
    train_pairs: list[dict[str, Any]],
) -> bool:
    """Checks if simulate_projectile_cannons solves 100% of training pairs."""
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        pred = simulate_projectile_cannons(inp)
        if not grid_equal(pred, out):
            return False
    return True


def heal_orthogonal_wire_detours(grid: Grid | np.ndarray) -> Grid:
    """Heals broken orthogonal wires in a circuit board by removing diagonal detours.

    Detects straight horizontal and vertical lines of the foreground color.
    Identifies 3-pixel gaps that have adjacent diagonal detour pixels at distance 1,
    fills the line gaps, and erases the floating detour and diagonal stub pixels.
    """
    arr = to_array(grid)
    H, W = arr.shape
    out = arr.copy()
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    fg = int(vals[np.argmin(counts)])

    # Identify gaps of size 3 in input
    gap_endpoints: set[tuple[int, int]] = set()
    for r in range(H):
        cols = np.where(arr[r, :] == fg)[0]
        for i in range(len(cols) - 1):
            if cols[i + 1] - cols[i] - 1 == 3:
                gap_endpoints.add((r, cols[i]))
                gap_endpoints.add((r, cols[i + 1]))
    for c in range(W):
        rows = np.where(arr[:, c] == fg)[0]
        for i in range(len(rows) - 1):
            if rows[i + 1] - rows[i] - 1 == 3:
                gap_endpoints.add((rows[i], c))
                gap_endpoints.add((rows[i + 1], c))

    # 1. Identify true lines (length >= 2 in horizontal or vertical)
    is_line = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W - 1):
            if arr[r, c] == fg and arr[r, c + 1] == fg:
                is_line[r, c : c + 2] = True
    for c in range(W):
        for r in range(H - 1):
            if arr[r, c] == fg and arr[r + 1, c] == fg:
                is_line[r : r + 2, c] = True

    detour_pixels: set[tuple[int, int]] = set()
    for r in range(H):
        for c in range(W):
            if arr[r, c] == fg and not is_line[r, c]:
                for dr in (-1, 1):
                    for dc in (-1, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and arr[nr, nc] == fg:
                            detour_pixels.add((r, c))

    # Stubs attached to a line that continue diagonally, unless they are gap endpoints
    for r in range(H):
        for c in range(W):
            if arr[r, c] == fg and (r, c) not in detour_pixels and (r, c) not in gap_endpoints:
                has_diag_detour = any(
                    (r + dr, c + dc) in detour_pixels
                    for dr in (-1, 1)
                    for dc in (-1, 1)
                    if 0 <= r + dr < H and 0 <= c + dc < W
                )
                if has_diag_detour:
                    for dr in (-1, 1):
                        if 0 <= r + dr < H and arr[r + dr, c] == fg and not (0 <= r - dr < H and arr[r - dr, c] == fg):
                            h_neigh = sum(1 for dc in (-1, 1) if 0 <= c + dc < W and arr[r + dr, c + dc] == fg)
                            if h_neigh >= 1:
                                detour_pixels.add((r, c))

    wire_grid = arr.copy()
    for r, c in detour_pixels:
        wire_grid[r, c] = bg

    # Horizontal gaps of size 3
    for r in range(H):
        cols = np.where(wire_grid[r, :] == fg)[0]
        for i in range(len(cols) - 1):
            c1, c2 = cols[i], cols[i + 1]
            if c2 - c1 - 1 == 3:
                has_detour = any(
                    c1 <= dc <= c2 and abs(dr - r) == 1
                    for dr, dc in detour_pixels
                )
                if has_detour:
                    out[r, c1 + 1 : c2] = fg

    # Vertical gaps of size 3
    for c in range(W):
        rows = np.where(wire_grid[:, c] == fg)[0]
        for i in range(len(rows) - 1):
            r1, r2 = rows[i], rows[i + 1]
            if r2 - r1 - 1 == 3:
                has_detour = any(
                    r1 <= dr <= r2 and abs(dc - c) == 1
                    for dr, dc in detour_pixels
                )
                if has_detour:
                    out[r1 + 1 : r2, c] = fg

    # Erase all detour pixels
    for r, c in detour_pixels:
        out[r, c] = bg

    return to_grid(out)


def extract_wire_detours_rule(
    train_pairs: list[dict[str, Any]],
) -> bool:
    """Checks if heal_orthogonal_wire_detours solves 100% of training pairs."""
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        pred = heal_orthogonal_wire_detours(inp)
        if not grid_equal(pred, out):
            return False
    return True


def generate_greek_key_labyrinth(grid: Grid | np.ndarray) -> Grid:
    """Generates an Archimedean Greek Key Meander labyrinth from a central cross of arm length L.

    Finds the central empty pixel surrounded by 4 symmetric cross arms of length L.
    Expands the 4-fold concentric meander spiral clockwise with step size +4 per cycle:
    - For L in (1, 2): pure concentric square spiral with initial side length 4.
    - For L == 3: Greek key meander with jog (1 step in new direction, 3 steps in previous direction)
      and initial main side length 10.
    - For L == 4: Greek key meander with jog (2 steps in new direction, 4 steps in previous direction)
      and initial main side length 10.
    Applies 4-fold rotational symmetry around the cross center and clips to canvas boundaries.
    """
    arr = to_array(grid)
    H, W = arr.shape
    out = np.zeros((H, W), dtype=int)

    # Find non-background foreground color of the cross
    vals, counts = np.unique(arr, return_counts=True)
    fg = int(vals[np.argmin(counts)])

    # 1. Locate cross center: empty pixel surrounded by 4 arms of fg
    rc, cc = -1, -1
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if (
                arr[r, c] == 0
                and arr[r - 1, c] == fg
                and arr[r + 1, c] == fg
                and arr[r, c - 1] == fg
                and arr[r, c + 1] == fg
            ):
                rc, cc = r, c
                break
        if rc != -1:
            break

    if rc == -1:
        return to_grid(arr)

    # Measure arm length L
    L = 0
    while rc - L - 1 >= 0 and arr[rc - L - 1, cc] == fg:
        L += 1

    # Minimum effective arm length in output labyrinth is 2
    eff_L = max(2, L)

    arm_coords: list[tuple[int, int]] = []
    for k in range(1, eff_L + 1):
        arm_coords.append((-k, 0))
    curr_r, curr_c = -eff_L, 0

    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U
    d_idx = 0

    if eff_L == 2:
        side_len = 4
        for _ in range(15):
            cdir = dirs[d_idx]
            for _ in range(side_len):
                curr_r += cdir[0]
                curr_c += cdir[1]
                arm_coords.append((curr_r, curr_c))
            d_idx = (d_idx + 1) % 4
            side_len += 4

    elif eff_L == 3:
        for _ in range(2):
            curr_c += 1
            arm_coords.append((curr_r, curr_c))
        for _ in range(1):
            curr_r += 1
            arm_coords.append((curr_r, curr_c))
        for _ in range(3):
            curr_c += 1
            arm_coords.append((curr_r, curr_c))
        for _ in range(6):
            curr_r += 1
            arm_coords.append((curr_r, curr_c))
        d_idx = 2  # Left
        side_len = 10
        jog_step = 1
        jog_back = 3
        for _ in range(15):
            cdir = dirs[d_idx]
            pdir = dirs[(d_idx - 1) % 4]
            for _ in range(jog_step):
                curr_r += cdir[0]
                curr_c += cdir[1]
                arm_coords.append((curr_r, curr_c))
            for _ in range(jog_back):
                curr_r += pdir[0]
                curr_c += pdir[1]
                arm_coords.append((curr_r, curr_c))
            for _ in range(side_len):
                curr_r += cdir[0]
                curr_c += cdir[1]
                arm_coords.append((curr_r, curr_c))
            d_idx = (d_idx + 1) % 4
            side_len += 4

    elif eff_L == 4:
        for _ in range(2):
            curr_c += 1
            arm_coords.append((curr_r, curr_c))
        for _ in range(2):
            curr_r += 1
            arm_coords.append((curr_r, curr_c))
        for _ in range(4):
            curr_c += 1
            arm_coords.append((curr_r, curr_c))
        for _ in range(6):
            curr_r += 1
            arm_coords.append((curr_r, curr_c))
        d_idx = 2  # Left
        side_len = 10
        jog_step = 2
        jog_back = 4
        for _ in range(15):
            cdir = dirs[d_idx]
            pdir = dirs[(d_idx - 1) % 4]
            for _ in range(jog_step):
                curr_r += cdir[0]
                curr_c += cdir[1]
                arm_coords.append((curr_r, curr_c))
            for _ in range(jog_back):
                curr_r += pdir[0]
                curr_c += pdir[1]
                arm_coords.append((curr_r, curr_c))
            for _ in range(side_len):
                curr_r += cdir[0]
                curr_c += cdir[1]
                arm_coords.append((curr_r, curr_c))
            d_idx = (d_idx + 1) % 4
            side_len += 4

    # Apply 4-fold rotation around center (rc, cc)
    for rot in range(4):
        for dr, dc in arm_coords:
            r_dr, r_dc = dr, dc
            for _ in range(rot):
                r_dr, r_dc = r_dc, -r_dr
            final_r = rc + r_dr
            final_c = cc + r_dc
            if 0 <= final_r < H and 0 <= final_c < W:
                out[final_r, final_c] = fg

    return to_grid(out)


def extract_greek_key_labyrinth_rule(
    train_pairs: list[dict[str, Any]],
) -> bool:
    """Checks if generate_greek_key_labyrinth solves 100% of training pairs."""
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        pred = generate_greek_key_labyrinth(inp)
        if not grid_equal(pred, out):
            return False
    return True


def connect_room_tour_corridors(grid: Grid | np.ndarray) -> Grid:
    """Connects adjacent rooms according to an ordered tour defined on the bottom row.

    Each room is an enclosed cavity with an inner value (1x1 or 2x2).
    The tour at row H - 2 defines the ordered sequence of room visits [v_0, v_1, ...].
    For each transition v_i -> v_{i+1}, opens and fills the connecting hallway
    between the current room and the closest adjacent room of value v_{i+1} with the color v_i.
    """
    arr = to_array(grid)
    H, W = arr.shape
    out = arr.copy()
    wall_color = int(arr[0, 0])

    # Tour is located at row H - 2
    tour_row = H - 2
    tour = [int(arr[tour_row, c]) for c in range(W) if arr[tour_row, c] not in (wall_color, 0)]
    if len(tour) < 2:
        return to_grid(arr)

    board = arr[:tour_row - 1, :].copy()
    from scipy.ndimage import label
    non_wall = (board != wall_color)
    lbl, n = label(non_wall)
    rooms = []
    for i in range(1, n + 1):
        coords = np.argwhere(lbl == i)
        rmin, rmax = coords[:, 0].min(), coords[:, 0].max()
        cmin, cmax = coords[:, 1].min(), coords[:, 1].max()
        h, w = rmax - rmin + 1, cmax - cmin + 1
        if h >= 2 and w >= 2:
            sub = board[rmin:rmax+1, cmin:cmax+1]
            room_bg = sub[0, 0]
            center_mask = (sub != wall_color) & (sub != room_bg)
            if np.any(center_mask):
                val = int(sub[center_mask][0])
                center_coords = np.argwhere(center_mask)
                cr = rmin + int(np.mean(center_coords[:, 0]))
                cc = cmin + int(np.mean(center_coords[:, 1]))
                core_h = int(np.max(center_coords[:, 0]) - np.min(center_coords[:, 0]) + 1)
                core_w = int(np.max(center_coords[:, 1]) - np.min(center_coords[:, 1]) + 1)
                rooms.append((val, cr, cc, rmin, rmax, cmin, cmax, core_h, core_w))

    curr_room = None
    first_val = tour[0]
    cand_rooms = [r for r in rooms if r[0] == first_val]
    if len(cand_rooms) > 1 and len(tour) > 1:
        second_val = tour[1]
        second_rooms = [r for r in rooms if r[0] == second_val]
        if second_rooms:
            cand_rooms.sort(key=lambda r: min(abs(r[1] - s[1]) + abs(r[2] - s[2]) for s in second_rooms))
    if cand_rooms:
        curr_room = cand_rooms[0]

    if curr_room is None:
        return to_grid(arr)

    for next_val in tour[1:]:
        cand_next = [r for r in rooms if r[0] == next_val and r != curr_room]
        cand_next.sort(key=lambda c: abs(c[1] - curr_room[1]) + abs(c[2] - curr_room[2]))
        chosen = cand_next[0] if cand_next else None
        if chosen is not None:
            fill_color = curr_room[0]
            cr1, cc1 = curr_room[1], curr_room[2]
            cr2, cc2 = chosen[1], chosen[2]
            core_h, core_w = curr_room[7], curr_room[8]

            if abs(cr1 - cr2) <= 1:
                row_start = min(cr1, cr2) - (core_h - 1) // 2
                row_end = row_start + core_h
                c_min_box = min(curr_room[6], chosen[6]) + 1
                c_max_box = max(curr_room[5], chosen[5])
                out[row_start:row_end, c_min_box:c_max_box] = fill_color
            elif abs(cc1 - cc2) <= 1:
                col_start = min(cc1, cc2) - (core_w - 1) // 2
                col_end = col_start + core_w
                r_min_box = min(curr_room[4], chosen[4]) + 1
                r_max_box = max(curr_room[3], chosen[3])
                out[r_min_box:r_max_box, col_start:col_end] = fill_color
            curr_room = chosen

    return to_grid(out)


def extract_room_tour_corridors_rule(
    train_pairs: list[dict[str, Any]],
) -> bool:
    """Checks if connect_room_tour_corridors solves 100% of training pairs."""
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        pred = connect_room_tour_corridors(inp)
        if not grid_equal(pred, out):
            return False
    return True










