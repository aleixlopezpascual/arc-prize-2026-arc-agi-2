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







