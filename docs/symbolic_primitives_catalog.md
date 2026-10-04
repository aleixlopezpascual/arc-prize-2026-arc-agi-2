# Catalog of Symbolic ARC Primitives (`src/dsl/`)

This catalog details the complete set of verified symbolic perception and transformation primitives implemented in `src/dsl/primitives.py`, `src/dsl/cellular_automata.py`, and `src/dsl/overlay.py`.

---

## 1. 2D Periodic Motif & Room Repair (`repair_periodic_subgrids`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved **Task `135a2760`** in the official ARC-AGI-2 Evaluation set with 100% exact match.
- **Concept:** ARC tasks frequently partition a grid into independent "rooms" or "subgrids" separated by solid divider lines. Inside each room, an internal 2D block (such as an alternating diagonal bouncing wave of color or periodic geometric tile) repeats across rows or columns. Corruptions or noise in the input grid violate the dominant repeating frequency.
- **Algorithm:**
  1. Detects divider rows and columns where $\ge W-1$ or $\ge H-1$ pixels share a border color.
  2. Bounds each room $(r_1, r_2, c_1, c_2)$.
  3. Extracts the interior room block (inside borders).
  4. For periods $P \in \{2 \dots 8\}$, constructs candidate repeating blocks via mode voting:
     $$\text{Block}[r, c] = \text{mode}\left(\{ \text{Room}[r + k \cdot P, c] \}_{k}\right)$$
  5. Selects the period $P$ with highest cell agreement ($\ge 75\%$).
  6. Reconstructs and repairs damaged cells to match the dominant periodic motif.

---

## 2. 3x3 Local Cellular Automata (`extract_3x3_local_rules` & `apply_3x3_local_rules`)
- **File:** `src/dsl/cellular_automata.py`
- **Benchmark Lift:** Solved **7 training tasks** (`3618c87e`, `50cb2852`, `6f8cd79b`, `84f2aca1`, `a699fb00`, `bb43febb`, `dc1df850`).
- **Concept:** For isometric tasks ($H_{\text{out}} = H_{\text{in}}, W_{\text{out}} = W_{\text{in}}$), cell transitions are frequently governed by local, shift-invariant neighborhood rules:
  $$Y[r, c] = f\left( X[r-1:r+2, \, c-1:c+2] \right)$$
- **Algorithm:**
  1. Pads demonstrations with $-1$ representing the out-of-bounds boundary.
  2. Extracts all 9-tuples representing $3 \times 3$ patches mapped to target cell values $Y[r, c]$.
  3. Checks for contradiction: if any patch maps to multiple distinct colors, the task is non-local and the rule extractor aborts (returns `None`).
  4. For test inputs, looks up each $3 \times 3$ patch in the learned rule dictionary.
  5. Unseen patches gracefully fall back to the input cell value (identity fallback), preserving background and unperturbed regions.

---

## 3. Split & Logical Overlays (`split_and_overlay`)
- **File:** `src/dsl/overlay.py`
- **Benchmark Lift:** Solved tasks **`e98196ab`** and **`e3497940`**.
- **Concept:** Inputs contain composite symmetric halves or sub-regions separated by a midline or divider. The output is the logical or bitwise combination of the two halves.
- **Supported Operations:**
  - `vsplit_div_or`, `vsplit_div_and`, `vsplit_div_xor`: Vertical split across a divider column.
  - `vsplit_or`, `vsplit_and`, `vsplit_xor`: Vertical split across even width midline.
  - `vsplit_mirror_or`, `vsplit_mirror_and`, `vsplit_mirror_xor`: Vertical split where right half is mirrored horizontally before overlaying.
  - `hsplit_div_or`, `hsplit_div_and`, `hsplit_div_xor`: Horizontal split across a divider row.
  - `hsplit_or`, `hsplit_and`, `hsplit_xor`: Horizontal split across even height midline.
  - `hsplit_mirror_or`, `hsplit_mirror_and`, `hsplit_mirror_xor`: Horizontal split where bottom half is mirrored vertically before overlaying.

---

## 4. Odd-One-Out Block Selector (`select_odd_one_out_block`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved task **`a87f7484`** 100% exactly on both train and test.
- **Concept:** An input presents a sequence of candidate $3 \times 3$ blocks (stacked vertically, horizontally, or in a grid). The majority of blocks share a common structural symmetry (e.g. four symmetric crosses and one asymmetric block, or vice-versa).
- **Algorithm:**
  1. Partitions the input grid into non-overlapping $(win\_h, win\_w)$ blocks.
  2. Evaluates the horizontal and vertical reflection symmetry signature for each block:
     $$\text{sig}(B) = \left( B == \text{flipud}(B) \right) \land \left( B == \text{fliplr}(B) \right)$$
  3. Counts signature frequencies: $\text{counts} = \text{Counter}(\{ \text{sig}(B_i) \})$.
  4. Identifies the unique signature where $\text{count} == 1$ (the odd one out).
  5. Emits the corresponding block as the solution.

---

## 5. Marker-Relative Window Extraction (`extract_window_relative_to_marker`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved task **`48d8fb45`** 100% exactly on both train and test.
- **Concept:** A unique single-pixel "marker" dot of rare color pinpoints the region of interest in a larger grid.
- **Supported Spatial Relations:**
  - `"below"`: Extracts $(win\_h, win\_w)$ window centered horizontally directly below the marker $(r_m + 1 : r_m + 1 + win\_h, \, c_m - win\_w//2 : \dots)$.
  - `"above"`: Extracts window directly above the marker.
  - `"center"`: Extracts window centered on the marker.
  - `"right"`, `"left"`: Extracts window to the cardinal sides of the marker.

---

## 6. Rigid-Body Object Gravity (`rigid_object_gravity`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved task **`1e0a9b12`** 100% exactly.
- **Concept:** Implements Spelke physical object priors (cohesion, continuity, solidity). Unlike pixel-by-pixel liquid gravity, rigid-body gravity treats each connected component as an indivisible solid shape.
- **Algorithm:**
  1. Segments all non-background connected components.
  2. Sorts components by their position along the gravity vector (e.g. lowest objects drop first for downward gravity).
  3. Moves each entire multi-pixel object step-by-step along the vector until any of its pixels collides with an obstacle or the grid boundary.
  4. Locks the object into the destination canvas before dropping subsequent objects.

---

## 7. Dot Line Connections (`connect_matching_dots`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved **3 training tasks** (`1f876c06`, `22168020`, `22eb0ac0`).
- **Concept:** Identifies pairs of identical colored single-pixel endpoints and draws straight connecting lines between them.
- **Behavior:**
  - Evaluates pairs sharing the same row, column, or diagonal.
  - Validates line-of-sight: only draws the connecting line if the path between endpoints consists exclusively of background or the endpoint color (no intersecting third-party obstacles).

---

## 8. Enclosed Region Flood Fill (`fill_enclosed_regions`)
- **File:** `src/dsl/primitives.py`
- **Concept:** Topological boundary filling. Identifies background regions that cannot reach the exterior boundary of the grid (i.e. surrounded by walls of non-background color) and fills them with a target color.
- **Algorithm:**
  1. Seeds a BFS/flood-fill queue from all boundary background cells.
  2. Marks all reachable background cells as "exterior".
  3. Any background cell that remains unvisited is enclosed, and is colored with `fill_color`.

---

## 9. Hollow Rectangle Interior Extraction (`crop_hollow_rectangle_interior`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved task **`1c786137`**.
- **Concept:** Detects rectangular frames of uniform color $C$ with perimeter $\ge 3 \times 3$, verifies that the entire perimeter is color $C$, and crops out the interior payload $(r_{\min}+1 : r_{\max}, \, c_{\min}+1 : c_{\max})$.

---

## 10. Directional Raycasting & Shadow Projection (`raycast_directional`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved task **`d037b0a7`** 100% exactly.
- **Concept:** Non-background pixels project straight directional rays or cast shadows across background cells along cardinal directions (`down`, `up`, `left`, `right`, `vertical`, `horizontal`, `cross`).

---

## 11. Diagonal Billiard Bouncing Rays (`raycast_diagonal_bouncing`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved task **`623ea044`** 100% exactly.
- **Concept:** A colored pixel projects diagonal laser rays that bounce off grid boundaries like a billiard ball ($dr \to -dr, dc \to -dc$), generating reflective trajectories across the canvas.

---

## 12. Enclosed Pixel Color Summary (`extract_enclosed_pixel_color`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved task **`d9fac9be`** 100% exactly.
- **Concept:** Locates a single colored pixel enclosed inside a hollow $3 \times 3$ frame and emits its color as a $1 \times 1$ grid `[[C]]`.

---

## 13. Bilateral Symmetry Binary Classifier (`classify_by_symmetry`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved task **`44f52bb0`** 100% exactly.
- **Concept:** Recognizes visual mirror symmetry (vertical or horizontal bilateral reflection). Maps symmetric inputs to class color $C_{\text{sym}}$ (e.g. blue) and asymmetric inputs to $C_{\text{asym}}$ (e.g. orange).

---

## 14. Binary Shape-to-Color Class Classifier (`classify_binary_shape`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved task **`27a28665`** 100% exactly.
- **Concept:** Shape symbol encoder. Converts input to an invariant binary mask and classifies each unique geometrical glyph into a class label.

---

## 15. Component Coloring by Metric Property (`color_objects_by_property`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved **4 training tasks** (`1d61978c`, `6e82a1ae`, `ae58858e`, `25d8a9c8`).
- **Concept:** Recolors connected components based on their geometric metrics (`size`/area, `height`, or `width`).

---

## 16. Kronecker Fractal Self-Expansion (`kronecker_fractal_expansion`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved **2 training tasks** (`007bbfb7`, `5b6cbef5`).
- **Concept:** Computes self-similarity fractal zoom ($A \otimes A$). Each non-background pixel in the $N \times N$ template grid is replaced by a complete copy of the template, expanding into an $N^2 \times N^2$ fractal matrix.

---

## 17. 2x2 Kaleidoscope Mirror Tiling (`tile_kaleidoscope_2x2`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved **3 training tasks** (`3af2c5a8`, `62c24649`, `67e8384a`).
- **Concept:** Assembles a 4-quadrant kaleidoscope matrix of size $(2H, 2W)$ where Quadrant II is $A$, Quadrant I is $\text{fliplr}(A)$, Quadrant III is $\text{flipud}(A)$, and Quadrant IV is $\text{flipud}(\text{fliplr}(A))$.

---

## 18. Horizontal & Vertical Mirror Tiling (`tile_mirror_horizontal`, `tile_mirror_vertical`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved **4 training tasks** (`6d0aefbc`, `c9e6f938`, `6fa7a44f`, `8be77c9e`).
- **Concept:** Doubles canvas along the horizontal axis $[A, \text{fliplr}(A)]$ or vertical axis $\text{vstack}([A, \text{flipud}(A)])$ with reflection symmetry.

---

## 19. Target Shape Periodic Tiling (`tile_to_target_shape`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved task **`d10ecb37`**.
- **Concept:** Periodically tiles an input pattern along both axes until filling a fixed target bounding dimension $(H_{\text{target}}, W_{\text{target}})$.

---

## 20. Topological Hole Counting & Genus Classification (`count_holes_in_object`, `color_objects_by_holes`, `extract_object_by_hole_count`)
- **File:** `src/dsl/primitives.py`
- **Benchmark Lift:** Solved **5 training tasks** (`0a2355a6`, `37d3e8b2`, `810b9b61`, `1cf80156`, `c909285e`).
- **Concept:** Measures topological Euler genus (the count of enclosed hollow cavities within a connected component).
  - 0 holes: Solid blob (solves `1cf80156`).
  - 1 hole: Hollow frame or donut (solves `c909285e`).
  - Multiple holes: Maps components to discrete class colors based on cavity count (solves `0a2355a6`, `37d3e8b2`, `810b9b61`).

---

## 21. Morphological & Geometric Transforms
- **File:** `src/dsl/primitives.py`
- **`crop_largest_object` & `crop_smallest_object`:** Connected-component bounding box extraction by size extrema.
- **`remove_small_objects`:** Filters connected components of size $\le \text{threshold}$ (noise suppression).
- **`rotate_90`, `rotate_180`, `rotate_270`:** Euclidean rotations.
- **`reflect_horizontal`, `reflect_vertical`, `transpose`:** Dihedral reflections.
- **`scale_grid` & `tile_grid`:** Integer zoom factor expansion and periodic grid repetition.
- **`dynamic_remap_colors`:** Computes global bijective or surjective color mappings between input and output demonstration pairs.
