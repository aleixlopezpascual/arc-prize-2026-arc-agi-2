from __future__ import annotations

import pytest

from src.dsl.primitives import (
    Object,
    color_counts,
    complete_symmetry,
    crop_to_bounding_box,
    crop_to_content,
    detect_symmetry,
    find_connected_components,
    find_multicolor_objects,
    flood_fill,
    gravity,
    least_frequent_color,
    mirror_append,
    most_frequent_color,
    pad_grid,
    reflect_horizontal,
    reflect_vertical,
    remap_colors,
    replace_color,
    rotate_90,
    rotate_180,
    rotate_270,
    scale_grid,
    tile_grid,
    transpose,
)


class TestObjectExtraction:
    def test_find_connected_components_4_conn(self):
        grid = [
            [1, 1, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 3, 3],
        ]
        objs = find_connected_components(grid, background=0, connectivity=4)
        assert len(objs) == 3

        # Color 1 object: 3 cells at (0,0), (0,1), (1,0)
        c1 = [o for o in objs if o.color == 1][0]
        assert c1.size == 3
        assert c1.bbox == (0, 0, 1, 1)
        assert c1.height == 2
        assert c1.width == 2
        assert c1.coords == {(0, 0), (0, 1), (1, 0)}

        # Color 2 object: 2 cells at (0,3), (1,3)
        c2 = [o for o in objs if o.color == 2][0]
        assert c2.size == 2
        assert c2.bbox == (0, 3, 1, 3)
        assert c2.height == 2
        assert c2.width == 1

        # Color 3 object: 2 cells at (2,2), (2,3)
        c3 = [o for o in objs if o.color == 3][0]
        assert c3.size == 2
        assert c3.bbox == (2, 2, 2, 3)
        assert c3.height == 1
        assert c3.width == 2

    def test_connectivity_8_vs_4(self):
        grid = [
            [1, 0],
            [0, 1],
        ]
        objs_4 = find_connected_components(grid, background=0, connectivity=4)
        assert len(objs_4) == 2

        objs_8 = find_connected_components(grid, background=0, connectivity=8)
        assert len(objs_8) == 1
        assert objs_8[0].size == 2
        assert objs_8[0].bbox == (0, 0, 1, 1)

    def test_multicolor_objects(self):
        grid = [
            [1, 2, 0],
            [0, 0, 3],
        ]
        objs = find_multicolor_objects(grid, background=0, connectivity=4)
        assert len(objs) == 2
        multi = [o for o in objs if o.size == 2][0]
        assert multi.colors == {1, 2}
        assert multi.bbox == (0, 0, 0, 1)


class TestSpatialTransforms:
    def test_crop_to_bounding_box(self):
        grid = [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0],
        ]
        cropped = crop_to_bounding_box(grid, (1, 1, 2, 2))
        assert cropped == [
            [1, 2],
            [3, 4],
        ]

    def test_crop_to_content(self):
        grid = [
            [0, 0, 0, 0],
            [0, 5, 0, 0],
            [0, 0, 5, 0],
            [0, 0, 0, 0],
        ]
        cropped = crop_to_content(grid, background=0)
        assert cropped == [
            [5, 0],
            [0, 5],
        ]

    def test_pad_grid(self):
        grid = [
            [1, 2],
            [3, 4],
        ]
        padded = pad_grid(grid, target_h=4, target_w=4, pad_val=0)
        assert len(padded) == 4
        assert len(padded[0]) == 4
        assert padded[0][:2] == [1, 2]
        assert padded[1][:2] == [3, 4]
        assert padded[2] == [0, 0, 0, 0]
        assert padded[3] == [0, 0, 0, 0]

    def test_scale_grid(self):
        grid = [
            [1, 2],
            [3, 4],
        ]
        scaled = scale_grid(grid, factor=2)
        assert scaled == [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ]

    def test_gravity_down_simple(self):
        grid = [
            [1, 0, 2],
            [0, 0, 0],
            [0, 3, 0],
        ]
        fallen = gravity(grid, direction="down", background=0)
        assert fallen == [
            [0, 0, 0],
            [0, 0, 0],
            [1, 3, 2],
        ]

    def test_gravity_down_with_obstacle(self):
        # Color 8 is obstacle
        grid = [
            [1, 0],
            [8, 0],
            [0, 0],
        ]
        fallen = gravity(grid, direction="down", obstacle_colors={8}, background=0)
        # 1 stays above 8
        assert fallen == [
            [1, 0],
            [8, 0],
            [0, 0],
        ]

    def test_gravity_left(self):
        grid = [
            [0, 1, 0],
            [0, 0, 2],
        ]
        fallen = gravity(grid, direction="left", background=0)
        assert fallen == [
            [1, 0, 0],
            [2, 0, 0],
        ]

    def test_flood_fill(self):
        grid = [
            [1, 1, 2],
            [1, 0, 2],
            [0, 2, 2],
        ]
        filled = flood_fill(grid, r=0, c=0, new_color=9)
        assert filled == [
            [9, 9, 2],
            [9, 0, 2],
            [0, 2, 2],
        ]


class TestGeometricTransforms:
    def test_rotations(self):
        grid = [
            [1, 2],
            [3, 4],
        ]
        assert rotate_90(grid) == [[3, 1], [4, 2]]
        assert rotate_180(grid) == [[4, 3], [2, 1]]
        assert rotate_270(grid) == [[2, 4], [1, 3]]

    def test_reflections_and_transpose(self):
        grid = [
            [1, 2],
            [3, 4],
        ]
        assert reflect_horizontal(grid) == [[3, 4], [1, 2]]
        assert reflect_vertical(grid) == [[2, 1], [4, 3]]
        assert transpose(grid) == [[1, 3], [2, 4]]


class TestPatternAndSymmetry:
    def test_detect_symmetry(self):
        sym_v = [
            [1, 2, 1],
            [3, 4, 3],
        ]
        res = detect_symmetry(sym_v)
        assert res["vertical"] is True
        assert res["horizontal"] is False

    def test_mirror_append(self):
        grid = [
            [1, 2],
            [3, 4],
        ]
        appended_v = mirror_append(grid, axis="vertical")
        assert appended_v == [
            [1, 2, 2, 1],
            [3, 4, 4, 3],
        ]

        appended_h = mirror_append(grid, axis="horizontal")
        assert appended_h == [
            [1, 2],
            [3, 4],
            [3, 4],
            [1, 2],
        ]

    def test_complete_symmetry(self):
        half = [
            [1, 0],
            [2, 0],
        ]
        completed = complete_symmetry(half, axis="vertical", source="left")
        assert completed == [
            [1, 1],
            [2, 2],
        ]

    def test_tile_grid(self):
        grid = [
            [1, 2],
            [3, 4],
        ]
        tiled = tile_grid(grid, repeats_h=2, repeats_w=2)
        assert tiled == [
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [1, 2, 1, 2],
            [3, 4, 3, 4],
        ]


class TestColorOperators:
    def test_replace_and_remap(self):
        grid = [
            [1, 2],
            [2, 3],
        ]
        assert replace_color(grid, 2, 5) == [
            [1, 5],
            [5, 3],
        ]
        assert remap_colors(grid, {1: 7, 2: 8, 3: 9}) == [
            [7, 8],
            [8, 9],
        ]

    def test_color_counts_and_frequencies(self):
        grid = [
            [0, 0, 1],
            [0, 1, 2],
            [0, 1, 1],
        ]
        counts = color_counts(grid, ignore_background=False)
        assert counts == {0: 4, 1: 4, 2: 1}

        # Ignore background 0
        assert most_frequent_color(grid, ignore_background=True, background=0) == 1
        assert least_frequent_color(grid, ignore_background=True, background=0) == 2


class TestAdvancedObjectPrimitives:
    def test_fill_enclosed_regions(self):
        from src.dsl import fill_enclosed_regions

        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
        filled = fill_enclosed_regions(grid, fill_color=2, background=0)
        assert filled[2][2] == 2
        assert filled[0][0] == 0
        assert filled[1][1] == 1

    def test_crop_largest_and_smallest_object(self):
        from src.dsl import crop_largest_object, crop_smallest_object

        grid = [
            [1, 1, 0, 0],
            [1, 1, 0, 2],
            [0, 0, 0, 0],
        ]
        largest = crop_largest_object(grid, background=0)
        assert largest == [[1, 1], [1, 1]]

        smallest = crop_smallest_object(grid, background=0)
        assert smallest == [[2]]

    def test_remove_small_objects(self):
        from src.dsl import remove_small_objects

        grid = [
            [1, 1, 0, 3],
            [1, 1, 0, 0],
        ]
        # Color 3 has size 1, color 1 has size 4
        cleaned = remove_small_objects(grid, max_size=1, background=0, replacement=0)
        assert cleaned == [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]


class TestDSLSearchAndSolver:
    def test_search_and_solve_rotation(self):
        from src.dsl import DSLSolver, search_dsl_programs, rotate_90

        # Construct demonstration task: rotate 90
        train_pairs = [
            {"input": [[1, 2], [3, 4]], "output": rotate_90([[1, 2], [3, 4]])},
            {"input": [[5, 6], [7, 8]], "output": rotate_90([[5, 6], [7, 8]])},
        ]
        test_pairs = [{"input": [[9, 1], [2, 3]]}]
        task = {"train": train_pairs, "test": test_pairs}

        programs = search_dsl_programs(train_pairs)
        assert len(programs) >= 1
        assert any(p[0].name == "rot90" for p in programs)

        solver = DSLSolver()
        preds = solver.solve_task(task)
        assert len(preds) == 1
        assert preds[0]["attempt_1"] == rotate_90([[9, 1], [2, 3]])

    def test_search_and_solve_color_remap(self):
        from src.dsl import DSLSolver, search_dsl_programs

        train_pairs = [
            {"input": [[1, 2], [2, 3]], "output": [[7, 8], [8, 9]]},
            {"input": [[2, 1], [3, 2]], "output": [[8, 7], [9, 8]]},
        ]
        test_pairs = [{"input": [[1, 1], [3, 3]]}]
        task = {"train": train_pairs, "test": test_pairs}

        programs = search_dsl_programs(train_pairs)
        assert len(programs) >= 1

        solver = DSLSolver()
        preds = solver.solve_task(task)
        assert len(preds) == 1
        assert preds[0]["attempt_1"] == [[7, 7], [9, 9]]

    def test_solver_fallback_and_multi_test(self):
        from src.dsl import DSLSolver

        # Unsolvable by simple primitives -> fallback
        task = {
            "train": [
                {"input": [[1, 2]], "output": [[9, 9, 9]]},
            ],
            "test": [
                {"input": [[3, 4]]},
                {"input": [[5, 6]]},
            ],
        }
        solver = DSLSolver()
        preds = solver.solve_task(task)
        assert len(preds) == 2
        for pred in preds:
            assert "attempt_1" in pred and "attempt_2" in pred
            assert len(pred["attempt_1"]) > 0
            assert len(pred["attempt_2"]) > 0

    def test_connect_matching_dots(self):
        from src.dsl import connect_matching_dots

        grid = [
            [1, 0, 0, 1],
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0],
        ]
        connected = connect_matching_dots(grid, background=0)
        # Row 0 connected with 1s
        assert connected[0] == [1, 1, 1, 1]
        # Col 0 connected with 2s
        assert [connected[r][0] for r in range(4)] == [1, 0, 2, 2]

    def test_crop_hollow_rectangle_interior(self):
        from src.dsl import crop_hollow_rectangle_interior

        grid = [
            [0, 0, 0, 0, 0],
            [0, 3, 3, 3, 0],
            [0, 3, 5, 3, 0],
            [0, 3, 3, 3, 0],
            [0, 0, 0, 0, 0],
        ]
        inner = crop_hollow_rectangle_interior(grid, background=0)
        assert inner == [[5]]

    def test_search_and_solve_connect_dots(self):
        from src.dsl import DSLSolver, search_dsl_programs, connect_matching_dots

        train_pairs = [
            {"input": [[1, 0, 1]], "output": [[1, 1, 1]]},
            {"input": [[2, 0, 2]], "output": [[2, 2, 2]]},
        ]
        test_pairs = [{"input": [[3, 0, 3]]}]
        task = {"train": train_pairs, "test": test_pairs}

        programs = search_dsl_programs(train_pairs)
        assert len(programs) >= 1

        solver = DSLSolver()
        preds = solver.solve_task(task)
        assert preds[0]["attempt_1"] == [[3, 3, 3]]


