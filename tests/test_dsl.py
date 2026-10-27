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

    def test_extract_and_apply_3x3_local_rules(self):
        from src.dsl import extract_3x3_local_rules, apply_3x3_local_rules

        # Invert color 1 to 2 whenever surrounded by 0
        train_pairs = [
            {"input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]], "output": [[0, 0, 0], [0, 2, 0], [0, 0, 0]]},
            {"input": [[0, 0, 0], [0, 0, 0], [0, 0, 0]], "output": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]},
        ]
        rules = extract_3x3_local_rules(train_pairs)
        assert rules is not None

        test_in = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        pred = apply_3x3_local_rules(test_in, rules)
        assert pred == [[0, 0, 0], [0, 2, 0], [0, 0, 0]]

    def test_split_and_overlay(self):
        from src.dsl import split_and_overlay

        # 2x5 grid with vertical divider at col 2
        grid = [
            [1, 0, 5, 0, 2],
            [0, 3, 5, 4, 0],
        ]
        or_result = split_and_overlay(grid, op="vsplit_div_or", background=0)
        assert or_result == [
            [1, 2],
            [4, 3],
        ]

    def test_select_odd_one_out_block(self):
        from src.dsl import select_odd_one_out_block

        # 3x6 grid: left 3x3 is symmetric, right 3x3 is asymmetric
        # And a third 3x3 symmetric block in a 3x9 grid
        grid = [
            [1, 0, 1,   2, 0, 0,   3, 0, 3],
            [0, 1, 0,   0, 2, 0,   0, 3, 0],
            [1, 0, 1,   0, 0, 2,   3, 0, 3],
        ]
        odd = select_odd_one_out_block(grid, win_h=3, win_w=3)
        assert odd == [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
        ]

    def test_extract_window_relative_to_marker(self):
        from src.dsl import extract_window_relative_to_marker

        # 5x5 grid with unique marker color 5 at (1, 1)
        grid = [
            [0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0],
            [1, 2, 3, 0, 0],
            [4, 0, 6, 0, 0],
            [7, 8, 9, 0, 0],
        ]
        win = extract_window_relative_to_marker(grid, marker_color=5, relation="below", win_h=3, win_w=3)
        assert win == [
            [1, 2, 3],
            [4, 0, 6],
            [7, 8, 9],
        ]

    def test_rigid_object_gravity(self):
        from src.dsl import rigid_object_gravity

        # 4x4 grid: 2x2 square floating at rows 0..1, col 1..2
        grid = [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        dropped = rigid_object_gravity(grid, direction="down", background=0)
        assert dropped == [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ]

    def test_raycast_directional(self):
        from src.dsl import raycast_directional

        # 3x3 grid with single pixel 2 at (0, 1)
        grid = [
            [0, 2, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        down = raycast_directional(grid, direction="down", background=0)
        assert down == [
            [0, 2, 0],
            [0, 2, 0],
            [0, 2, 0],
        ]

    def test_raycast_diagonal_bouncing(self):
        from src.dsl import raycast_diagonal_bouncing

        # 3x3 grid with single pixel at (0, 0)
        grid = [
            [7, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        bounced = raycast_diagonal_bouncing(grid, background=0)
        assert bounced[1][1] == 7
        assert bounced[2][2] == 7

    def test_color_objects_by_property(self):
        from src.dsl import color_objects_by_property, extract_object_property_color_rule

        train_pairs = [
            # Object of size 1 becomes 2, object of size 2 becomes 3
            {"input": [[1, 0, 1], [0, 0, 1]], "output": [[2, 0, 3], [0, 0, 3]]},
            {"input": [[1, 0, 0], [0, 0, 1], [0, 0, 1]], "output": [[2, 0, 0], [0, 0, 3], [0, 0, 3]]},
        ]
        rule = extract_object_property_color_rule(train_pairs, property_name="size", background=0)
        assert rule == {1: 2, 2: 3}

        test_in = [[0, 1, 0], [0, 0, 0], [1, 1, 0]]
        recolored = color_objects_by_property(test_in, rule, property_name="size", background=0)
        assert recolored == [[0, 2, 0], [0, 0, 0], [3, 3, 0]]

    def test_extract_enclosed_pixel_color(self):
        from src.dsl import extract_enclosed_pixel_color

        grid = [
            [0, 0, 0, 0],
            [0, 2, 2, 2],
            [0, 2, 5, 2],
            [0, 2, 2, 2],
        ]
        out = extract_enclosed_pixel_color(grid, background=0)
        assert out == [[5]]

    def test_classify_by_symmetry(self):
        from src.dsl import classify_by_symmetry

        sym_grid = [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ]
        asym_grid = [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]
        assert classify_by_symmetry(sym_grid, sym_color=1, asym_color=7) == [[1]]
        assert classify_by_symmetry(asym_grid, sym_color=1, asym_color=7) == [[7]]

    def test_extract_and_classify_binary_shape(self):
        from src.dsl import extract_binary_shape_classifier, classify_binary_shape

        train_pairs = [
            {"input": [[1, 1], [0, 1]], "output": [[4]]},
            {"input": [[2, 2], [0, 2]], "output": [[4]]},
            {"input": [[3, 0], [0, 3]], "output": [[9]]},
        ]
        s_map = extract_binary_shape_classifier(train_pairs)
        assert s_map is not None

        test_in = [[8, 8], [0, 8]]
        pred = classify_binary_shape(test_in, s_map)
        assert pred == [[4]]

    def test_kronecker_fractal_expansion(self):
        from src.dsl import kronecker_fractal_expansion

        grid = [
            [1, 0],
            [0, 1],
        ]
        expanded = kronecker_fractal_expansion(grid, background=0)
        assert len(expanded) == 4 and len(expanded[0]) == 4
        # Top-left block is [[1, 0], [0, 1]]
        assert expanded[0][:2] == [1, 0]
        assert expanded[1][:2] == [0, 1]
        # Top-right block is 0s
        assert expanded[0][2:] == [0, 0]
        assert expanded[1][2:] == [0, 0]

    def test_tile_kaleidoscope_2x2(self):
        from src.dsl import tile_kaleidoscope_2x2

        grid = [
            [1, 2],
            [3, 4],
        ]
        k = tile_kaleidoscope_2x2(grid)
        assert len(k) == 4 and len(k[0]) == 4
        # Row 0: [1, 2, 2, 1]
        assert k[0] == [1, 2, 2, 1]
        # Row 3: [1, 2, 2, 1]
        assert k[3] == [1, 2, 2, 1]

    def test_tile_mirror_h_and_v(self):
        from src.dsl import tile_mirror_horizontal, tile_mirror_vertical

        grid = [
            [1, 2],
            [3, 4],
        ]
        th = tile_mirror_horizontal(grid)
        assert th == [
            [1, 2, 2, 1],
            [3, 4, 4, 3],
        ]
        tv = tile_mirror_vertical(grid)
        assert tv == [
            [1, 2],
            [3, 4],
            [3, 4],
            [1, 2],
        ]

    def test_tile_to_target_shape(self):
        from src.dsl import tile_to_target_shape

        grid = [[1, 2], [3, 4]]
        tiled = tile_to_target_shape(grid, target_h=3, target_w=5)
        assert len(tiled) == 3 and len(tiled[0]) == 5
        assert tiled[0] == [1, 2, 1, 2, 1]
        assert tiled[1] == [3, 4, 3, 4, 3]
        assert tiled[2] == [1, 2, 1, 2, 1]

    def test_color_objects_by_holes(self):
        from src.dsl import color_objects_by_holes, extract_holes_to_color_rule

        # Object 1: solid 2x2 square (0 holes)
        # Object 2: 3x3 hollow square (1 hole)
        train_pairs = [
            {
                "input": [
                    [1, 1, 0, 2, 2, 2],
                    [1, 1, 0, 2, 0, 2],
                    [0, 0, 0, 2, 2, 2],
                ],
                "output": [
                    [3, 3, 0, 4, 4, 4],
                    [3, 3, 0, 4, 0, 4],
                    [0, 0, 0, 4, 4, 4],
                ],
            },
            {
                "input": [
                    [1, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                ],
                "output": [
                    [3, 3, 0, 0, 0, 0],
                    [3, 3, 0, 0, 0, 0],
                ],
            },
        ]
        h_rule = extract_holes_to_color_rule(train_pairs, background=0)
        assert h_rule == {0: 3, 1: 4}

        test_in = [
            [5, 5, 5],
            [5, 0, 5],
            [5, 5, 5],
        ]
        recolored = color_objects_by_holes(test_in, h_rule, background=0)
        assert recolored == [
            [4, 4, 4],
            [4, 0, 4],
            [4, 4, 4],
        ]

    def test_extract_object_by_hole_count(self):
        from src.dsl import extract_object_by_hole_count

        grid = [
            [1, 1, 0, 2, 2, 2],
            [1, 1, 0, 2, 0, 2],
            [0, 0, 0, 2, 2, 2],
        ]
        # Extract object with 1 hole (the 3x3 hollow square of 2s)
        extracted = extract_object_by_hole_count(grid, target_holes=1, background=0)
        assert extracted == [
            [2, 2, 2],
            [2, 0, 2],
            [2, 2, 2],
        ]

    def test_d4_tta_search(self):
        from src.dsl import search_dsl_programs

        # Task that requires horizontal reflection, but oriented vertically
        train_pairs = [
            {"input": [[1, 2], [0, 0]], "output": [[1, 2], [1, 2]]},
            {"input": [[3, 4], [0, 0]], "output": [[3, 4], [3, 4]]},
        ]
        progs = search_dsl_programs(train_pairs, max_level=3)
        assert len(progs) > 0
        best_prog, score = progs[0]
        assert score == 1.0


class TestVersion9Primitives:
    def test_recolor_motif(self):
        from src.dsl import recolor_motif

        grid = [
            [2, 2, 2, 2, 2],
            [2, 0, 4, 0, 2],
            [2, 4, 4, 4, 2],
            [2, 0, 4, 0, 2],
            [2, 2, 2, 2, 2],
        ]
        cross_mask = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
        out = recolor_motif(grid, motif=cross_mask, src_color=4, dst_color=8)
        assert out[2][2] == 8
        assert out[1][2] == 8
        assert out[3][2] == 8
        assert out[2][1] == 8
        assert out[2][3] == 8
        assert out[0][0] == 2

    def test_fill_enclosed_regions_by_area(self):
        from src.dsl import fill_enclosed_regions_by_area

        grid = [
            [2, 2, 2, 0, 2, 2, 2, 2],
            [2, 0, 2, 0, 2, 0, 0, 2],
            [2, 2, 2, 0, 2, 0, 0, 2],
            [0, 0, 0, 0, 2, 2, 2, 2],
        ]
        area_to_color = {1: 3, 4: 8}
        out = fill_enclosed_regions_by_area(grid, area_to_color=area_to_color, background=0)
        assert out[1][1] == 3
        assert out[1][5] == 8
        assert out[2][5] == 8
        assert out[0][3] == 0

    def test_color_objects_by_rank(self):
        from src.dsl import color_objects_by_rank

        grid = [
            [0, 5, 0, 5, 0],
            [0, 5, 0, 5, 0],
            [0, 0, 0, 5, 0],
        ]
        rank_to_color = {0: 1, 1: 2}
        out = color_objects_by_rank(grid, rank_to_color=rank_to_color, metric="height", reverse=True, background=0)
        assert out[0][3] == 1
        assert out[1][3] == 1
        assert out[2][3] == 1
        assert out[0][1] == 2
        assert out[1][1] == 2

    def test_extrapolate_periodic_grid(self):
        from src.dsl import extrapolate_periodic_grid

        grid = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 2],
        ]
        # (r+c) % 3: (1,2)->0 (val 1? wait, 1+2=3=0), (2,1)->0 (val 1), (2,2)->1 (val 2)
        # So mod 0 is 1, mod 1 is 2. Mod 2 could be filled if known, or let's provide mod_map
        mod_map = {0: 1, 1: 2, 2: 4}
        out = extrapolate_periodic_grid(grid, mode="diag_sum", period=3, mod_map=mod_map, background=0)
        assert out[0][0] == 1  # (0+0)%3 = 0 -> 1
        assert out[0][1] == 2  # (0+1)%3 = 1 -> 2
        assert out[0][2] == 4  # (0+2)%3 = 2 -> 4

    def test_extrapolate_point_progression(self):
        from src.dsl import extrapolate_point_progression

        grid = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ]
        out = extrapolate_point_progression(grid, point_color=1, new_color=2, background=0)
        assert out[3][3] == 2

    def test_search_recolor_motif(self):
        from src.dsl import search_dsl_programs

        cross = [
            [2, 4, 2, 2],
            [4, 4, 4, 2],
            [2, 4, 2, 2],
            [2, 2, 2, 4],  # isolated 4 that stays 4
        ]
        cross_out = [
            [2, 8, 2, 2],
            [8, 8, 8, 2],
            [2, 8, 2, 2],
            [2, 2, 2, 4],  # remains 4
        ]
        train_pairs = [
            {"input": cross, "output": cross_out},
        ]
        progs = search_dsl_programs(train_pairs, max_level=1)
        assert len(progs) > 0
        assert any("recolor_motif" in p[0].name for p in progs)

    def test_search_enclosed_area(self):
        from src.dsl import search_dsl_programs

        p1_in = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
        p1_out = [
            [1, 1, 1],
            [1, 3, 1],
            [1, 1, 1],
        ]
        train_pairs = [{"input": p1_in, "output": p1_out}]
        progs = search_dsl_programs(train_pairs, max_level=1)
        assert len(progs) > 0
        assert any("fill_enclosed_area" in p[0].name for p in progs)

    def test_search_rank_color(self):
        from src.dsl import search_dsl_programs

        p_in = [
            [0, 5, 0, 5, 0],
            [0, 5, 0, 5, 0],
            [0, 0, 0, 5, 0],
        ]
        p_out = [
            [0, 2, 0, 1, 0],
            [0, 2, 0, 1, 0],
            [0, 0, 0, 1, 0],
        ]
        train_pairs = [{"input": p_in, "output": p_out}]
        progs = search_dsl_programs(train_pairs, max_level=1)
        assert len(progs) > 0
        assert any("color_objects_by_rank" in p[0].name for p in progs)

    def test_search_periodic_grid(self):
        from src.dsl import search_dsl_programs

        p_in = [
            [0, 0, 1],
            [0, 1, 2],
            [1, 2, 4],
        ]
        p_out = [
            [2, 4, 1],
            [4, 1, 2],
            [1, 2, 4],
        ]
        train_pairs = [{"input": p_in, "output": p_out}]
        progs = search_dsl_programs(train_pairs, max_level=1)
        assert len(progs) > 0
        assert any("extrapolate_periodic" in p[0].name for p in progs)

    def test_search_point_progression(self):
        from src.dsl import search_dsl_programs

        p_in = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ]
        p_out = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 2, 0],
        ]
        train_pairs = [{"input": p_in, "output": p_out}]
        progs = search_dsl_programs(train_pairs, max_level=1)
        assert len(progs) > 0
        assert any("extrapolate_point_progression" in p[0].name for p in progs)


class TestVersion10Primitives:
    def test_trigger_release_gravity(self):
        from src.dsl import trigger_release_gravity

        grid = [
            [4, 0, 0, 0],
            [0, 2, 2, 0],
            [0, 4, 4, 0],
            [0, 0, 0, 0],
        ]
        out = trigger_release_gravity(grid, background=0)
        assert out[0][0] == 4
        assert out[1][1] == 0
        assert out[1][2] == 0
        assert out[3][1] == 2
        assert out[3][2] == 2

    def test_fill_puzzle_socket(self):
        from src.dsl import fill_puzzle_socket

        grid = [
            [3, 3, 3, 3, 3],
            [3, 2, 2, 3, 3],
            [3, 2, 2, 3, 3],
            [3, 3, 3, 0, 0],
            [3, 3, 3, 0, 0],
        ]
        out = fill_puzzle_socket(grid, piece_color=2, hole_color=0, fill_color=1)
        assert out[1][1] == 0
        assert out[1][2] == 0
        assert out[2][1] == 0
        assert out[2][2] == 0
        assert out[3][3] == 1
        assert out[3][4] == 1
        assert out[4][3] == 1
        assert out[4][4] == 1

    def test_slide_dots_to_wall(self):
        from src.dsl import slide_dots_to_wall

        grid = [
            [0, 7, 0, 4, 0, 7, 0],
        ]
        out = slide_dots_to_wall(grid, dot_color=4, direction="right", background=0)
        assert out[0][3] == 0
        assert out[0][4] == 4
        assert out[0][5] == 7

    def test_project_diagonal_rays(self):
        from src.dsl import project_diagonal_rays

        grid = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        out = project_diagonal_rays(grid, ray_color=1, background=0)
        assert out[1][1] == 1
        assert out[0][0] == 1
        assert out[1][3] == 1
        assert out[0][4] == 1

    def test_straighten_orthogonal_lines(self):
        from src.dsl import straighten_orthogonal_lines

        grid = [
            [0, 6, 0, 0],
            [0, 6, 0, 0],
            [0, 0, 6, 0],
            [0, 6, 0, 0],
            [0, 6, 0, 0],
        ]
        out = straighten_orthogonal_lines(grid, line_color=6, background=0)
        assert out[2][1] == 6
        assert out[2][2] == 0

    def test_search_puzzle_socket(self):
        from src.dsl import search_dsl_programs

        p_in = [
            [3, 3, 3, 3, 3],
            [3, 2, 2, 3, 3],
            [3, 2, 2, 3, 3],
            [3, 3, 3, 0, 0],
            [3, 3, 3, 0, 0],
        ]
        p_out = [
            [3, 3, 3, 3, 3],
            [3, 0, 0, 3, 3],
            [3, 0, 0, 3, 3],
            [3, 3, 3, 1, 1],
            [3, 3, 3, 1, 1],
        ]
        train_pairs = [{"input": p_in, "output": p_out}]
        progs = search_dsl_programs(train_pairs, max_level=1)
        assert len(progs) > 0
        assert any("fill_puzzle_socket" in p[0].name for p in progs)

    def test_search_slide_dots(self):
        from src.dsl import search_dsl_programs

        p_in = [
            [0, 7, 0, 4, 0, 7, 0],
        ]
        p_out = [
            [0, 7, 0, 0, 4, 7, 0],
        ]
        train_pairs = [{"input": p_in, "output": p_out}]
        progs = search_dsl_programs(train_pairs, max_level=1)
        assert len(progs) > 0
        assert any("slide_dots" in p[0].name for p in progs)

    def test_search_project_diagonal_rays(self):
        from src.dsl import search_dsl_programs

        p_in = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
        p_out = [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ]
        train_pairs = [{"input": p_in, "output": p_out}]
        progs = search_dsl_programs(train_pairs, max_level=1)
        assert len(progs) > 0
        assert any("project_diagonal_rays" in p[0].name for p in progs)

    def test_search_straighten_orthogonal_lines(self):
        from src.dsl import search_dsl_programs

        p_in = [
            [0, 6, 0, 0],
            [0, 6, 0, 0],
            [0, 0, 6, 0],
            [0, 6, 0, 0],
            [0, 6, 0, 0],
        ]
        p_out = [
            [0, 6, 0, 0],
            [0, 6, 0, 0],
            [0, 6, 0, 0],
            [0, 6, 0, 0],
            [0, 6, 0, 0],
        ]
        train_pairs = [{"input": p_in, "output": p_out}]
        progs = search_dsl_programs(train_pairs, max_level=1)
        assert len(progs) > 0
        assert any("straighten_orthogonal_lines" in p[0].name for p in progs)


class TestVersion13Primitives:
    def test_inpaint_masked_reflection(self):
        from src.dsl import inpaint_masked_reflection

        grid = [
            [2, 0, 8, 8],
            [2, 2, 8, 8],
        ]
        out = inpaint_masked_reflection(grid, obj_color=2, mask_color=8, axis="vertical")
        assert out[0] == [2, 0, 0, 2]
        assert out[1] == [2, 2, 2, 2]

    def test_split_by_gridlines(self):
        from src.dsl import detect_gridline_lattice, remove_gridlines

        grid = [
            [1, 1, 5, 2, 2],
            [1, 1, 5, 2, 2],
            [5, 5, 5, 5, 5],
            [3, 3, 5, 4, 4],
            [3, 3, 5, 4, 4],
        ]
        sep_color, row_seps, col_seps = detect_gridline_lattice(grid)
        assert sep_color == 5
        assert row_seps == [2]
        assert col_seps == [2]
        panels = remove_gridlines(grid, sep_color=5, row_seps=[2], col_seps=[2])
        assert len(panels) == 2
        assert len(panels[0]) == 2
        assert panels[0][0] == [[1, 1], [1, 1]]
        assert panels[0][1] == [[2, 2], [2, 2]]
        assert panels[1][0] == [[3, 3], [3, 3]]
        assert panels[1][1] == [[4, 4], [4, 4]]

    def test_search_inpaint_masked_reflection(self):
        from src.dsl import search_dsl_programs

        p_in = [
            [2, 0, 8, 8],
            [2, 2, 8, 8],
        ]
        p_out = [
            [2, 0, 0, 2],
            [2, 2, 2, 2],
        ]
        train_pairs = [{"input": p_in, "output": p_out}]
        progs = search_dsl_programs(train_pairs, max_level=1)
        assert len(progs) > 0
        assert any("inpaint_masked_reflection" in p[0].name for p in progs)

    def test_restore_bilateral_symmetry(self):
        from src.dsl import restore_bilateral_symmetry, extract_restore_bilateral_symmetry_rule

        # Asymmetric noise pixel at (0, 3)
        grid_in = [
            [1, 0, 1, 1],
            [1, 1, 1, 0],
            [0, 1, 0, 0],
        ]
        # Restored bilateral vertical symmetry
        grid_expected = [
            [1, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 1, 0, 0],
        ]
        res = restore_bilateral_symmetry(grid_in, background=0)
        assert res == grid_expected
        assert extract_restore_bilateral_symmetry_rule([{"input": grid_in, "output": grid_expected}], background=0)

    def test_fill_ranked_pillars_by_indicator(self):
        from src.dsl import fill_ranked_pillars_by_indicator, extract_ranked_pillars_rule

        # Indicator at (0, 0) height 1 of color 2 -> fill 1st tallest pillar of color 2
        # Pillar at col 2: rows 2..4 (height 3)
        # Pillar at col 4: rows 1..4 (height 4, tallest)
        grid_in = [
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 2],
            [0, 0, 2, 0, 5],
            [0, 0, 5, 0, 5],
            [0, 0, 2, 0, 2],
        ]
        grid_out = [
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 2],
            [0, 0, 2, 0, 2],
            [0, 0, 5, 0, 2],
            [0, 0, 2, 0, 2],
        ]
        res = fill_ranked_pillars_by_indicator(grid_in)
        assert res == grid_out
        assert extract_ranked_pillars_rule([{"input": grid_in, "output": grid_out}])

    def test_fill_cavities_from_legend(self):
        from src.dsl import fill_cavities_from_legend, extract_legend_cavity_rule

        # Legend at top border (0:2, 0:2): keys = [2, 3], vals = [4, 5]
        # Frame of color 2 with enclosed hole should be filled with color 4
        grid_in = [
            [2, 3, 0, 0, 0],
            [4, 5, 0, 0, 0],
            [0, 0, 2, 2, 2],
            [0, 0, 2, 0, 2],
            [0, 0, 2, 2, 2],
        ]
        grid_out = [
            [2, 3, 0, 0, 0],
            [4, 5, 0, 0, 0],
            [0, 0, 2, 2, 2],
            [0, 0, 2, 4, 2],
            [0, 0, 2, 2, 2],
        ]
        res = fill_cavities_from_legend(grid_in)
        assert res == grid_out
        assert extract_legend_cavity_rule([{"input": grid_in, "output": grid_out}])

    def test_standardize_parallel_lines(self):
        from src.dsl import standardize_parallel_lines, extract_standardize_parallel_lines_rule

        # Horizontal lines with reference line of color 9 (length 3)
        # Line of color 2 has length 1 -> extended to length 3
        # Line of color 3 has length 4 -> truncated to length 3
        grid_in = [
            [0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 9, 9, 9, 0],
            [0, 3, 3, 3, 3],
            [0, 0, 0, 0, 0],
        ]
        grid_out = [
            [0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0],
            [0, 9, 9, 9, 0],
            [0, 3, 3, 3, 0],
            [0, 0, 0, 0, 0],
        ]
        res = standardize_parallel_lines(grid_in)
        assert res == grid_out
        assert extract_standardize_parallel_lines_rule([{"input": grid_in, "output": grid_out}])

    def test_simulate_liquid_gravity(self):
        from src.dsl import simulate_liquid_gravity, extract_liquid_gravity_rule

        # Elevated blue liquid (color 1) at top rolls down and fills container slot at bottom
        grid_in = [
            [1, 1, 0, 0, 0],
            [9, 9, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 3, 0, 3],
            [0, 0, 3, 3, 3],
        ]
        grid_out = [
            [0, 0, 0, 0, 0],
            [9, 9, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 3, 1, 3],
            [0, 0, 3, 3, 3],
        ]
        res = simulate_liquid_gravity(grid_in, liquid_color=1)
        assert res == grid_out
        assert extract_liquid_gravity_rule([{"input": grid_in, "output": grid_out}], liquid_color=1)

    def test_slide_sokoban_block_in_channel(self):
        from src.dsl import slide_sokoban_block_in_channel, extract_sokoban_channel_rule

        # Wall: 3, Corridor: 1, Hole: 0, Block: 2
        # Hole of size 1x1 at (1, 1). Block at (3, 3) (reachable via corridor of 1s)
        grid_in = [
            [3, 3, 3, 3, 3],
            [3, 0, 1, 1, 3],
            [3, 3, 3, 1, 3],
            [3, 3, 3, 2, 3],
            [3, 3, 3, 3, 3],
        ]
        grid_out = [
            [3, 3, 3, 3, 3],
            [3, 1, 1, 1, 3],
            [3, 3, 3, 1, 3],
            [3, 3, 3, 0, 3],
            [3, 3, 3, 3, 3],
        ]
        res = slide_sokoban_block_in_channel(grid_in, corridor_color=1, hole_color=0)
        assert res == grid_out
        assert extract_sokoban_channel_rule([{"input": grid_in, "output": grid_out}]) == (1, 0)

    def test_compose_partitioned_rooms(self):
        from src.dsl import compose_partitioned_rooms, extract_partitioned_rooms_rule

        # 7x25 canvas with divider 5, background 4, line 6, room2=2, room3=3
        grid_in = [
            [5]*25,
            [5, 6,4,4,4,6, 5, 2,4,4,4,2, 5, 4,4,3,4,4, 5, 4,4,4,4,4, 5],
            [5, 4,4,4,6,4, 5, 4,2,4,2,4, 5, 4,4,3,4,4, 5, 4,4,4,4,4, 5],
            [5, 4,4,6,4,4, 5, 4,4,2,4,4, 5, 3,3,3,3,3, 5, 4,4,4,4,4, 5],
            [5, 4,6,4,4,4, 5, 4,2,4,2,4, 5, 4,4,3,4,4, 5, 4,4,4,4,4, 5],
            [5, 6,4,4,4,4, 5, 2,4,4,4,2, 5, 4,4,3,4,4, 5, 4,4,4,4,4, 5],
            [5]*25,
        ]
        grid_out = [
            [5]*25,
            [5, 6,4,4,4,6, 5, 2,4,4,4,2, 5, 4,4,3,4,4, 5, 2,4,4,4,2, 5],
            [5, 4,4,4,6,4, 5, 4,2,4,2,4, 5, 4,4,3,4,4, 5, 4,2,4,2,4, 5],
            [5, 4,4,6,4,4, 5, 4,4,2,4,4, 5, 3,3,3,3,3, 5, 4,4,2,3,3, 5],
            [5, 4,6,4,4,4, 5, 4,2,4,2,4, 5, 4,4,3,4,4, 5, 4,2,3,4,4, 5],
            [5, 6,4,4,4,4, 5, 2,4,4,4,2, 5, 4,4,3,4,4, 5, 2,4,3,4,4, 5],
            [5]*25,
        ]
        res = compose_partitioned_rooms(grid_in)
        assert res == grid_out
        assert extract_partitioned_rooms_rule([{"input": grid_in, "output": grid_out}])

    def test_sort_vertical_pillars_ascending(self):
        from src.dsl import sort_vertical_pillars_ascending, extract_sort_vertical_pillars_rule

        # Horizontal divider axis at row 2 with color 6, background 8
        # Col 1: height 5 (color 2), Col 3: height 3 (color 4)
        grid_in = [
            [8, 2, 8, 8, 8],
            [8, 2, 8, 4, 8],
            [6, 2, 6, 4, 6],
            [8, 2, 8, 4, 8],
            [8, 2, 8, 8, 8],
        ]
        # In output: heights sorted ascending: col 1 gets height 3, col 3 gets height 5
        grid_out = [
            [8, 8, 8, 4, 8],
            [8, 2, 8, 4, 8],
            [6, 2, 6, 4, 6],
            [8, 2, 8, 4, 8],
            [8, 8, 8, 4, 8],
        ]
        res = sort_vertical_pillars_ascending(grid_in)
        assert res == grid_out
        assert extract_sort_vertical_pillars_rule([{"input": grid_in, "output": grid_out}])

    def test_simulate_diagonal_optical_mirrors(self):
        from src.dsl import simulate_diagonal_optical_mirrors, extract_diagonal_optical_mirrors_rule

        # 5x5 canvas with background 0, seeds 1 at (0, 0) and (4, 4), mirror 6 at (2, 2)
        grid_in = [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 6, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ]
        # Diagonal from (0,0) to (4,4) filled with 1. Mirror at (2,2) reflects perpendicular diagonal: (0,4) to (4,0) filled with 6!
        grid_out = [
            [1, 0, 0, 0, 6],
            [0, 1, 0, 6, 0],
            [0, 0, 6, 0, 0],
            [0, 6, 0, 1, 0],
            [6, 0, 0, 0, 1],
        ]
        res = simulate_diagonal_optical_mirrors(grid_in)
        assert res == grid_out
        assert extract_diagonal_optical_mirrors_rule([{"input": grid_in, "output": grid_out}])

    def test_transfer_circuit_tokens(self):
        from src.dsl import transfer_circuit_tokens, extract_circuit_tokens_rule

        # 8x8 grid with two 3x3 stamps: stamp 3 (center 4) and stamp 4 (center 2)
        # Stamp 3 points to stamp 4, so stamp 3 receives token 2 and stamp 4 is erased
        grid_in = [
            [3, 3, 3, 0, 0, 0, 0, 0],
            [3, 4, 3, 5, 5, 0, 0, 0],
            [3, 3, 3, 0, 5, 0, 0, 0],
            [0, 0, 0, 0, 5, 5, 5, 0],
            [0, 0, 0, 0, 0, 4, 4, 4],
            [0, 0, 0, 0, 0, 4, 2, 4],
            [0, 0, 0, 0, 0, 4, 4, 4],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        grid_out = [
            [3, 3, 3, 0, 0, 0, 0, 0],
            [3, 2, 3, 5, 5, 0, 0, 0],
            [3, 3, 3, 0, 5, 0, 0, 0],
            [0, 0, 0, 0, 5, 5, 5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        res = transfer_circuit_tokens(grid_in)
        assert res == grid_out
        assert extract_circuit_tokens_rule([{"input": grid_in, "output": grid_out}])

    def test_generate_room_histogram(self):
        from src.dsl import generate_room_histogram, extract_room_histogram_rule

        # Synthetic grid with noise 5, a 4x4 box of 1s (1 room) and two side-by-side 4x4 boxes of 2s (2 rooms)
        grid_in = [
            [1, 1, 1, 1, 0, 5, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 5, 0, 0],
            [1, 0, 0, 1, 0, 5, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 5, 0],
            [0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 2, 2, 2, 2, 2, 0],
            [5, 0, 2, 0, 0, 2, 0, 0, 2, 0],
            [0, 0, 2, 0, 0, 2, 0, 0, 2, 0],
            [0, 0, 2, 2, 2, 2, 2, 2, 2, 0],
            [0, 5, 0, 0, 0, 0, 5, 0, 0, 0],
        ]
        # Expected output: 1 of color 1 (padded with 5 to width 2), and 2 of color 2
        grid_out = [
            [1, 5],
            [2, 2],
        ]
        res = generate_room_histogram(grid_in)
        assert res == grid_out
        assert extract_room_histogram_rule([{"input": grid_in, "output": grid_out}])

    def test_hybrid_solver_generates_diverse_attempts(self):
        from src.models.hybrid_solver import HybridSolver

        solver = HybridSolver()
        # Task with identity transform
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}
            ],
            "test": [
                {"input": [[5, 6], [7, 8]]}
            ]
        }
        preds = solver.solve_task(task)
        assert len(preds) == 1
        assert "attempt_1" in preds[0]
        assert "attempt_2" in preds[0]
        # Attempt 1 and Attempt 2 must never be bitwise identical
        assert preds[0]["attempt_1"] != preds[0]["attempt_2"]

    def test_slide_particles_by_legend_compass(self):
        from src.dsl import slide_particles_by_legend_compass, extract_legend_compass_sliding_rule

        # 10x10 canvas: legend box of 5s at (0, 0), vector (1, 0) pointing DOWN towards obstacle of 5s at row 8
        grid_in = [
            [5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
            [5, 0, 1, 0, 5, 0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5, 0, 0, 0, 0, 0],
            [5, 0, 2, 0, 5, 0, 0, 0, 0, 0],
            [5, 5, 5, 5, 5, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        grid_out = [
            [5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
            [5, 0, 1, 0, 5, 0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5, 0, 0, 0, 0, 0],
            [5, 0, 2, 0, 5, 0, 0, 0, 0, 0],
            [5, 5, 5, 5, 5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        res = slide_particles_by_legend_compass(grid_in)
        assert res == grid_out
        assert extract_legend_compass_sliding_rule([{"input": grid_in, "output": grid_out}])

    def test_simulate_projectile_cannons(self):
        from src.dsl import simulate_projectile_cannons, extract_projectile_cannons_rule

        # 10x10 grid with background 7: an up-arrow cannon at (6, 4) shooting 5 pixels up to (1, 4)
        grid_in = [
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 0, 7, 7, 7, 7, 7],
            [7, 7, 7, 0, 7, 0, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        ]
        # Bullet 9 appears 5 units up at (1, 4)
        grid_out = [
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 9, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 0, 7, 7, 7, 7, 7],
            [7, 7, 7, 0, 7, 0, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
        ]
        res = simulate_projectile_cannons(grid_in)
        assert res == grid_out
        assert extract_projectile_cannons_rule([{"input": grid_in, "output": grid_out}])

    def test_heal_orthogonal_wire_detours(self):
        from src.dsl import heal_orthogonal_wire_detours, extract_wire_detours_rule

        # 10x10 grid with background 0, foreground 3.
        # A horizontal wire at row 4 broken at cols 3, 4, 5 (gap of 3)
        # with an adjacent diagonal detour at (5, 4), (6, 5)
        grid_in = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 0, 0, 0, 3, 3, 3, 3],
            [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        # In output, the gap at row 4 cols 3..5 is filled, and the diagonal detour is erased
        grid_out = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        res = heal_orthogonal_wire_detours(grid_in)
        assert res == grid_out
        assert extract_wire_detours_rule([{"input": grid_in, "output": grid_out}])

    def test_generate_greek_key_labyrinth(self):
        from src.dsl import generate_greek_key_labyrinth, extract_greek_key_labyrinth_rule

        # 12x12 grid with central cross at (4, 5) of color 8 and arm length 1
        grid_in = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 8, 0, 8, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        grid_out = [
            [0, 8, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [0, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8],
            [0, 8, 0, 8, 0, 8, 8, 8, 8, 8, 0, 8],
            [0, 8, 0, 8, 0, 8, 0, 0, 0, 8, 0, 8],
            [0, 8, 0, 8, 8, 0, 8, 8, 0, 8, 0, 8],
            [0, 8, 0, 0, 0, 8, 0, 8, 0, 8, 0, 8],
            [0, 8, 8, 8, 8, 8, 0, 8, 0, 8, 0, 8],
            [0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 0, 8, 0, 8],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 8],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
        ]
        res = generate_greek_key_labyrinth(grid_in)
        assert res == grid_out
        assert extract_greek_key_labyrinth_rule([{"input": grid_in, "output": grid_out}])

    def test_connect_room_tour_corridors(self):
        from src.dsl import connect_room_tour_corridors, extract_room_tour_corridors_rule

        # 9x9 grid with 2 rooms (center 2 and center 3), tour at row 7: [2, 3]
        grid_in = [
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 1, 1, 8, 1, 1, 8, 8, 8],
            [8, 1, 2, 8, 1, 3, 8, 8, 8],
            [8, 1, 1, 8, 1, 1, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 2, 8, 3, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
        ]
        # In output, the doorway between Room 2 and Room 3 at (2, 3) is filled with 2
        grid_out = [
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 1, 1, 8, 1, 1, 8, 8, 8],
            [8, 1, 2, 2, 1, 3, 8, 8, 8],
            [8, 1, 1, 8, 1, 1, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 2, 8, 3, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
        ]
        res = connect_room_tour_corridors(grid_in)
        assert res == grid_out
        assert extract_room_tour_corridors_rule([{"input": grid_in, "output": grid_out}])
































