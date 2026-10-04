from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import numpy as np

from src.dsl import primitives as P
from src.metrics import grid_equal

from src.dsl.overlay import SPLIT_OVERLAY_OPERATIONS, split_and_overlay
from src.dsl.cellular_automata import extract_3x3_local_rules, apply_3x3_local_rules

Grid = list[list[int]]


@dataclass
class Program:
    """Executable DSL program composed of one or more primitive functions."""

    name: str
    steps: list[tuple[str, Callable[[Grid], Grid]]]

    def execute(self, grid: Grid) -> Grid:
        curr = grid
        for _, fn in self.steps:
            curr = fn(curr)
        return curr


def find_dynamic_color_mapping(train_pairs: list[dict[str, Any]]) -> dict[int, int] | None:
    """Finds if a consistent global color remap explains the input->output pairs."""
    mapping: dict[int, int] = {}
    for pair in train_pairs:
        inp = np.asarray(pair["input"], dtype=int)
        out = np.asarray(pair["output"], dtype=int)
        if inp.shape != out.shape:
            return None
        for in_v, out_v in zip(inp.flat, out.flat):
            if in_v in mapping and mapping[in_v] != out_v:
                return None
            mapping[in_v] = int(out_v)
    return mapping if mapping else None


def generate_single_step_candidates(train_pairs: list[dict[str, Any]]) -> list[Program]:
    """Generates 1-step candidate transformation programs."""
    progs: list[Program] = []

    # 1. Fixed geometric transforms
    geom_ops: list[tuple[str, Callable[[Grid], Grid]]] = [
        ("identity", lambda g: [row[:] for row in g]),
        ("rot90", P.rotate_90),
        ("rot180", P.rotate_180),
        ("rot270", P.rotate_270),
        ("reflect_h", P.reflect_horizontal),
        ("reflect_v", P.reflect_vertical),
        ("transpose", P.transpose),
    ]
    for name, op in geom_ops:
        progs.append(Program(name=name, steps=[(name, op)]))

    # 2. Gravity drops along cardinal directions
    for direction in ["down", "up", "left", "right"]:
        op_name = f"gravity_{direction}"
        progs.append(
            Program(
                name=op_name,
                steps=[(op_name, lambda g, d=direction: P.gravity(g, direction=d))],
            )
        )

    # 3. Content cropping for various background colors
    # Find background colors present in inputs
    bg_candidates = {0}
    for p in train_pairs:
        inp_arr = np.asarray(p["input"], dtype=int)
        bg_candidates.add(int(inp_arr[0, 0]))

    for bg in sorted(bg_candidates):
        crop_name = f"crop_content_bg{bg}"
        progs.append(
            Program(
                name=crop_name,
                steps=[(crop_name, lambda g, b=bg: P.crop_to_content(g, background=b))],
            )
        )
        # Crop largest / smallest object
        for conn in [4, 8]:
            p_lg = f"crop_largest_bg{bg}_c{conn}"
            progs.append(
                Program(
                    name=p_lg,
                    steps=[(p_lg, lambda g, b=bg, c=conn: P.crop_largest_object(g, background=b, connectivity=c))],
                )
            )
            p_sm = f"crop_smallest_bg{bg}_c{conn}"
            progs.append(
                Program(
                    name=p_sm,
                    steps=[(p_sm, lambda g, b=bg, c=conn: P.crop_smallest_object(g, background=b, connectivity=c))],
                )
            )
        # Noise removal
        p_noise = f"remove_noise_bg{bg}"
        progs.append(
            Program(
                name=p_noise,
                steps=[(p_noise, lambda g, b=bg: P.remove_small_objects(g, max_size=1, background=b))],
            )
        )
        # Rigid body object gravity
        for direction in ["down", "up", "left", "right"]:
            p_rg = f"rigid_gravity_{direction}_bg{bg}"
            progs.append(
                Program(
                    name=p_rg,
                    steps=[(p_rg, lambda g, d=direction, b=bg: P.rigid_object_gravity(g, direction=d, background=b))],
                )
            )

    # Enclosed region filling
    output_colors = set()
    for p in train_pairs:
        output_colors.update(np.unique(np.asarray(p["output"])).tolist())
    for fill_c in sorted(output_colors):
        for bg in sorted(bg_candidates):
            if fill_c != bg:
                enc_name = f"fill_enclosed_bg{bg}_c{fill_c}"
                progs.append(
                    Program(
                        name=enc_name,
                        steps=[(enc_name, lambda g, b=bg, fc=fill_c: P.fill_enclosed_regions(g, fill_color=fc, background=b))],
                    )
                )

    # 4. Mirror appending
    for axis in ["vertical", "horizontal", "both"]:
        mirr_name = f"mirror_append_{axis}"
        progs.append(
            Program(
                name=mirr_name,
                steps=[(mirr_name, lambda g, a=axis: P.mirror_append(g, axis=a))],
            )
        )

    # 4b. Connect matching dots
    for bg in sorted(bg_candidates):
        cmd_name = f"connect_dots_bg{bg}"
        progs.append(
            Program(
                name=cmd_name,
                steps=[(cmd_name, lambda g, b=bg: P.connect_matching_dots(g, background=b))],
            )
        )

    # 4c. Repair periodic rooms / subgrids
    for pad in [1, 2]:
        per_name = f"repair_periodic_pad{pad}"
        progs.append(
            Program(
                name=per_name,
                steps=[(per_name, lambda g, p=pad: P.repair_periodic_subgrids(g, pad=p))],
            )
        )

    # 4d. Crop hollow rectangle interior
    for bg in sorted(bg_candidates):
        hr_name = f"crop_hollow_rect_bg{bg}"
        progs.append(
            Program(
                name=hr_name,
                steps=[(hr_name, lambda g, b=bg: P.crop_hollow_rectangle_interior(g, background=b))],
            )
        )

    # 5. Symmetry completions
    for axis in ["vertical", "horizontal"]:
        for src in (["left", "right"] if axis == "vertical" else ["top", "bottom"]):
            sym_name = f"complete_sym_{axis}_{src}"
            progs.append(
                Program(
                    name=sym_name,
                    steps=[
                        (
                            sym_name,
                            lambda g, a=axis, s=src: P.complete_symmetry(g, axis=a, source=s),
                        )
                    ],
                )
            )

    # 6. Integer scaling (factors 2, 3)
    for factor in [2, 3]:
        scale_name = f"scale_{factor}"
        progs.append(
            Program(
                name=scale_name,
                steps=[(scale_name, lambda g, f=factor: P.scale_grid(g, factor=f))],
            )
        )

    # 7. Tiling (2x2, 2x1, 1x2, 3x3)
    for rh, rw in [(2, 2), (2, 1), (1, 2), (3, 3)]:
        tile_name = f"tile_{rh}x{rw}"
        progs.append(
            Program(
                name=tile_name,
                steps=[(tile_name, lambda g, h=rh, w=rw: P.tile_grid(g, repeats_h=h, repeats_w=w))],
            )
        )

    # 8. Dynamic consistent color mapping
    cmap = find_dynamic_color_mapping(train_pairs)
    if cmap is not None:
        cmap_name = f"dynamic_remap_colors"
        progs.append(
            Program(
                name=cmap_name,
                steps=[(cmap_name, lambda g, m=cmap: P.remap_colors(g, m))],
            )
        )

    # 9. Split and Logical Overlay
    for op in SPLIT_OVERLAY_OPERATIONS:
        op_name = f"overlay_{op}"
        progs.append(
            Program(
                name=op_name,
                steps=[(op_name, lambda g, o=op: split_and_overlay(g, op=o) or g)],
            )
        )

    # 10. Cellular Automata 3x3 Local Rules
    rules_3x3 = extract_3x3_local_rules(train_pairs)
    if rules_3x3 is not None:
        ca_name = "cellular_automata_3x3"
        progs.append(
            Program(
                name=ca_name,
                steps=[(ca_name, lambda g, r=rules_3x3: apply_3x3_local_rules(g, r, fallback_to_input=True) or g)],
            )
        )

    # 11. Odd-one-out 3x3 Block Selection
    progs.append(
        Program(
            name="select_odd_one_out_3x3",
            steps=[("select_odd_one_out_3x3", lambda g: P.select_odd_one_out_block(g, win_h=3, win_w=3))],
        )
    )

    # 12. Window Relative to Single Marker
    for rel in ["below", "above", "center", "right", "left"]:
        m_name = f"marker_win_{rel}_3x3"
        progs.append(
            Program(
                name=m_name,
                steps=[(m_name, lambda g, r=rel: P.extract_window_relative_to_marker(g, relation=r, win_h=3, win_w=3))],
            )
        )

    # 13. Directional and Diagonal Raycasting
    for d in ["down", "up", "right", "left", "vertical", "horizontal", "cross"]:
        rc_name = f"raycast_{d}"
        progs.append(
            Program(
                name=rc_name,
                steps=[(rc_name, lambda g, d_dir=d: P.raycast_directional(g, direction=d_dir))],
            )
        )
    progs.append(
        Program(
            name="raycast_diagonal_bouncing",
            steps=[("raycast_diagonal_bouncing", lambda g: P.raycast_diagonal_bouncing(g))],
        )
    )

    # 14. Color Objects by Size, Height, or Width
    for prop in ["size", "height", "width"]:
        prop_rule = P.extract_object_property_color_rule(train_pairs, property_name=prop)
        if prop_rule is not None:
            p_name = f"color_objs_by_{prop}"
            progs.append(
                Program(
                    name=p_name,
                    steps=[(p_name, lambda g, pr=prop, pm=prop_rule: P.color_objects_by_property(g, pm, property_name=pr))],
                )
            )

    # 15. Enclosed Pixel Color Summary (1x1 output)
    progs.append(
        Program(
            name="extract_enclosed_pixel_color",
            steps=[("extract_enclosed_pixel_color", lambda g: P.extract_enclosed_pixel_color(g))],
        )
    )

    # 16. Symmetry Binary Classifier (1x1 output)
    unique_outs = {tr["output"][0][0] for tr in train_pairs if len(tr.get("output", [])) == 1 and len(tr["output"][0]) == 1}
    if len(unique_outs) == 2:
        c1, c2 = list(unique_outs)
        for sym_c, asym_c in [(c1, c2), (c2, c1)]:
            s_name = f"classify_symmetry_{sym_c}_{asym_c}"
            progs.append(
                Program(
                    name=s_name,
                    steps=[(s_name, lambda g, sc=sym_c, ac=asym_c: P.classify_by_symmetry(g, sym_color=sc, asym_color=ac))],
                )
            )

    # 17. Binary Shape to Color Classifier (1x1 output)
    shape_rule = P.extract_binary_shape_classifier(train_pairs)
    if shape_rule is not None:
        progs.append(
            Program(
                name="classify_binary_shape",
                steps=[("classify_binary_shape", lambda g, sm=shape_rule: P.classify_binary_shape(g, shape_map=sm))],
            )
        )

    # 18. Kaleidoscope & Mirror Tiling
    progs.append(
        Program(
            name="tile_kaleidoscope_2x2",
            steps=[("tile_kaleidoscope_2x2", lambda g: P.tile_kaleidoscope_2x2(g))],
        )
    )
    progs.append(
        Program(
            name="tile_mirror_h",
            steps=[("tile_mirror_h", lambda g: P.tile_mirror_horizontal(g))],
        )
    )
    progs.append(
        Program(
            name="tile_mirror_v",
            steps=[("tile_mirror_v", lambda g: P.tile_mirror_vertical(g))],
        )
    )

    # 19. Kronecker Fractal Self-Expansion
    progs.append(
        Program(
            name="kronecker_fractal_expansion",
            steps=[("kronecker_fractal_expansion", lambda g: P.kronecker_fractal_expansion(g))],
        )
    )

    # 20. Fixed Target Shape Tiling
    out_shapes = {np.asarray(p["output"]).shape for p in train_pairs}
    if len(out_shapes) == 1:
        th, tw = list(out_shapes)[0]
        if th <= 30 and tw <= 30:
            p_name = f"tile_to_shape_{th}x{tw}"
            progs.append(
                Program(
                    name=p_name,
                    steps=[(p_name, lambda g, t_h=th, t_w=tw: P.tile_to_target_shape(g, target_h=t_h, target_w=t_w))],
                )
            )

    return progs


def generate_two_step_candidates(
    train_pairs: list[dict[str, Any]],
    promising_step1: list[Program],
) -> list[Program]:
    """Generates 2-step pipelines combining promising transformations."""
    progs: list[Program] = []

    # Target output sizes across train pairs
    target_shapes = {np.asarray(p["output"]).shape for p in train_pairs}

    # Step 2 candidate pool
    step2_pool: list[tuple[str, Callable[[Grid], Grid]]] = [
        ("rot90", P.rotate_90),
        ("rot180", P.rotate_180),
        ("rot270", P.rotate_270),
        ("reflect_h", P.reflect_horizontal),
        ("reflect_v", P.reflect_vertical),
        ("transpose", P.transpose),
    ]

    cmap = find_dynamic_color_mapping(train_pairs)
    if cmap is not None:
        step2_pool.append(("remap_colors", lambda g, m=cmap: P.remap_colors(g, m)))

    for p1 in promising_step1:
        # Check if output dimension of p1 matches target shapes on first train pair
        try:
            sample_out = p1.execute(train_pairs[0]["input"])
            sample_shape = (len(sample_out), len(sample_out[0]) if sample_out else 0)
        except Exception:
            continue

        for s2_name, s2_fn in step2_pool:
            composite_name = f"{p1.name} >> {s2_name}"
            progs.append(
                Program(
                    name=composite_name,
                    steps=list(p1.steps) + [(s2_name, s2_fn)],
                )
            )

    return progs


def evaluate_program(
    program: Program,
    train_pairs: list[dict[str, Any]],
    short_circuit: bool = True,
) -> float:
    """Evaluates program on training demonstration pairs.

    Returns fraction of training pairs perfectly solved (0.0 to 1.0).
    """
    if not train_pairs:
        return 0.0

    solved_count = 0
    for pair in train_pairs:
        try:
            pred = program.execute(pair["input"])
            if grid_equal(pred, pair["output"]):
                solved_count += 1
            elif short_circuit:
                return 0.0
        except Exception:
            if short_circuit:
                return 0.0

    return solved_count / len(train_pairs)


def search_dsl_programs(
    train_pairs: list[dict[str, Any]],
    max_level: int = 2,
    max_perfect_programs: int = 5,
) -> list[tuple[Program, float]]:
    """Searches for DSL programs solving 100% of demonstration pairs.

    Returns:
        List of (program, train_score) tuples, sorted by simplicity.
    """
    perfect_programs: list[tuple[Program, float]] = []

    # Level 1 Search
    level1_candidates = generate_single_step_candidates(train_pairs)
    promising_for_step2: list[Program] = []

    for prog in level1_candidates:
        score = evaluate_program(prog, train_pairs, short_circuit=True)
        if score == 1.0:
            perfect_programs.append((prog, 1.0))
            if len(perfect_programs) >= max_perfect_programs:
                return perfect_programs
        else:
            # Check if dimension matches
            try:
                out0 = prog.execute(train_pairs[0]["input"])
                target0 = train_pairs[0]["output"]
                if len(out0) == len(target0) and len(out0[0]) == len(target0[0]):
                    promising_for_step2.append(prog)
            except Exception:
                pass

    if max_level >= 2 and len(perfect_programs) < max_perfect_programs:
        # Level 2 Search
        level2_candidates = generate_two_step_candidates(
            train_pairs, promising_for_step2[:15]
        )
        for prog in level2_candidates:
            score = evaluate_program(prog, train_pairs, short_circuit=True)
            if score == 1.0:
                perfect_programs.append((prog, 1.0))
                if len(perfect_programs) >= max_perfect_programs:
                    break

    return perfect_programs
