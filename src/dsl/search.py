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

    # 21. Color Objects by Hole Count (Euler Genus)
    holes_rule = P.extract_holes_to_color_rule(train_pairs)
    if holes_rule is not None:
        progs.append(
            Program(
                name="color_objects_by_holes",
                steps=[("color_objects_by_holes", lambda g, hr=holes_rule: P.color_objects_by_holes(g, hr))],
            )
        )

    # 22. Extract Object by Hole Count
    for h_target in [0, 1, 2]:
        p_name = f"extract_obj_{h_target}_holes"
        progs.append(
            Program(
                name=p_name,
                steps=[(p_name, lambda g, ht=h_target: P.extract_object_by_hole_count(g, target_holes=ht))],
            )
        )

    # 23. Recolor Motif / Template Matching
    motif_rule = P.extract_recolor_motif_rule(train_pairs)
    if motif_rule is not None:
        m_mask, s_c, d_c = motif_rule
        progs.append(
            Program(
                name="recolor_motif",
                steps=[(
                    "recolor_motif",
                    lambda g, m=m_mask, s=s_c, d=d_c: P.recolor_motif(g, motif=m, src_color=s, dst_color=d),
                )],
            )
        )

    # 24. Fill Enclosed Regions by Area
    for bg in sorted(bg_candidates):
        enc_area_rule = P.extract_enclosed_area_rule(train_pairs, background=bg)
        if enc_area_rule is not None:
            p_name = f"fill_enclosed_area_bg{bg}"
            progs.append(
                Program(
                    name=p_name,
                    steps=[(
                        p_name,
                        lambda g, ar=enc_area_rule, b=bg: P.fill_enclosed_regions_by_area(g, area_to_color=ar, background=b),
                    )],
                )
            )

    # 25. Color Objects by Rank
    for bg in sorted(bg_candidates):
        rank_rule = P.extract_rank_color_rule(train_pairs, background=bg)
        if rank_rule is not None:
            metric, rev, r_map = rank_rule
            p_name = f"color_objects_by_rank_{metric}_bg{bg}"
            progs.append(
                Program(
                    name=p_name,
                    steps=[(
                        p_name,
                        lambda g, rm=r_map, m=metric, r=rev, b=bg: P.color_objects_by_rank(g, rank_to_color=rm, metric=m, reverse=r, background=b),
                    )],
                )
            )

    # 26. Extrapolate Periodic Grid
    for bg in sorted(bg_candidates):
        per_rule = P.extract_periodic_grid_rule(train_pairs, background=bg)
        if per_rule is not None:
            mode, period = per_rule
            p_name = f"extrapolate_periodic_{mode}_p{period}_bg{bg}"
            progs.append(
                Program(
                    name=p_name,
                    steps=[(
                        p_name,
                        lambda g, md=mode, p=period, b=bg: P.extrapolate_periodic_grid(g, mode=md, period=p, background=b),
                    )],
                )
            )

    # 27. Collinear Point Progression Extrapolation
    for bg in sorted(bg_candidates):
        prog_rule = P.extract_point_progression_rule(train_pairs, background=bg)
        if prog_rule is not None:
            pt_c, dst_c = prog_rule
            p_name = f"extrapolate_point_progression_c{pt_c}_to_{dst_c}_bg{bg}"
            progs.append(
                Program(
                    name=p_name,
                    steps=[(
                        p_name,
                        lambda g, pc=pt_c, dc=dst_c, b=bg: P.extrapolate_point_progression(g, point_color=pc, new_color=dc, background=b),
                    )],
                )
            )

    # 28. Trigger Release Gravity
    for bg in sorted(bg_candidates):
        progs.append(
            Program(
                name=f"trigger_release_gravity_bg{bg}",
                steps=[(
                    f"trigger_release_gravity_bg{bg}",
                    lambda g, b=bg: P.trigger_release_gravity(g, background=b),
                )],
            )
        )

    # 29. Fill Puzzle Socket
    socket_rule = P.extract_puzzle_socket_rule(train_pairs)
    if socket_rule is not None:
        p_c, h_c, f_c = socket_rule
        progs.append(
            Program(
                name="fill_puzzle_socket",
                steps=[(
                    "fill_puzzle_socket",
                    lambda g, p=p_c, h=h_c, f=f_c: P.fill_puzzle_socket(g, piece_color=p, hole_color=h, fill_color=f),
                )],
            )
        )

    # 30. Slide Dots to Wall
    for bg in sorted(bg_candidates):
        slide_rule = P.extract_slide_dots_rule(train_pairs, background=bg)
        if slide_rule is not None:
            dot_c, d_dir = slide_rule
            p_name = f"slide_dots_{d_dir}_c{dot_c}_bg{bg}"
            progs.append(
                Program(
                    name=p_name,
                    steps=[(
                        p_name,
                        lambda g, dc=dot_c, dd=d_dir, b=bg: P.slide_dots_to_wall(g, dot_color=dc, direction=dd, background=b),
                    )],
                )
            )

    # 31. Multi-source Diagonal Ray Projection
    colors_in_inputs = set()
    for p in train_pairs:
        colors_in_inputs.update(np.unique(p["input"]).tolist())
    for bg in sorted(bg_candidates):
        for rc in sorted(list(colors_in_inputs)):
            if rc != bg:
                progs.append(
                    Program(
                        name=f"project_diagonal_rays_c{rc}_bg{bg}",
                        steps=[(
                            f"project_diagonal_rays_c{rc}_bg{bg}",
                            lambda g, r_c=rc, b=bg: P.project_diagonal_rays(g, ray_color=r_c, background=b),
                        )],
                    )
                )

    # 32. Straighten Orthogonal Lines
    for bg in sorted(bg_candidates):
        for lc in sorted(list(colors_in_inputs)):
            if lc != bg:
                progs.append(
                    Program(
                        name=f"straighten_orthogonal_lines_c{lc}_bg{bg}",
                        steps=[(
                            f"straighten_orthogonal_lines_c{lc}_bg{bg}",
                            lambda g, l_c=lc, b=bg: P.straighten_orthogonal_lines(g, line_color=l_c, background=b),
                        )],
                    )
                )

    # 33. Masked Reflection Inpainting
    refl_rule = P.extract_masked_reflection_rule(train_pairs)
    if refl_rule is not None:
        obj_c, mask_c, ax = refl_rule
        progs.append(
            Program(
                name=f"inpaint_masked_reflection_{ax}",
                steps=[(
                    f"inpaint_masked_reflection_{ax}",
                    lambda g, o=obj_c, m=mask_c, a=ax: P.inpaint_masked_reflection(g, obj_color=o, mask_color=m, axis=a),
                )],
            )
        )

    # 34. Restore Bilateral Reflection Symmetry
    if P.extract_restore_bilateral_symmetry_rule(train_pairs):
        progs.append(
            Program(
                name="restore_bilateral_symmetry",
                steps=[("restore_bilateral_symmetry", lambda g: P.restore_bilateral_symmetry(g))],
            )
        )

    # 35. Indicator-Ranked Pillar Filling
    if P.extract_ranked_pillars_rule(train_pairs):
        progs.append(
            Program(
                name="fill_ranked_pillars_by_indicator",
                steps=[("fill_ranked_pillars_by_indicator", lambda g: P.fill_ranked_pillars_by_indicator(g))],
            )
        )

    # 36. Legend / Palette-Guided Cavity Filling
    if P.extract_legend_cavity_rule(train_pairs):
        progs.append(
            Program(
                name="fill_cavities_from_legend",
                steps=[("fill_cavities_from_legend", lambda g: P.fill_cavities_from_legend(g))],
            )
        )

    # 37. Standardize Parallel Lines Length
    if P.extract_standardize_parallel_lines_rule(train_pairs):
        progs.append(
            Program(
                name="standardize_parallel_lines",
                steps=[("standardize_parallel_lines", lambda g: P.standardize_parallel_lines(g))],
            )
        )

    # 38. Liquid Gravity Flow into Receptacle Cavities
    if P.extract_liquid_gravity_rule(train_pairs):
        progs.append(
            Program(
                name="simulate_liquid_gravity",
                steps=[("simulate_liquid_gravity", lambda g: P.simulate_liquid_gravity(g))],
            )
        )

    # 39. Sokoban KxK Channel Sliding Navigation
    sokoban_params = P.extract_sokoban_channel_rule(train_pairs)
    if sokoban_params is not None:
        corr_c, hole_c = sokoban_params
        progs.append(
            Program(
                name="slide_sokoban_block_in_channel",
                steps=[
                    (
                        "slide_sokoban_block_in_channel",
                        lambda g, c=corr_c, h=hole_c: P.slide_sokoban_block_in_channel(
                            g, corridor_color=c, hole_color=h
                        ),
                    )
                ],
            )
        )

    # 40. Multi-Room Partition Overlay
    if P.extract_partitioned_rooms_rule(train_pairs):
        progs.append(
            Program(
                name="compose_partitioned_rooms",
                steps=[("compose_partitioned_rooms", lambda g: P.compose_partitioned_rooms(g))],
            )
        )

    # 41. Ascending Vertical Pillar Height Sorting
    if P.extract_sort_vertical_pillars_rule(train_pairs):
        progs.append(
            Program(
                name="sort_vertical_pillars_ascending",
                steps=[("sort_vertical_pillars_ascending", lambda g: P.sort_vertical_pillars_ascending(g))],
            )
        )

    # 42. Diagonal Optical Ray Tracing & Mirror Reflection
    if P.extract_diagonal_optical_mirrors_rule(train_pairs):
        progs.append(
            Program(
                name="simulate_diagonal_optical_mirrors",
                steps=[("simulate_diagonal_optical_mirrors", lambda g: P.simulate_diagonal_optical_mirrors(g))],
            )
        )

    # 43. Circuit Token Packet Passing
    if P.extract_circuit_tokens_rule(train_pairs):
        progs.append(
            Program(
                name="transfer_circuit_tokens",
                steps=[("transfer_circuit_tokens", lambda g: P.transfer_circuit_tokens(g))],
            )
        )

    # 44. Room Histogram Generator
    if P.extract_room_histogram_rule(train_pairs):
        progs.append(
            Program(
                name="generate_room_histogram",
                steps=[("generate_room_histogram", lambda g: P.generate_room_histogram(g))],
            )
        )

    # 45. Legend Compass Particle Sliding
    if P.extract_legend_compass_sliding_rule(train_pairs):
        progs.append(
            Program(
                name="slide_particles_by_legend_compass",
                steps=[("slide_particles_by_legend_compass", lambda g: P.slide_particles_by_legend_compass(g))],
            )
        )

    # 46. Projectile Cannon Physics & Exploding Targets
    if P.extract_projectile_cannons_rule(train_pairs):
        progs.append(
            Program(
                name="simulate_projectile_cannons",
                steps=[("simulate_projectile_cannons", lambda g: P.simulate_projectile_cannons(g))],
            )
        )

    # 47. Circuit Wire Detour Healing & Air Gap Preservation
    if P.extract_wire_detours_rule(train_pairs):
        progs.append(
            Program(
                name="heal_orthogonal_wire_detours",
                steps=[("heal_orthogonal_wire_detours", lambda g: P.heal_orthogonal_wire_detours(g))],
            )
        )

    # 48. Archimedean Greek Key Meander Labyrinth
    if P.extract_greek_key_labyrinth_rule(train_pairs):
        progs.append(
            Program(
                name="generate_greek_key_labyrinth",
                steps=[("generate_greek_key_labyrinth", lambda g: P.generate_greek_key_labyrinth(g))],
            )
        )

    # 49. Graph Tour Room Corridor Routing
    if P.extract_room_tour_corridors_rule(train_pairs):
        progs.append(
            Program(
                name="connect_room_tour_corridors",
                steps=[("connect_room_tour_corridors", lambda g: P.connect_room_tour_corridors(g))],
            )
        )

    # 50. Fractal Self-Similar Tessellation Inpainting
    if P.extract_fractal_tessellation_rule(train_pairs):
        progs.append(
            Program(
                name="inpaint_fractal_tessellation",
                steps=[("inpaint_fractal_tessellation", lambda g: P.inpaint_fractal_tessellation(g))],
            )
        )

    # 51. Circuit Wire Flood Fill of Powered Enclosures
    if P.extract_circuit_enclosures_rule(train_pairs):
        progs.append(
            Program(
                name="power_circuit_enclosures",
                steps=[("power_circuit_enclosures", lambda g: P.power_circuit_enclosures(g))],
            )
        )

    # 52. Periodic Optical Beam Ray Extrapolation
    if P.extract_optical_beams_rule(train_pairs):
        progs.append(
            Program(
                name="extrapolate_periodic_optical_beams",
                steps=[("extrapolate_periodic_optical_beams", lambda g: P.extrapolate_periodic_optical_beams(g))],
            )
        )

    # 53. Planar and Point Symmetry Mirror Reflections
    if P.extract_planar_mirrors_rule(train_pairs):
        progs.append(
            Program(
                name="reflect_shapes_across_planar_mirrors",
                steps=[("reflect_shapes_across_planar_mirrors", lambda g: P.reflect_shapes_across_planar_mirrors(g))],
            )
        )

    # 54. Isotropic 2D Lattice Ruler Schedule Replication
    if P.extract_isotropic_ruler_schedule_rule(train_pairs):
        progs.append(
            Program(
                name="replicate_isotropic_ruler_schedule",
                steps=[("replicate_isotropic_ruler_schedule", lambda g: P.replicate_isotropic_ruler_schedule(g))],
            )
        )

    # 55. Quadrant Dihedral Symmetry Template Scaling
    if P.extract_quadrant_dihedral_rule(train_pairs):
        progs.append(
            Program(
                name="scale_quadrant_dihedral_templates",
                steps=[("scale_quadrant_dihedral_templates", lambda g: P.scale_quadrant_dihedral_templates(g))],
            )
        )

    # 56. Cellular Automata 3x3 Local Rules (fallback generic memorizer)
    rules_3x3 = extract_3x3_local_rules(train_pairs)
    if rules_3x3 is not None:
        ca_name = "cellular_automata_3x3"
        progs.append(
            Program(
                name=ca_name,
                steps=[(ca_name, lambda g, r=rules_3x3: apply_3x3_local_rules(g, r, fallback_to_input=True) or g)],
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
        ("tile_mirror_h", P.tile_mirror_horizontal),
        ("tile_mirror_v", P.tile_mirror_vertical),
        ("kaleidoscope_2x2", P.tile_kaleidoscope_2x2),
        ("tile_2x2", lambda g: P.tile_grid(g, repeat_y=2, repeat_x=2)),
        ("gravity_down", lambda g: P.gravity(g, direction="down")),
        ("crop_content_bg0", lambda g: P.crop_to_content(g, background=0)),
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

        # Dynamic intermediate color remapping: learn mapping from p1(X) -> Y
        try:
            p1_outs = [p1.execute(pair["input"]) for pair in train_pairs]
            inter_pairs = [{"input": o1, "output": pair["output"]} for o1, pair in zip(p1_outs, train_pairs)]
            p1_cmap = find_dynamic_color_mapping(inter_pairs)
            if p1_cmap is not None:
                composite_name = f"{p1.name} >> remap_colors"
                progs.append(
                    Program(
                        name=composite_name,
                        steps=list(p1.steps) + [("remap_colors", lambda g, m=p1_cmap: P.remap_colors(g, m))],
                    )
                )
        except Exception:
            pass

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

    def priority_key(item: tuple[Program, float]) -> tuple[int, int]:
        p = item[0]
        name = p.name
        if name.startswith("cellular_automata"):
            tier = 2
        elif any(name.startswith(x) for x in ["tile_", "scale_"]):
            tier = 1
        else:
            tier = 0
        return (tier, len(p.steps))

    # Level 1 Search
    level1_candidates = generate_single_step_candidates(train_pairs)
    promising_for_step2: list[Program] = []

    for prog in level1_candidates:
        score = evaluate_program(prog, train_pairs, short_circuit=True)
        if score == 1.0:
            perfect_programs.append((prog, 1.0))
        else:
            # Check if dimension matches
            try:
                out0 = prog.execute(train_pairs[0]["input"])
                target0 = train_pairs[0]["output"]
                if len(out0) == len(target0) and len(out0[0]) == len(target0[0]):
                    promising_for_step2.append(prog)
            except Exception:
                pass

    if perfect_programs:
        perfect_programs.sort(key=priority_key)
        if len(perfect_programs) >= max_perfect_programs:
            return perfect_programs[:max_perfect_programs]

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

    # Level 3: D4 Dihedral Test-Time Induction (rotations & reflections)
    if not perfect_programs and max_level >= 3:
        d4_transforms = [
            ("rot90", lambda g: np.rot90(g, 1).tolist(), lambda g: np.rot90(g, 3).tolist()),
            ("rot180", lambda g: np.rot90(g, 2).tolist(), lambda g: np.rot90(g, 2).tolist()),
            ("rot270", lambda g: np.rot90(g, 3).tolist(), lambda g: np.rot90(g, 1).tolist()),
            ("fliplr", lambda g: np.fliplr(g).tolist(), lambda g: np.fliplr(g).tolist()),
            ("flipud", lambda g: np.flipud(g).tolist(), lambda g: np.flipud(g).tolist()),
        ]
        for t_name, fwd, inv in d4_transforms:
            try:
                aug_train = [{"input": fwd(tr["input"]), "output": fwd(tr["output"])} for tr in train_pairs]
                aug_progs = search_dsl_programs(aug_train, max_level=1, max_perfect_programs=1)
                if aug_progs:
                    best_prog, _ = aug_progs[0]
                    composite_name = f"{t_name} >> {best_prog.name} >> inv_{t_name}"
                    tta_prog = Program(
                        name=composite_name,
                        steps=[
                            (t_name, fwd),
                            *best_prog.steps,
                            (f"inv_{t_name}", inv),
                        ],
                    )
                    perfect_programs.append((tta_prog, 1.0))
                    if len(perfect_programs) >= max_perfect_programs:
                        break
            except Exception:
                pass

    perfect_programs.sort(key=priority_key)
    return perfect_programs[:max_perfect_programs]
