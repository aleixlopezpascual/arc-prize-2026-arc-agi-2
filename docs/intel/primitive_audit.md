# DSL Primitive Generalization Audit (2026-10-29)

Script: `python3 -m src.tools.audit_primitives` (raw output: `docs/intel/primitive_audit_raw.json`).
For every task in the train (1000), eval (120) and public test (240) splits, collects **all** programs that fit 100% of the demo pairs (`max_level=3`), runs them on the test inputs, and records whether they reproduce the ground truth (train/eval only; test has no solutions, so only "fires" is measured). 0 errors/timeouts across 1,360 tasks.

## Headline

| Split | Tasks with a fitting program | Top-ranked program correct |
|---|---|---|
| train (never targeted by v14-v28 primitives) | 191 / 1000 | 94 (49%) |
| eval (where the new primitives were written) | 46 / 120 | 44 (96%) |
| public test (240 tasks, no labels) | 53 / 240 | unknown |

- **43 of the DSL families fire on exactly one eval task and on zero train tasks and zero public-test tasks.** They account for **42 of the 44** locally solved eval tasks. The other two eval solves come from `recolor_motif` and `serialize_snake_path_runs`, which also fire once on train.
- The "36.67% local eval" is therefore a measure of how many eval tasks were hand-fitted, not of generalization. On the 240 public test tasks these 43 families fire **0 times**; expected contribution to the hidden test is near zero.
- Held-out evidence of the generic DSL: it solves **98 / 1000 train tasks (9.8%)** and fits 53 / 240 public test tasks, 34 of them with non-cellular-automata programs.
- The generic cellular-automata fallback (`cellular_automatax3`) is low precision: it is the top-ranked program on 83 train tasks and correct on 6 (7%). On the public test it is the only fitting program for 19 of 53 tasks. Since a fitting program becomes attempt 1 and displaces the neural top guess, those 19 are probably net-negative. Non-CA top-1 programs are correct 88/108 (81%) on train.
- The `0.7900 (850/1076)` "train coverage" in `docs/experiments.md` is the neural Pass B 5-fold CV, not symbolic coverage. Symbolic train coverage is 98/1000.

## Classification

- **Generic (fire on many tasks, held-out accuracy high):** geometric transforms, gravity/rigid_gravity, crop_*, mirror/tile/scale, overlay_*split + remap_colors, fill_enclosed, connect_dots, color_objs_by_*, kronecker/kaleidoscope. Keep.
- **Generic but unreliable:** `cellular_automatax3` (7% precision), `classify_binary_shape` (1/8), `color_objs_by_size` (3/11). Rank below the neural guess or require stronger evidence (e.g. fit on leave-one-out).
- **Task-specific (1 eval task, 0 elsewhere):** the 43 families below. Each solves its source task and nothing else.

| Family | Eval task(s) it fires on | Solves it |
|---|---|---|
| `absorb_maze_particles_into_chamber` | dd6b8c4b | yes |
| `assemble_prototype_tile_macrogrid` | 65b59efc | yes |
| `compose_partitioned_rooms` | 7491f3cf | yes |
| `connect_room_tour_corridors` | 3e6067c3 | yes |
| `energize_conductive_circuit_path` | 7b0280bc | yes |
| `execute_abacus_bead_addition` | 9aaea919 | yes |
| `execute_turtle_bytecode_path` | 136b0064 | yes |
| `extrapolate_aligned_harmonic_combs` | 16de56c4 | yes |
| `extrapolate_periodic_optical_beams` | 53fb4810 | yes |
| `fill_cavities_from_legend` | dbff022c | yes |
| `fill_ranked_pillars_by_indicator` | 97d7923e | yes |
| `fire_container_nozzle_jets` | 3dc255db | yes |
| `generate_greek_key_labyrinth` | da515329 | yes |
| `generate_room_histogram` | 89565ca0 | yes |
| `heal_orthogonal_wire_detours` | 7b80bb43 | yes |
| `inpaint_fractal_tessellation` | b99e7126 | yes |
| `inpaint_symmetric_mandala_window` | 0934a4d8 | yes |
| `invert_apparatus_dual_polarity` | c7f57c3e | yes |
| `modulate_optical_pulse_interference` | 1ae2feb7 | yes |
| `operate_pin_tumbler_lock` | 4c3d4a41 | yes |
| `pairx4_hollow_solid_squares` | 2ba387bc | yes |
| `plot_incontext_histogram_legend` | 58490d8a | yes |
| `populate_hollow_boxes_from_tokens` | 8f215267 | yes |
| `power_circuit_enclosures` | 8b7bacbf | yes |
| `project_corner_bracket_templates` | 4c416de3 | yes |
| `pull_particles_into_tractor_containers` | 8b9c3697 | yes |
| `reflect_shapes_across_planar_mirrors` | 7ed72f31 | yes |
| `repair_periodic_pad2` | 135a2760 | yes |
| `replicate_isotropic_ruler_schedule` | c4d067a0 | yes |
| `replicate_satellite_constellation` | a395ee82 | yes |
| `restore_bilateral_symmetry` | 8e5c0c38 | yes |
| `scale_quadrant_dihedral_templates` | b10624e5 | yes |
| `simulate_diagonal_optical_mirrors` | db695cfb | yes |
| `simulate_liquid_gravity` | 28a6681f | yes |
| `simulate_projectile_cannons` | 409aa875 | yes |
| `slide_particles_by_legend_compass` | 88e364bc | yes |
| `slide_sokoban_block_in_channel` | 332f06d7 | yes |
| `solve_four_box_analogy_puzzle` | 4c7dc4dd | yes |
| `sort_vertical_pillars_ascending` | 31f7f899 | yes |
| `stamp_crosshair_palette_glyphs` | 247ef758 | yes |
| `standardize_parallel_lines` | e376de54 | yes |
| `transfer_circuit_tokens` | d35bdbdc | no |
| `transpose_and_stack_axis_blocks` | 291dc1e1 | yes |
## Caveats

- "Generalizes" is measured only on the labeled train and eval splits and on firing rate on the 240-task public test. The hidden test set is larger and different.
- Family names collapse parameter suffixes, so two variants of one primitive count as one family.
- Primitives are matched by program name, not by an explicit provenance record. Source tasks for the 43 families are inferred from where they fire.
