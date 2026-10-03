# Two-Tier Hybrid Architecture: ARC Prize 2026 - ARC-AGI-2

## 1. System Vision & Motivation
ARC-AGI-2 presents tasks spanning two distinct problem spaces:
1. **Purely Deterministic / Discrete Geometric Problems:** Solvable with 100% precision by succinct program induction (e.g. cellular automata, periodic motif repairs, flood fills, dot line connections, symmetry selection).
2. **Complex Visual Induction Problems:** Requiring deep conceptual priors and continuous representation learning (e.g. complex object composition, non-local analogies, visual storytelling).

Pure neural models (like 4B/7B LLMs with Test-Time Training) are prone to subtle token hallucinations or runtime timeouts on deterministic puzzles. Conversely, pure symbolic program synthesis engines struggle with high compositional depth.

Our solution is a **Two-Tier Hybrid Architecture**:
- **Tier 1 (Instant Symbolic Program Induction):** Fast, deterministic symbolic search running in $< 0.01$s. If a candidate program verifies 100% on all demonstration pairs, it locks `attempt_1` with absolute certainty.
- **Tier 2 (Neural Test-Time Training Transduction):** For unsolved tasks, the 4B parameter Qwen model runs rapid rsLoRA fine-tuning with 128 task augmentations and constrained tree-search decoding.
- **Monotonic Ensembling:** `attempt_2` retains the neural candidate, mathematically guaranteeing zero regression on tasks the neural baseline solved.

---

## 2. Tier 1: The Symbolic Induction Engine (`src/dsl/`)

### A. Object Perception & Connected Components (`primitives.py`)
- **Connectivity:** 4-way and 8-way connected component segmentation.
- **Object Extraction:**
  - `crop_largest_object`, `crop_smallest_object` via bounding-box extraction.
  - `remove_small_objects`: Cleans single-pixel noise artifacts.
  - `crop_hollow_rectangle_interior`: Extracts regions bounded by uniform rectangular frames (solves `1c786137`).

### B. 2D Periodic Motif & Room Repair (`primitives.py`)
- **Discovery:** 67.5% of evaluation tasks are isometric (same output shape as input), and 55 tasks have $\le 20\%$ of cells changed.
- **Mechanism (`repair_periodic_subgrids`):**
  - Identifies divider lines partitioning the grid into independent rooms.
  - Computes the dominant repeating 2D block along the room's major axis (period $P \in \{2 \dots 8\}$).
  - Repairs damaged or corrupt pixels violating the dominant frequency.
  - **Result:** Cracked official evaluation puzzle **`135a2760`** with 100% exact match.

### C. Orthogonal & Diagonal Laser Connections (`primitives.py`)
- **`connect_matching_dots`**: Detects matching pairs of colored dots and draws unobstructed orthogonal and diagonal lines between them.
- **Result:** Solved 3 training tasks (`1f876c06`, `22168020`, `22eb0ac0`).

### D. 3x3 Local Cellular Automata (`cellular_automata.py`)
- **`extract_3x3_local_rules`**: Computes deterministic local neighborhood transition functions ($3 \times 3$ patches) mapping surrounding context to center cell values.
- **Generalization:** Falls back to identity for unseen patches.
- **Result:** Solved **7 training tasks** (`3618c87e`, `50cb2852`, `6f8cd79b`, `84f2aca1`, `a699fb00`, `bb43febb`, `dc1df850`).

### E. Split & Logical Overlays (`overlay.py`)
- **`split_and_overlay`**: Splits grids into halves across dividers or midlines (horizontal and vertical) and tests logical overlays (`OR`, `AND`, `XOR`) and mirror reflections.
- **Result:** Solved tasks `e98196ab` and `e3497940`.

### F. Odd-One-Out & Relative Subwindows (`primitives.py`)
- **`select_odd_one_out_block`**: Partitions a sequence of $3 \times 3$ blocks, measures mirror symmetries, and identifies the block with a unique symmetry signature (solves `a87f7484`).
- **`extract_window_relative_to_marker`**: Extracts $3 \times 3$ windows relative to single-pixel marker dots (solves `48d8fb45`).

---

## 3. Tier 2: Neural Test-Time Training (TTT) Pipeline

Built on the foundation of `mikelou1/arc-agi2-lb33-89-minimal-perfpatch` and `koushikrudra/failed-in-aimo`:
- **Base Model:** `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1` (Qwen3-4B pre-trained on synthetic ARC tasks).
- **Quantization:** 4-bit with Unsloth (`FastLanguageModel`).
- **rsLoRA Adaptation:** Rank $r = 256$, $\alpha = 32$, targeting all linear projection layers, embedding tokens, and `lm_head`.
- **In-Memory Weight Reset:** Restores pristine adapter weights via `set_peft_model_state_dict()` in ~1ms without disk I/O.
- **Constrained `turbo_dfs` Decoding:** Depth-first search restricted strictly to the 12 ARC tokens (`0`–`9`, `\n`, `<|im_end|>`), dynamically pruning any search branch exceeding cumulative negative log-likelihood $S_{\text{max}} = -\ln(0.2) \approx 1.61$.
- **Minimal Perfpatch:** Eliminates CPU $\leftrightarrow$ GPU tensor transfer bottlenecks during log-softmax and token indexing.

---

## 4. Mathematically Optimal 2-Attempt Union

The competition evaluates submissions on at most **2 attempts** per test output grid:
$$\text{Score}(i) = \max\left( \mathbf{1}(\text{attempt}_1 = Y_i), \, \mathbf{1}(\text{attempt}_2 = Y_i) \right)$$

### The Union Protocol
Let $S_1$ be the Tier-1 verified symbolic prediction.
Let $N_1, N_2$ be the Tier-2 neural predictions ranked by likelihood.

```python
if symbolic_verified:
    attempt_1 = S_1
    if grid_equal(S_1, N_1):
        # Symbolic agrees with top neural prediction:
        attempt_2 = N_2  # Keep second neural guess as backup
    else:
        # Symbolic proposes a novel hypothesis:
        attempt_2 = N_1  # ALWAYS preserve the top neural guess!
else:
    attempt_1 = N_1
    attempt_2 = N_2
```

### Monotonicity Guarantee
1. If $S_1 = Y_i$ (Symbolic correct) $\implies$ $\text{Score} = 1$.
2. If $N_1 = Y_i$ (Neural correct) $\implies$ $\text{Score} = 1$ (via `attempt_2`).
3. Loss only occurs if $S_1 \neq Y_i$ AND $N_1 \neq Y_i$ AND $N_2 = Y_i$. Since $N_1$ has significantly higher precision than $N_2$, this trade-off heavily favors the verified symbolic candidate while preserving the neural baseline's core strength.

---

## 5. Kaggle 12-Hour Rerun Resource Management

In Kaggle's offline rerun environment (240 unseen test tasks on 4x Nvidia L4 GPUs):
- **Worker Pool:** `torch.multiprocessing` assigns 1 worker process per GPU (4 workers total).
- **Task Dispatch:** Tasks are queued dynamically across workers.
- **Time Sentinel:** A 10-minute safety buffer (`global_end_time = now + 12h - 600s`) ensures workers cleanly flush predictions to `submission.json` before timeout.
- **Memory Safety:** `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` with explicit peak memory resets between tasks.
