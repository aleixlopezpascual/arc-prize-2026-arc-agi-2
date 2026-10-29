# Comprehensive Audit of Unconsidered & Recent Public Notebooks (October 30, 2026)

**Audit Date:** October 30, 2026  
**Target Benchmark:** ARC Prize 2026 (ARC-AGI-2)  
**Total Unique Kernels Analyzed:** 248  
**Unconsidered Kernels Evaluated:** 211  
**High-Value Kernels Downloaded & Archived:** 5 (in `notebooks/public_baselines/`)

---

## 1. Executive Summary & Strategic Landscape

Following an exhaustive automated query of all public notebooks on Kaggle across multiple sorting dimensions (`scoreDescending`, `dateRun`, `dateCreated`, `voteCount`, and `hotness`), we identified and evaluated the complete corpus of public approaches that were previously unconsidered in this repository.

The state of the art in the community has bifurcated into two major development tracks:
1. **High-Performance Neuro-Symbolic Hybrids:** Combining CPU-based verified geometric and symbolic solvers (fast execution, 0 GPU load, deterministic accuracy) with test-time trained (TTT) dual-pass LLM ensembles (Qwen 4B / Qwen 3.8 backbones with 13-token vocabulary slicing and Unsloth LoRA).
2. **Pure Symbolic & Cognitive Engines:** Novel non-neural approaches, most notably the implementation of Jeff Hawkins's **Thousand Brains Theory (Monty-ARC)** featuring 8 independent sensorimotor learning modules, and **Factored Topological Synthesizers** using Minimum Description Length (MDL) beam search over 44 primitives.

Five particularly high-value notebooks have been downloaded and archived into `notebooks/public_baselines/` for codebase inspection and algorithmic integration.

---

## 2. High-Value Downloaded Baselines Matrix

| Directory in `notebooks/public_baselines/` | Source Ref | Author | Date | Votes | Key Methodology / Architecture |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `monty_arc/` | `rahim3/monty-arc-thousand-brains-arc-agi-2` | rahim3 | 2026-10-26 | 1 | **Thousand-Brains Theory Engine:** 8 sensorimotor modules (`periodic`, `association`, `field`, `pattern`, `symmetry`, `behavior`, `dynamics`, `composition`) with honest leave-one-out (LOO) validation and 2:1 consensus voting. |
| `symbolic_synthesizer_v2/` | `vigneshalugunulla/arc-prize-2026-symbolic-synthesizer-v2` | Vigneshwara K. Alugunulla | 2026-10-26 | 0 | **44-Primitive Factored DSL:** 6 algebraic families (dihedral framing, CC object graphs, panel Boolean algebra, topological flood/cavity fill, gravity/ray-casting, symmetry/tiling) with MDL ranking. |
| `verified_geometry_hybrid/` | `ayodejiibrahimlateef/arc2-v54-verified-geometry-hybrid-l4` | Ayodeji | 2026-07-23 | 20 | **High-LB Verified Geometry Hybrid:** CPU border-ray obstacle propagation, masked reflection recovery, panel overlays, combined with 4x L4 Qwen 4B LoRA fine-tuning. |
| `qwen_symbolic_fallback/` | `medvax/arc-agi-2-qwen-ttt-with-symbolic-fallback` | medvax | 2026-10-26 | 5 | **Two-Pass Grid Adaptation + Geometry:** Bounded CPU geometry pass (border-ray, mandala completion) short-circuiting a dual-pass Qwen TTT pipeline with anchored primary selections. |
| `priyalmantri_neuro_symbolic/` | `priyalmantri/arc-agi-2-neuro-symbolic-solver-with-qwen` | Priyal Mantri | 2026-10-26 | 0 | **Object Scene Graph + Retrieval Memory:** 4-connectivity `PerceptionEngine`, `SceneGraphBuilder`, `MirrorMatrixMemory` (indexing by shape delta), and Qwen 2.5 Coder hypothesis generation. |

---

## 3. Deep-Dive Technical Breakdowns

### 3.1 Rahim3: *Monty-ARC — Thousand Brains Theory for ARC-AGI-2*
- **Ref:** `rahim3/monty-arc-thousand-brains-arc-agi-2`
- **Location:** `notebooks/public_baselines/monty_arc/monty-arc-thousand-brains-arc-agi-2.py`
- **Theoretical Basis:** Implements the Thousand Brains theory of intelligence (Hawkins et al., Numenta). Rather than learning a single end-to-end mapping, intelligence is modeled as thousands of cortical columns running independent sensorimotor reference-frame models that vote on the state of the world.
- **Architectural Mechanics:**
  - **`periodic.py` (Toroidal Reference Frames):** Models a grid region wrapping around with period $(p_y, p_x)$. Cells are predicted by the majority color of their phase `((y % py) * px + (x % px))`. Supports two modes:
    - `repair`: Overwrites conflicting cells with the phase majority (solves corrupted periodic patterns).
    - `extend`: Fills background holes using extrapolated phase predictions.
    - Evaluated across 3 spatial scopes: the entire grid, panels between uniform separator lines, and interiors of rectangular rings.
  - **`association.py` (In-Task Legend & Template Mapping):** Identifies compact 2-cell color keys (legends) or template shapes under $D_8$ poses to recolor or fill enclosed cavities.
  - **`field.py` (Coordinate Field Functions):** Expresses output color as a function of coordinate displacement $(\Delta y, \Delta x)$ anchored on a unique single-cell marker. Displacement codes include `row`, `col`, `ring`, `manhattan`, `diag`, `anti`, and periodic variants `(code % period)` with $k \in \{2..10\}$.
  - **`pattern.py` (Cluttered Object Recognition):** Discovers changed cell clusters, normalizes them under $D_8$, and applies them via sliding-window matching with wildcard (-1) support.
  - **`vote.py` (Evidence Pooling & Consensus):**
    - Every hypothesis must achieve 100% training pair reconstruction to enter the candidate pool.
    - Honest leave-one-out (LOO) evidence validation scores generalizability.
    - **Consensus Rule:** The top candidate is selected only if its evidence $\ge 0.99$ and it out-votes any conflicting hypothesis by at least 2:1 in terms of distinct learning modules agreeing on the exact output grid.
- **Strategic Value for Our Pipeline:**
  - The `periodic.py` toroidal phase repair and `field.py` anchor-displacement functions represent clean, mathematically rigorous primitives that expand our Tier-1 Symbolic Induction Engine with zero GPU compute overhead.

---

### 3.2 Vigneshwara K. Alugunulla: *Factored Topological Program Synthesizer (v2.0)*
- **Ref:** `vigneshalugunulla/arc-prize-2026-symbolic-synthesizer-v2`
- **Location:** `notebooks/public_baselines/symbolic_synthesizer_v2/arc-prize-2026-symbolic-synthesizer-v2.py`
- **Architectural Mechanics:**
  - Implements a 44-primitive Domain-Specific Language (DSL) structured into 6 orthogonal algebraic families:
    1. *Dihedral $D_4$ & Framing:* Identity, 90°/180°/270° rotations, horizontal/vertical reflections, diagonal transpositions, non-background bounding box crop, uniform border padding.
    2. *Connected-Component Graphs:* 4- and 8-connectivity labeling, filtering by color/area/aspect-ratio, odd-one-out shape/color discrimination, scale-invariant component matching, noise removal.
    3. *Divider-Line Grid Partitioning:* Automated detection of single-color horizontal/vertical grid separators, cutting into $M \times N$ cell blocks, followed by panelwise Boolean logic (`AND`, `OR`, `XOR`, `NOR`).
    4. *Topological Morphology:* Topological flood fill, inner/outer boundary contour extraction, cavity/enclosure filling (distinguishing fully enclosed background cavities from concave border indentations).
    5. *Physical Sedimentation & Ray-Casting:* 4-directional gravity simulation with rigid vs granular object physics, collinear ray shooting, connecting colored vertices with straight/diagonal lines.
    6. *Symmetry & Mappings:* Radial symmetry completion, reflection completion across horizontal/vertical midlines, multi-scale tile repetition, and functional color permutation induction.
  - **Search Engine:** Depth-1 exhaustive search ($\sim 0.05$s) $\to$ Depth-2 composition search ($\sim 0.2$s) $\to$ Heuristic Depth-3 beam search guided by Minimum Description Length (MDL) and demonstration consistency.
- **Strategic Value for Our Pipeline:**
  - The panel Boolean algebra (`XOR`, `AND`, `OR`, `NOR`) over sub-partitioned divider grids and cavity-filling algorithms directly target common failure modes observed in ARC-AGI-2 evaluation puzzles.

---

### 3.3 Ayodeji & Medvax: *Verified Geometry Hybrids with Qwen TTT*
- **Refs:** `ayodejiibrahimlateef/arc2-v54-verified-geometry-hybrid-l4` and `medvax/arc-agi-2-qwen-ttt-with-symbolic-fallback`
- **Locations:** `notebooks/public_baselines/verified_geometry_hybrid/` and `notebooks/public_baselines/qwen_symbolic_fallback/`
- **Architectural Mechanics:**
  - **Border-Ray Obstacle Propagation:** Traces rays projecting from border boundary markers inward across the grid until an obstacle color is encountered, painting the ray path.
  - **Masked Reflection Recovery:** Detects when a solid single-color rectangular patch is occluding a symmetric grid, identifies the symmetry axis (horizontal, vertical, or diagonal), and reconstructs the occluded pixels from the opposite unoccluded quadrant.
  - **Hybrid Routing Policy:**
    1. CPU Geometry Engine evaluates all tasks first under a strict per-task deadline ($<0.5$s).
    2. If a geometric program perfectly fits 100% of the training demonstrations and passes shape validation, its test prediction is emitted immediately.
    3. Remaining tasks are routed to the 4x L4 GPU pool for dual-pass Qwen LoRA test-time adaptation.
  - **Anchored Ensembling (`anchored(primary, pooled)`):**
    - Rather than blindly averaging predictions across passes, the primary high-confidence guess is preserved as `attempt_1`.
    - `attempt_2` is populated from the diverse secondary candidate pool only if it differs from `attempt_1`, preventing redundant slot wastage.

---

### 3.4 ChristopherDAleman: *NVARC + TRM Asymmetric GPU Portfolio*
- **Ref:** `christopherdaleman/arc-2026-nvarc-trm-evidence-cost-v1`
- **Architectural Mechanics:**
  - Addresses the 4x L4 GPU resource allocation bottleneck by breaking GPU symmetry:
    - GPUs 0, 1, and 2 immediately launch NVARC LoRA workers to process the cheapest tasks.
    - GPU 3 is dedicated to training a Tiny Recursive Model (TRM) through 4,000 iterations.
    - After TRM checkpoints are exported, GPU 3 joins GPUs 0-2 as a 4th NVARC worker for the remaining queue.
  - Demonstrates aggressive evidence-based cost ordering where tasks with smaller spatial grids and fewer demonstration pairs are prioritized.

---

### 3.5 Priyal Mantri: *Object Scene Graph & Retrieval Memory with Qwen*
- **Ref:** `priyalmantri/arc-agi-2-neuro-symbolic-solver-with-qwen`
- **Location:** `notebooks/public_baselines/priyalmantri_neuro_symbolic/arc-agi-2-neuro-symbolic-solver-with-qwen.ipynb`
- **Architectural Mechanics:**
  - Explicitly structures ARC tasks into formal scene graphs: objects (connected components) with spatial relations (`contains`, `adjacent`, `aligned_h`, `aligned_v`, `centroid`).
  - Implements a `MirrorMatrixMemory` associative cache keyed by transformation shape deltas $(\Delta H, \Delta W)$ to retrieve proven candidate program templates.
  - Uses Qwen 2.5 Coder to synthesize Python programs directly conditioned on the extracted scene-graph facts when standard DSL search fails.

---

## 4. Key Lessons & Integration Roadmap for Our Repository

1. **Toroidal Periodic Induction & Coordinate Fields:**
   - Incorporate the lightweight periodic phase-repair algorithm from `monty_arc/periodic.py` and coordinate anchor field mappings from `monty_arc/field.py` into `src/dsl/primitives.py`.
2. **Panel Boolean Algebra:**
   - Adopt the sub-grid divider extraction and panelwise Boolean algebra (`XOR`, `AND`, `OR`) from `symbolic_synthesizer_v2.py`.
3. **Occluded Mandala/Symmetric Inpainting:**
   - Our Primitive 74 (`inpaint_symmetric_mandala_window`) already solves mandala tasks; the border-ray obstacle propagation from `ayodeji` provides a complementary mechanism for ray-casting tasks.
4. **Anchored Diversity Selection:**
   - The `anchored(primary, pooled)` selection rule in `medvax` aligns with our `Attempt 2 Structural Diversity Reranker` design, validating our direction of preserving top confidence while diversifying attempt 2.
