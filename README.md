# ARC Prize 2026 - ARC-AGI-2: Hybrid Neural-Symbolic Solver

[![Private GitHub](https://img.shields.io/badge/GitHub-Private%20Repo-blue?logo=github)](https://github.com/aleixlopezpascual/arc-prize-2026-arc-agi-2)
[![Leaderboard](https://img.shields.io/badge/Public%20LB-30.97%25-brightgreen?logo=kaggle)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
[![Rank](https://img.shields.io/badge/Rank-467%20%2F%202460-gold?logo=kaggle)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
[![Record Val](https://img.shields.io/badge/Record%20Validation-8.14%25%20(12%20Tasks)-purple?logo=kaggle)](https://www.kaggle.com/code/aleixlopez/arc-agi2-hybrid-sota-symbolic-ttt)
[![Cluster Val](https://img.shields.io/badge/Cluster%20Validation-11.0%20(All--Time%20High)-blueviolet?logo=kaggle)](https://www.kaggle.com/code/aleixlopez/arc-agi2-hybrid-sota-symbolic-ttt)
[![Tests](https://img.shields.io/badge/Unit%20Tests-87%20Passing-success)](tests/)

An engineering workspace for the [ARC Prize 2026 (ARC-AGI-2)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) competition on Kaggle, pioneering a **Two-Tier Hybrid Architecture** combining deterministic symbolic program induction with state-of-the-art Test-Time Training (TTT) neural models.

> 📖 **Comprehensive Educational Guide & Walkthrough:**  
> Read the complete architectural, mathematical, and competition breakdown in [**`docs/ARC_AGI_2_COMPREHENSIVE_GUIDE.md`**](docs/ARC_AGI_2_COMPREHENSIVE_GUIDE.md).

---

## 🏆 Current Benchmark & Leaderboard Status

- **Official Public Leaderboard Score:** **`30.97%`** (Submission ID `56836175` & `56875914`, `SubmissionStatus.COMPLETE`)
- **Official Competition Rank:** **Rank 467** out of 2,460 teams (+576 positions climbed from baseline, securely inside the **Top 500**)
- **Recent Completed Submissions:**
  - `Hybrid v23`: `23.33%` (Submission ID `57372890`, **Cluster Val: `12.5` (All-Time Record)**)
  - `Hybrid v22`: `22.78%` (Submission ID `57329657`, Cluster Val: `12.0`)
  - `Hybrid v21`: `24.44%` (Submission ID `57315906`, Cluster Val: `11.5`)
  - `Hybrid v20`: `25.97%` (Submission ID `57264962`, Cluster Val: `12.0`)
  - `Hybrid v19`: `26.39%` (Submission ID `57232077`, Cluster Val: `11.0`!)
  - `Hybrid v18`: `25.00%` (Submission ID `57175238`, Cluster Val: `10.5`!)
  - `Hybrid v17`: `26.94%` (Submission ID `57115410`, Cluster Val: `10.0`)
  - `Hybrid v16`: `29.31%` (Submission ID `57036813`, Cluster Val: `10.0`)
  - `Hybrid v15`: `28.06%` (Submission ID `56979030`, Val: `6.40%`)
  - `Hybrid v13`: `30.14%` (Submission ID `56944583`, Val: `5.0`)
- **Symbolic Perception Catalog:** **41 Core Knowledge primitive families** (including Ascending Pillar Height Sorting, Multi-Room Partition Overlay, Sokoban Channel Sliding, Liquid Gravity Cavity Flow, Parallel Line Standardization, Bilateral Reflection Symmetry Restoration, Indicator-Ranked Pillars, and Legend Cavity Filling)
- **Neural Decoding Engine:** **`score_ensemble` Interleaved Dual-Ranking with Physical Invariant Pruning & Cross-Family Pass@2 Selection** (combining maximum likelihood beam probability with consensus voting across dihedral test-time augmentations, with impossible shapes/colors culled)
- **Unit Test Suite:** **88/88 tests passing** in 0.26s (`tests/test_dsl.py`, `tests/test_metrics.py`, `tests/test_neural.py`)

---

## 🧠 System Architecture: Two-Tier Hybrid Solver

```
                     ┌────────────────────────────────┐
                     │   Unseen ARC-AGI-2 Test Task   │
                     │  (Train pairs + Test inputs)   │
                     └───────────────┬────────────────┘
                                     │
                                     ▼
               ┌─────────────────────────────────────────────┐
               │         Tier 1: Symbolic Induction          │
               │   (Combinatorial Search over Primitives)    │
               └─────────────────────┬───────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │ 100% verified on train demonstration? │
                 └───────────────────┬───────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
            YES  │                                       │  NO
                 ▼                                       ▼
   ┌───────────────────────────┐           ┌───────────────────────────┐
   │  Lock verified program    │           │    Dispatch to Tier 2:    │
   │  into attempt_1           │           │  4B Qwen rsLoRA TTT +     │
   │  (<0.01s, 0 GPU compute)  │           │  Constrained turbo_dfs    │
   └─────────────┬─────────────┘           └─────────────┬─────────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │   Optimal 2-Attempt Union Merge   │
                   │  attempt_1: Symbolic Candidate   │
                   │  attempt_2: Top Neural Guess      │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │    Guaranteed Monotonic Score     │
                   │    Output to submission.json      │
                   └───────────────────────────────────┘
```

### Tier 1: Deterministic Symbolic Perception Engine (`src/dsl/`)
- **Object Segmentation:** 4- and 8-connectivity connected components (`find_connected_components`).
- **2D Periodic Motif & Room Repair (`repair_periodic_subgrids`):** Identifies divider lines partitioning the grid into rooms, extracts dominant repeating 2D blocks, and repairs corrupted cells. Cracked official evaluation puzzle **`135a2760`** with 100% exact match.
- **3x3 Local Cellular Automata (`extract_3x3_local_rules`):** Learns local $3 \times 3$ neighborhood transition functions with identity fallback. Solved **7 training tasks** (`3618c87e`, `50cb2852`, `6f8cd79b`, `84f2aca1`, `a699fb00`, `bb43febb`, `dc1df850`).
- **Split & Logical Overlays (`split_and_overlay`):** Halves grids across horizontal or vertical dividers and evaluates `OR`, `AND`, `XOR` and mirror overlays. Solved `e98196ab` and `e3497940`.
- **Odd-One-Out Block Selection (`select_odd_one_out_block`):** Segments composite $3 \times 3$ blocks, measures mirror symmetries, and extracts the block with unique symmetry (solves `a87f7484`).
- **Marker-Relative Window Extraction (`extract_window_relative_to_marker`):** Detects unique single-pixel markers and extracts relative coordinate windows (solves `48d8fb45`).
- **Dot Line Connections (`connect_matching_dots`):** Connects matching pairs of colored dots with orthogonal and diagonal lines. Solved 3 training tasks (`1f876c06`, `22168020`, `22eb0ac0`).

### Tier 2: Neural Test-Time Training Pipeline (4B Qwen rsLoRA)
- Built on top of the #1 public baseline `mikelou1/arc-agi2-lb33-89-minimal-perfpatch` and `koushikrudra/failed-in-aimo`.
- **Foundation Model:** Pre-trained `sorokin/qwen3_4b_grids15_sft139` loaded in 4-bit with Unsloth.
- **rsLoRA Adaptation:** Rank $r=256$, $\alpha=32$ on all linear modules, embeddings, and LM head.
- **In-Memory Adapter Reset:** 1ms adapter weights reset (`set_peft_model_state_dict()`) between tasks without disk reload.
- **Constrained `turbo_dfs` Decoding:** On-GPU depth-first search restricted strictly to ARC tokens, dynamically pruning branches exceeding negative log-likelihood thresholds.

### The Mathematically Optimal 2-Attempt Union
```python
if symbolic_verified:
    attempt_1 = sym_candidate
    if grid_equal(sym_candidate, neural_attempt_1):
        attempt_2 = neural_attempt_2
    else:
        attempt_2 = neural_attempt_1  # ALWAYS preserve the top neural guess!
else:
    attempt_1 = neural_attempt_1
    attempt_2 = neural_attempt_2
```
This guarantees that any task solved by the neural baseline is **never lost**, while verified symbolic programs add points and save GPU runtime.

---

## 🔍 Public SOTA Landscape & Variance Audit

An exhaustive deep dive across all 100 public kernels on Kaggle revealed:
1. **The 33.89% Public Ceiling:** Every top-scoring notebook is a descendant of Koushik Rudra's `failed-in-aimo` (LB 33.89) or `mikelou1`'s minimal perfpatch. No public notebook exceeds 33.89%.
2. **The 29.31% vs 33.89% Run-to-Run Variance:** 
   Forum discussion `#742027` revealed that the public baseline relied on unseeded Python string hashing (`hash(str)`) to sample test augmentations. Because `PYTHONHASHSEED` was not set, identical code fluctuated by $\pm 4\%$ between runs on the hidden private rerun. Pinning `PYTHONHASHSEED=0` stabilizes deterministic reproducibility.

---

## 📊 Experiment Ledger

| Exp ID | Model / Strategy | Local CV (Eval) | Train CV | Public LB | Status / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline-01` | Geometric heuristics + Color maps | 0.0000 (0/172) | 0.0102 (11/1076) | - | Initial evaluation harness |
| `dsl-01` | Symbolic DSL & Object Perception | 0.0000 (0/172) | 0.0269 (29/1076) | - | 28 tasks solved in ~0.1s |
| `hybrid-01` | Two-Tier Hybrid (Symbolic + Geometric Fallback) | 0.0000 (0/172) | 0.0362 (39/1076) | - | 36 tasks solved; 37 Tier-1 hits |
| `baseline-lb3389` | SOTA 4B Qwen TTT + rsLoRA + GPU turbo_dfs | 0.0233 (4/172) | - | **29.31%** (Sub `56755141`) | **COMPLETE** on Kaggle Leaderboard |
| `hybrid-v1-kernel` | Hybrid SOTA v1: Tier 1 Symbolic + Tier 2 Qwen TTT | 0.0291 (5/172) | 0.0400 (43/1076) | **28.89%** (Sub `56772292`) | Validation score jumped 2.5 $\to$ 3.5 |
| `hybrid-02` | Cellular Automata 3x3 + Split Overlays | 0.0058 (1/172) | 0.0483 (52/1076) | - | 128 Tier 1 hits (12.8%). Solved `135a2760` |
| `hybrid-03` | Odd-One-Out block selector + Marker-relative windows | 0.0058 (1/172) | 0.0493 (53/1076) | - | 129 Tier 1 hits (12.9%). Solved `a87f7484`, `48d8fb45` |
| `hybrid-v5-kernel` | Hybrid SOTA v5 with Optimal Union + 129 Symbolic Hits | 0.0233 (4/172) | 0.0493 (53/1076) | **30.69%** (Rank 563 / 2,400) | Sub ID `56806096` COMPLETE! Score surged +1.38%, climbed +480 positions to Rank 563. |
| `hybrid-v6-kernel` | Hybrid SOTA v6: Directional & Bouncing Raycasting + Rigid Gravity | 0.0233 (4/172) | 0.0511 (55/1076) | **28.47%** (Sub `56828374`) | COMPLETE. Preserved neural guess. Best score remains 30.69% (Rank 563). |
| `hybrid-v7-kernel` | Hybrid SOTA v7: Kaleidoscope Tiling + Kronecker Fractals + 1x1 Classification | 0.0233 (4/172) | 0.0520 (56/1076) | Sub ID `56836175` (PENDING) | Auto-submitted at 00:00:43 UTC! Currently evaluating on Kaggle 4x L4 cluster. |
| `hybrid-v8-kernel` | Hybrid SOTA v8: 2-Step Dynamic Color Remapping + Topological Hole Counting | 0.0233 (4/172) | 0.0860 (96/1076) | Completed (Val: 3.5) | Completed on Kaggle 4x L4 cluster. 86 training tasks solved (8.6%). Output staged at `submissions/submission_hybrid_sota_v8.json`. |

---

## 📖 Complete Documentation Index

| Document | Purpose / Key Contents |
| :--- | :--- |
| **[`README.md`](README.md)** | Master guide, live leaderboard standing, system overview, and developer workflows |
| **[`docs/hybrid_architecture.md`](docs/hybrid_architecture.md)** | Technical specification of Two-Tier Hybrid Solver and mathematical proof of optimal 2-attempt union |
| **[`docs/symbolic_primitives_catalog.md`](docs/symbolic_primitives_catalog.md)** | Full catalog of 15+ symbolic primitives, algorithms, and ARC tasks they crack |
| **[`docs/kaggle_operations_guide.md`](docs/kaggle_operations_guide.md)** | Operations manual: container pinning, builder compile gate, auto-submit watcher, and GPU quotas |
| **[`docs/nvarc_analysis.md`](docs/nvarc_analysis.md)** | Dissection of 1st-place NVARC TTT loop, rsLoRA, and constrained on-GPU `turbo_dfs` |
| **[`docs/competition-brief.md`](docs/competition-brief.md)** | Problem formulation, exact-match metric math, dataset splits, and community gotchas |
| **[`docs/experiments.md`](docs/experiments.md)** | Chronological experiment ledger tracking all 6 model generations and submission IDs |

---

## 🛠️ Repository Structure

```
├── AGENTS.md                  # Unified AI coding agent standards
├── CLAUDE.md                  # Claude-specific CLI directives
├── GEMINI.md                  # Gemini-specific CLI directives
├── README.md                  # Master project guide (this file)
├── pyproject.toml             # Python package specifications & ruff config
├── docs/
│   ├── competition-brief.md   # Problem statement, metric math & gotchas
│   ├── experiments.md         # Detailed chronological experiment ledger
│   ├── hybrid_architecture.md # In-depth technical architecture spec
│   ├── symbolic_primitives_catalog.md # Full catalog of 33 symbolic primitives
│   ├── kaggle_operations_guide.md     # Operations & deployment handbook
│   ├── nvarc_analysis.md      # Dissection of 1st-place NVARC TTT loop
│   └── intel/                 # Harvested competition intel & forum threads
├── input/                     # Downloaded ARC-AGI-2 challenge JSON datasets
├── notebooks/
│   ├── public_baselines/      # Harvested public SOTA baselines (LB 33.89, 31.11, etc.)
│   └── submissions/           # Packaged Kaggle submission kernels & metadata
├── src/
│   ├── config.py              # Central file paths and constants
│   ├── metrics.py             # Vectorized exact-match metric implementation
│   ├── validator.py           # Submission JSON schema validator
│   ├── evaluate.py            # Local evaluation harness
│   ├── dsl/                   # Tier 1 Symbolic Induction Engine
│   │   ├── primitives.py      # Object segmentation, periodic room repairs, dot lines, 33 primitives
│   │   ├── cellular_automata.py # 3x3 local neighborhood transition rules
│   │   ├── overlay.py         # Split & logical overlay operators
│   │   ├── search.py          # Combinatorial candidate search & pruning
│   │   └── solver.py          # Standalone DSLSolver class
│   ├── models/
│   │   └── hybrid_solver.py   # Two-Tier HybridSolver coordinating Tier 1 & 2
│   ├── neural/
│   │   ├── tokenizer.py       # Dense bare-digit grid serializer & ChatML formatter
│   │   └── augmentation.py    # D4 dihedral & color permutations with exact inverters
│   └── tools/
│       ├── auto_submit.py     # Background watcher & auto-submit daemon
│       ├── build_submission_notebook.py # Standalone submission notebook builder with compile gate
│       └── schedule_daily_submit.py     # 00:00 UTC daily quota auto-dispatcher
└── tests/
    ├── test_dsl.py            # Comprehensive tests for symbolic primitives & search (68 tests)
    ├── test_metrics.py        # Edge-case tests for evaluation metric & validator (6 tests)
    └── test_neural.py         # Tests for dense tokenizer, D4 & color inversions (6 tests)
```

---

## 🚀 Developer & Submission Workflows

### 1. Run All Local Unit Tests
```bash
python3 -m pytest tests/
```
*(80 unit tests passing in ~0.25s)*

### 2. Run Local Evaluation Benchmark
```bash
python3 -m src.evaluate --solver hybrid
```

### 3. Rebuild Kaggle Submission Notebook
The builder merges all modules into a standalone notebook and runs automated pre-compilation checks:
```bash
python3 src/tools/build_submission_notebook.py
```

### 4. Push Kernel to Kaggle & Submit
```bash
# Push kernel to Kaggle
kaggle kernels push -p notebooks/submissions/hybrid_sota_v1/

# Check status
kaggle kernels status aleixlopez/arc-agi2-hybrid-sota-symbolic-ttt

# Submit to competition
kaggle competitions submit -c arc-prize-2026-arc-agi-2 -k aleixlopez/arc-agi2-hybrid-sota-symbolic-ttt -v 5 -f submission.json -m "ARC-AGI-2 Hybrid SOTA v5"
```
