# ARC Prize 2026 - ARC-AGI-2: Hybrid Neural-Symbolic & Dual-Pass TTT Solver

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-ARC%20Prize%202026-blue?logo=kaggle)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
[![Top Public LB](https://img.shields.io/badge/Top%20Public%20LB-31.39%25-brightgreen?logo=kaggle)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
[![Anchor Baseline](https://img.shields.io/badge/Anchor%20Baseline-30.97%25-blue?logo=kaggle)](https://www.kaggle.com/code/aleixlopez/arc-agi2-hybrid-sota-symbolic-ttt)
[![Unit Tests](https://img.shields.io/badge/Unit%20Tests-189%20Passing-success)](tests/)
[![Status](https://img.shields.io/badge/Competition%20Status-Complete%20(Dec%204%20Announcement)-orange)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An open-source competitive engineering and research workspace for the [ARC Prize 2026 (ARC-AGI-2)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) benchmark on Kaggle.

This repository implements, audits, and evaluates two complementary AI reasoning paradigms:
1. **The High-Ceiling Neural SOTA (31.39% Public LB):** An optimized, hardened dual-pass Test-Time Training (TTT) pipeline built on Qwen-4B BF16 with independent seed pooling (`seed=1` and `seed=137`), on-GPU constrained DFS logit slicing, and dynamic CLI batch mount resolution.
2. **The Two-Tier Hybrid Architecture (30.97% Anchor):** A deterministic Domain Specific Language (DSL) perception engine with combinatorial search over geometric primitives, integrated with neural TTT via an optimal 2-attempt monotonic union.

---

## 🏆 Final Competition Results & Official Standing

- **Top Official Submission:** **`31.39%`** (Submission ID `57844167`, `SubmissionStatus.COMPLETE`, `notebooks/boyko_dualpass_fork/`)
- **Anti-Shakeup Anchor Baseline:** **`30.97%`** (Submission IDs `56836175` & `56875914`, `notebooks/submissions/hybrid_sota_v1/`)
- **Official Competition Rank:** **Rank #807** out of 3,589 international teams (top tier, just 0.83 pts from the 32.22% Bronze medal cutoff).
- **Unit Test Suite:** **189 / 189 unit tests passing** (`python3 -m pytest tests/` in 2.36s).
- **Final Official Selections:** Designated under the **Barbell Portfolio Strategy** (Slot 1: Boyko Dual-Pass SOTA `31.39%`; Slot 2: Decoupled Hybrid Anchor `30.97%`). Official winner verification announced on **December 4, 2026**.

---

## 🔬 Core Methodological Post-Mortem: Overfitting vs. Generalization

A central scientific contribution of this workspace is the empirical analysis of **Goodhart's Law** when developing symbolic solvers for ARC:

```
           Handcrafted Eval Primitives              Unseen Hidden Test Generalization
                  (Local Score)                                (Public LB)
                     47.33%                                      23.06%
                 ▲ RECORD HIGH                               ▼ SEVERE DROP
```

### The $r = -0.86$ Anti-Correlation Finding
Between iterations v14 and v28, we expanded our DSL catalog to 76 primitive families. Local evaluation benchmark scores soared from `5.0%` to an all-time record of **`47.33%`**. However, upon submitting to Kaggle's hidden test set, leaderboard scores collapsed from **`30.97%` down to `23.06%`**.

A systematic audit (`docs/intel/local_vs_lb.md` and `docs/intel/primitive_audit.md`) revealed:
1. **Eval Set Contamination:** 43 of the 76 primitive families fired on exactly 1 evaluation puzzle and on 0 training or test puzzles. They were memorized task-specific algorithms, not general abstract priors.
2. **The Attempt-1 Displacement Trap:** When low-precision rules coincidentally matched the 2–3 training demonstrations of a test puzzle, they were assigned to `attempt_1`. This **displaced and cannibalized the valid #1 neural prediction** from Qwen 4B, directly eroding leaderboard accuracy.
3. **The Solution:** Stripping away overfit evaluation heuristics and restoring pure dual-pass neural TTT immediately restored peak performance, surging to our all-time best of **`31.39%`**.

---

## 🧠 System Architecture

### 1. Dual-Pass Neural Test-Time Training (TTT)
*(Implemented in `notebooks/boyko_dualpass_fork/`)*
- **Backbone Model:** Pretrained `Qwen3-4B-BF16` adapted via Unsloth.
- **Dual-Pass Independent Exploration:** 
  - **Pass A:** Evaluates candidate solutions using seed 1 with $N=8$ training augmentations and $N=1$ evaluation view.
  - **Pass B:** Repeats test-time training under seed 137 with an independent parameter initialization trajectory.
  - **Cross-Pass Logit Pooling:** Merges candidate beams across both passes by negative log-likelihood (NLL) confidence.
- **Constrained `turbo_dfs` Slicing:** Extracts and scores only the 12 ARC grid tokens directly in GPU memory, avoiding full vocabulary softmax overhead.
- **Hardened Resiliency Layer:** Automatically writes a valid identity-grid fallback `submission.json` before GPU compute begins to guarantee against timeout `0.00` scores.

### 2. Two-Tier Hybrid Architecture
*(Implemented in `src/dsl/` and `src/models/hybrid_solver.py`)*

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

---

## 📂 Repository Structure

```
├── AGENTS.md                  # Universal AI agent coding standards
├── CLAUDE.md                  # Claude Code CLI configuration
├── GEMINI.md                  # Gemini CLI configuration
├── README.md                  # Master repository documentation (this file)
├── HANDOVER.md                # Detailed technical briefing and session handover
├── pyproject.toml             # Python build configuration and dependencies
├── docs/
│   ├── FINAL_DAYS_PLAN.md     # Final 72-hour execution plan and post-mortem
│   ├── ARC_AGI_2_COMPREHENSIVE_GUIDE.md # Exhaustive educational ARC walkthrough
│   ├── hybrid_architecture.md # Formal specification of Two-Tier Hybrid Solver
│   ├── symbolic_primitives_catalog.md # Catalog of verified geometric primitives
│   ├── nvarc_analysis.md      # Dissection of 1st-place NVARC TTT loop
│   └── intel/                 # Harvested competition intelligence and forum audits
├── input/                     # ARC-AGI-2 evaluation datasets
├── notebooks/
│   ├── boyko_dualpass_fork/   # ⭐ Top 31.39% Kaggle submission kernel and builder
│   ├── submissions/           # Packaged hybrid submission kernels (v1 - v29)
│   └── public_baselines/      # Verified community reference baselines
├── src/
│   ├── config.py              # Global paths and configuration constants
│   ├── metrics.py             # Vectorized exact-match ARC-AGI-2 evaluation metric
│   ├── validator.py           # Submission JSON format and grid schema validator
│   ├── evaluate.py            # Local evaluation harness
│   ├── dsl/                   # Tier 1 Symbolic Perception Engine
│   │   ├── primitives.py      # Core knowledge primitives (geometry, symmetry, gravity)
│   │   ├── cellular_automata.py # Local 3x3 neighborhood transition rules
│   │   ├── overlay.py         # Boolean and split grid overlay operators
│   │   ├── search.py          # Combinatorial beam search engine
│   │   └── solver.py          # Standalone DSLSolver implementation
│   ├── models/
│   │   ├── hybrid_solver.py   # Two-Tier coordinator with optimal 2-attempt union
│   │   ├── diversity_reranker.py # Structural diversity reranker for attempt 2
│   │   └── baseline_heuristics.py # Fast geometric fallbacks
│   ├── neural/
│   │   ├── tokenizer.py       # Dense bare-digit grid serializer & ChatML formatter
│   │   └── augmentation.py    # D4 dihedral and color permutation group operations
│   └── tools/
│       ├── build_submission_notebook.py # Automated compilation gate for submission kernels
│       ├── heldout_proxy.py   # Unbiased A/B evaluation proxy
│       └── audit_primitives.py # Primitive generalization auditor
└── tests/                     # Comprehensive test suite (189 tests passing)
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/aleixlopezpascual/arc-prize-2026-arc-agi-2.git
cd arc-prize-2026-arc-agi-2

# Install dependencies (Python 3.11 or 3.12 recommended)
pip install -e .
```

### 2. Run the Test Suite
```bash
python3 -m pytest tests/
```
*(All 189 unit tests pass in ~2.4 seconds).*

### 3. Rebuild the Submission Kernels
To generate the standalone submission notebook with all DSL primitives and neural decoders compiled inline:
```bash
python3 src/tools/build_submission_notebook.py
```

### 4. Inspect or Run the Top-Scoring Boyko Dual-Pass Kernel
The top-performing **31.39%** kernel is fully contained in `notebooks/boyko_dualpass_fork/`:
- **Code:** `notebooks/boyko_dualpass_fork/submission_kernel.ipynb`
- **Metadata:** `notebooks/boyko_dualpass_fork/kernel-metadata.json`
- **Builder:** `python3 notebooks/boyko_dualpass_fork/build_fork.py`

To push and run directly on Kaggle:
```bash
kaggle kernels push -p notebooks/boyko_dualpass_fork/
```

---

## 📜 License & Citation

This repository is licensed under the [MIT License](LICENSE).  
The ARC-AGI-2 challenge dataset is provided by the [Abstraction and Reasoning Corpus Prize](https://arcprize.org/).

If you find this work, our post-mortem analysis, or the dual-pass TTT architecture helpful in your research, please cite:
```bibtex
@misc{lopez2026arcprize,
  author = {Aleix L{\'o}pez},
  title = {ARC Prize 2026 - ARC-AGI-2: Hybrid Neural-Symbolic & Dual-Pass TTT Solver},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/aleixlopezpascual/arc-prize-2026-arc-agi-2}}
}
```
