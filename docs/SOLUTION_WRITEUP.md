# [Solution Writeup] 31.39% Public LB — Dual-Pass Neural TTT & Two-Tier Hybrid Architecture

**Competition:** [Kaggle ARC Prize 2026 (ARC-AGI-2)](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)  
**Author:** Aleix López  
**Repository:** [https://github.com/aleixlopezpascual/arc-prize-2026-arc-agi-2](https://github.com/aleixlopezpascual/arc-prize-2026-arc-agi-2)  
**Top Public Leaderboard Score:** **`31.39%`** (Rank #807 / 3,589 teams, 0.83 pts from Bronze)  
**Anti-Shakeup Anchor Score:** **`30.97%`** (Versions 7 & 8)  
**Selected Portfolio:** Barbell Strategy (Slot 1: Boyko Dual-Pass SOTA `31.39%`; Slot 2: Decoupled Hybrid Anchor `30.97%`)

---

## 1. Executive Summary

This writeup presents our technical solution and post-mortem for the **ARC Prize 2026 (ARC-AGI-2)** competition on Kaggle.

Our work explored two distinct reasoning paradigms under tight 12-hour GPU compute constraints:
1. **The High-Ceiling Neural SOTA (31.39% Public LB):** A hardened, dual-pass Test-Time Training (TTT) pipeline built on Qwen-4B BF16 with independent seed exploration (`seed=1` and `seed=137`), on-GPU constrained DFS logit slicing, and dynamic CLI batch mount resolution.
2. **The Two-Tier Hybrid Architecture (30.97% Anchor):** A deterministic Domain Specific Language (DSL) perception engine with combinatorial search over geometric primitives, integrated with neural TTT via an optimal 2-attempt monotonic union.

In addition to our competitive results, we share an empirical post-mortem documenting **Goodhart's Law in abstract reasoning**: how expanding handcrafted symbolic primitives inflated local evaluation benchmarks from `5.0%` to a record `47.33%`, while causing public leaderboard scores to collapse to `23.06%` ($r = -0.86$ anti-correlation) through neural candidate displacement.

---

## 2. High-Ceiling Neural SOTA: Dual-Pass Test-Time Training (31.39%)

The foundation of our peak scoring submission (`notebooks/boyko_dualpass_fork/`, Submission ID `57844167`) is an optimized fork of the Konstantin Boyko dual-pass NVARC architecture.

```
                    Unseen ARC-AGI-2 Task (Demos + Test Input)
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 │                                           │
                 ▼                                           ▼
       ┌───────────────────┐                       ┌───────────────────┐
       │      Pass A       │                       │      Pass B       │
       │  Qwen-4B rsLoRA   │                       │  Qwen-4B rsLoRA   │
       │  (Init Seed = 1)  │                       │ (Init Seed = 137) │
       │  8 Train Augs     │                       │  8 Train Augs     │
       └─────────┬─────────┘                       └─────────┬─────────┘
                 │                                           │
                 ▼                                           ▼
       ┌───────────────────┐                       ┌───────────────────┐
       │   turbo_dfs (A)   │                       │   turbo_dfs (B)   │
       │ (12 ARC Logits)   │                       │ (12 ARC Logits)   │
       └─────────┬─────────┘                       └─────────┬─────────┘
                 │                                           │
                 └─────────────────────┬─────────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  Cross-Pass Candidate Pooling │
                       │   (NLL Confidence Reranking)  │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │   submission.json (2 Guesses) │
                       │    Attempt 1: Top Pooled Beam │
                       │    Attempt 2: Second Best     │
                       └───────────────────────────────┘
```

### Key Technical Components:

#### A. Dual-Pass Independent Optimization
Single-pass fine-tuning often converges into narrow local minima depending on the augmentation sampling order. We execute two full, independent optimization and decoding trajectories per task:
- **Pass A:** Trained on $N=8$ dihedral demonstrations using `seed=1` and evaluated with $N=1$ view.
- **Pass B:** Re-initializes adapters and optimizes under `seed=137` with an independent parameter trajectory.
- **Pooling Policy:** Candidates from both passes are pooled by negative log-likelihood (NLL). If Pass A's top candidate is absent from Pass B, it is explicitly preserved in attempt 2, maximizing 2-attempt coverage across distinct basins of attraction.

#### B. On-GPU Constrained `turbo_dfs` Logit Slicing
Standard auto-regressive generation over the full 152,000-token Qwen vocabulary wastes memory bandwidth on irrelevant text tokens. 
- We slice logits strictly to the 12 ARC tokens:
  $$\text{ARC\_TOKENS} = \{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, \text{\textbackslash n}, \text{<\|im\_end\|>}\}$$
- The negative log-likelihood is computed directly on GPU:
  $$\text{NLL}_i = \text{scores}_i + \text{LogSumExp}(\text{logits}) - \text{logits}[\text{ARC\_TOKENS}]$$
- Candidates with scores exceeding a dynamic threshold ($\text{max\_score} = -\log(0.2)$) are pruned early in the depth-first search tree.

#### C. In-Memory LoRA State Reset (1ms)
Reloading model weights from disk between tasks causes severe I/O bottlenecks. We maintain base model weights frozen in memory and perform an in-place tensor clone of the initial LoRA state dict (`set_peft_model_state_dict()`), resetting the adapter in $\approx 1\text{ ms}$ per puzzle.

#### D. Hardened Pre-Execution Fallback Layer
In Kaggle hidden evaluations, container timeouts or out-of-memory errors produce an instant `0.00` score. Before initiating heavy GPU compute, our pipeline dynamically reads the challenge inputs and writes a valid, complete `submission.json` where `attempt_1` and `attempt_2` are identity input grids. If the kernel is killed prematurely by Kaggle's wall clock, identity predictions still score non-zero and prevent disqualification.

---

## 3. Two-Tier Hybrid Architecture: The 30.97% Anti-Shakeup Anchor

Our secondary designated baseline (`notebooks/submissions/hybrid_sota_v1/`, Submission ID `56875914`) implements a **Two-Tier Hybrid Engine** designed to combine high-precision deterministic synthesis with neural generation.

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
                  │    Output to submission.json      │
                  └───────────────────────────────────┘
```

### Core Primitives & Perception Modules (`src/dsl/`)
- **Connected Component Segmentation:** 4- and 8-connectivity object isolation with background subtraction.
- **Symmetry & Dihedral Invariance:** Reflection, rotation, transposed projection, and kaleidoscope tiling.
- **Periodic Room & Grid Repair (`repair_periodic_subgrids`):** Detects divider bars, partitions grids into sub-rooms, extracts dominant 2D repeating tiles, and restores corrupted cells (solved official eval task `135a2760` with 100% accuracy).
- **Boolean & Division Overlays:** Horizontal/vertical half-grid folding under `AND`, `OR`, `XOR`, and inverted color operations.
- **Gravity & Cavity Infilling:** Directional particle flow, rigid block dropping, and enclosed cavity flood filling.

### The Mathematically Optimal 2-Attempt Union
To merge symbolic and neural predictions without losing neural coverage:
```python
if symbolic_verified:
    attempt_1 = symbolic_candidate
    if grid_equal(symbolic_candidate, neural_attempt_1):
        attempt_2 = neural_attempt_2
    else:
        attempt_2 = neural_attempt_1  # ALWAYS preserve the primary neural guess!
else:
    attempt_1 = neural_attempt_1
    attempt_2 = neural_attempt_2
```

---

## 4. What Didn't Work: The $r = -0.86$ Overfitting Discovery

The most instructive finding of our campaign occurred between versions **v14 and v28**:

```
Iteration       Local Eval Benchmark       Public Leaderboard Score
v7 / v8                 3.8%                        30.97%
v16                    10.0%                        29.31%
v20                    12.0%                        25.97%
v25                    14.0%                        23.47%
v28                    47.33% (Record!)             23.06% (Severe drop!)
```

### Root Cause Analysis:

#### 1. The Eval-Set Memorization Trap
We audited our expanded catalog of 76 primitive families across 1,360 total tasks (1,000 train, 120 eval, 240 public test).
- **43 of the 76 primitive families** solved **exactly 1 evaluation task**, and fired **0 times on train tasks and 0 times on test tasks** (e.g., `generate_greek_key_labyrinth`, `execute_turtle_bytecode_path`, `absorb_maze_particles_into_chamber`).
- They accounted for **42 of the 44 local evaluation solves**.
- The `47.33%` local score was an illusion of progress created by handcrafting specialized algorithms for visible evaluation examples.

#### 2. The Attempt-1 Displacement Trap (Neural Cannibalization)
Because our policy assigned any verified demonstration-matching program to `attempt_1`:
- Several generic fallback primitives (such as `cellular_automatax3`) matched the 2–3 training examples by coincidence, but had only a **7% precision** on held-out test grids.
- Whenever a low-precision symbolic candidate coincidentally matched the demonstrations, it **displaced the true #1 prediction of the neural model**. The neural network had genuine generalization ability; overriding it with coincidentally-fitted symbolic grids cannibalized valid neural solves.

#### 3. Pearson Correlation: $r = -0.86$
Across versions v5 through v24, the correlation between local evaluation score and public leaderboard score was **$r = -0.86$** (Spearman $\rho = -0.88$). Every "+1.0%" gained in local benchmark accuracy corresponded to a **-0.88 pt drop** on the true hidden test set.

**The Fix:** Stripping out overfit primitives and restoring pure dual-pass neural TTT immediately drove our score to our all-time high of **`31.39%`**.

---

## 5. Endgame Operations: The Barbell Portfolio Strategy

In the final 48 hours of the competition, high GPU contention on Kaggle's 4x L4 cluster caused hidden test rerun queues to swell to **30+ hours**.

To protect against private leaderboard shakeup, we implemented the **Barbell Portfolio Strategy** for our 2 final designated submissions:

| Selection Slot | Candidate | Score Profile | Strategic Purpose |
|---|---|---|---|
| **Slot 1 (High-Ceiling SOTA)** | **Boyko Dual-Pass Fork v3** (`57844167`) | **31.39%** Public LB | Maximizes medal potential (Gold/Silver/Bronze) if private distribution mirrors public test. |
| **Slot 2 (Anti-Shakeup Anchor)** | **Hybrid SOTA v8** (`56875914`) | **30.97%** Public LB | Completely decoupled architecture (single-pass Qwen + geometric priors); immune to dual-pass correlation traps. |

### Golden Rules Followed:
- **Zero Unscored Selections:** Never select a submission that is `PENDING` or `ERROR` (unscored runs receive `0.00`).
- **Explicit UI Selection:** Manually locked in checkboxes on the Kaggle interface rather than relying on auto-selection.

---

## 6. Key Takeaways for Future ARC Competitions

1. **Test-Time Training (TTT) is Essential for Novel Generalization:** Pure frozen prompt-based LLMs struggle with ARC's novel geometries. Dynamic parameter adaptation on task demonstrations remains the strongest baseline.
2. **Beware of Inverted Validation Loops:** Developing symbolic primitives while looking at evaluation tasks almost inevitably leads to memorization. Primitives must be strictly evaluated on large held-out training splits (1,000+ tasks) using leave-one-out cross-validation.
3. **Consensus & Multi-Seed Exploration Outperforms Complex Heuristics:** Simply running two independent optimization trajectories (`seed=1` and `seed=137`) and pooling candidate logits by confidence provided a larger, cleaner score boost (+1.5–2.0%) than dozens of handcrafted algorithmic rules.
4. **Engineering Resiliency Wins Competitions:** Pre-execution fallback identity grids and dynamic mount resolvers prevented silent failures that disqualified numerous competing teams during queue congestion.

---

## 7. Open Source Code & Reproducibility

The complete repository, test suite (189 passing tests), and submission kernel builders are open-sourced under the MIT License:
- **GitHub Repository:** [https://github.com/aleixlopezpascual/arc-prize-2026-arc-agi-2](https://github.com/aleixlopezpascual/arc-prize-2026-arc-agi-2)
- **Top Submission Kernel:** `notebooks/boyko_dualpass_fork/`
- **Hybrid Solver Implementation:** `src/dsl/` and `src/models/`

### Citation
```bibtex
@misc{lopez2026arcprize,
  author = {Aleix L{\'o}pez},
  title = {ARC Prize 2026 - ARC-AGI-2: Hybrid Neural-Symbolic & Dual-Pass TTT Solver},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/aleixlopezpascual/arc-prize-2026-arc-agi-2}}
}
```
