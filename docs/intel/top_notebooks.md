# Top Public Notebooks & Community Implementations: arc-prize-2026-arc-agi-2

This document tracks and analyzes the most influential public notebooks and code releases for the ARC Prize 2026 (ARC-AGI-2) competition on Kaggle.

---

## 🏆 Summary Matrix of Top Public Notebooks

| Title | Author | Date | Votes | Public LB | Kernel Ref | Architecture / Focus |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **From Solver Control to Capability Acquisition** | Dale Zhong | 2026-10-15 | 1 | **`31.94%`** | `dalezhong/from-solver-control-to-capability-acquisition` | Evidence-guided co-evolution; analyzed augmentation-group failures. |
| **ARC-AGI 31.11** | Muhammad Junaid | 2026-10-13 | 147 | **`31.11%`** | `junaid512/arc-agi-31-11` | Interleaved `score_ensemble` (ProbMul + Voting) on Qwen 4B. |
| **ARC AGI2 LB33.89 Minimal Perfpatch** | mikelou1 | 2026-07-14 | 162 | **`33.89%`** | `mikelou1/arc-agi2-lb33-89-minimal-perfpatch` | Minimal performance patch: GPU logit slicing + turbo DFS. |
| **Fine-tuning NVARC with Six Added Layers** | Dmitrii Gluzdov | 2026-10-13 | 9 | **`29.17%`** | `dmitriigluzdov/fine-tuning-nvarc-with-six-added-layers` | Expanded Qwen3 to 42 layers pretrained on 342K synthetic tasks. |
| **ARC-AGI 2 Agentic Multi-turn Reasoning** | Khoi Nguyen | 2026-10-10 | 8 | - | `khoinguyennguyen/arc-agi-2-agentic-multi-turn-reasoning` | vLLM + persistent Jupyter sandbox code-execution feedback loop. |
| **ARC2 NVARC+ v1** | Luxshan Thavarasa | 2026-09-08 | 101 | **`29.31%`** | `luxluxshan/arc2-nvarc-v1` | Qwen 4B TTT with test-time data augmentation. |
| **[Baseline] NVARC Winning Solution (T4x2)** | parthenos | 2026-04-03 | 223 | **`28.89%`** | `nihilisticneuralnet/baseline-nvarc-arc-25-winning-solution-for-t4x2` | 2025 NVARC baseline ported to Kaggle T4 dual accelerators. |

---

## 🔬 Detailed Technical Breakdowns & Strategic Takeaways

### 1. Dale Zhong: *"From Solver Control to Capability Acquisition"* (Oct 15, 2026)
- **Kernel:** `dalezhong/from-solver-control-to-capability-acquisition` (Linked to Submission ID `57113549`, **`31.94%`**)
- **Methodology:** Explores the joint co-evolution of model space $\mathcal{M}$, evidence space $\mathcal{E}$, and update mechanisms $\mathcal{U}$.
- **Key Breakthrough Finding:**
  - *The Augmentation Symmetry Trap:* Augmentation-Inverse-Voting (AIRV) assumes tasks are closed under dihedral $D_4$ symmetries.
  - **Empirical Reality:** Many ARC-2 puzzles have **broken directional symmetries** (e.g., directional gravity where fluid only falls down, or fixed row-0 indicators). Flipping or rotating such puzzles corrupts the LoRA fine-tuning process by feeding contradictory physics.
- **Actionable Takeaway for Our Pipeline:** Before applying $D_4$ test-time augmentations, test whether the task exhibits directional invariance. If vertical/horizontal reflection alters the task semantics, disable those augmentation paths.

---

### 2. Muhammad Junaid: *"ARC-AGI 31.11"* (Oct 13, 2026)
- **Kernel:** `junaid512/arc-agi-31-11` (**147 votes**, Public LB: **`31.11%`**)
- **Methodology:** Replicates the interleaved `score_ensemble` dual-ranking algorithm:
  - Generates candidate grids via `turbo_dfs` GPU beam search.
  - Computes normalized likelihoods (`score_full_probmul_3`) and consensus counts (`score_kgmon`).
  - Interleaves the top selections to balance likelihood with consensus frequency.
- **Actionable Takeaway for Our Pipeline:** Confirms that our Tier-2 neural ranking implementation (`score_ensemble`) matches the gold standard of public competition ensembles.

---

### 3. Dmitrii Gluzdov: *"Fine-tuning NVARC with Six Added Layers"* (Oct 13, 2026)
- **Kernel:** `dmitriigluzdov/fine-tuning-nvarc-with-six-added-layers` (Public LB: **`29.17%`**)
- **Methodology:** 
  - Added 6 transformer layers to the Qwen 4B backbone (increasing depth from 36 to 42 hidden layers).
  - Pretrained on 342,000 synthetic ARC task families before test-time adaptation.
- **Result:** Scored `29.17%`, **failing to outperform the standard 36-layer baseline (`30.97%`)**.
- **Key Lesson:** Pretraining scale and raw model depth have hit diminishing returns on ARC-2. The bottleneck is not parameter capacity—it is topological reasoning, symbolic priors, and search guidance.

---

### 4. Khoi Nguyen: *"Agentic Multi-Turn Reasoning"* (Oct 10, 2026)
- **Kernel:** `khoinguyennguyen/arc-agi-2-agentic-multi-turn-reasoning` (8 votes)
- **Methodology:** Deploys a persistent Jupyter kernel sandbox inside Kaggle with vLLM. The language model writes transformation scripts, executes them, observes `grid_diff` errors, and refines the code iteratively.
- **Limitation:** In a 12-hour offline environment with 240 hidden test puzzles, multi-turn LLM agent loops take 2–3 minutes per puzzle, frequently hitting global timeouts.
- **Our Advantage:** Our Tier-1 Fast Symbolic Induction Engine evaluates 39 Core Knowledge primitives deterministically in Python in **$<0.25$ seconds per task**, providing the precision of code execution without timeout risk.

---

### 5. Mike Lou: *"ARC AGI2 LB33.89 Minimal Perfpatch"* (July 2026)
- **Kernel:** `mikelou1/arc-agi2-lb33-89-minimal-perfpatch` (162 votes, Public LB: **`33.89%`**)
- **Methodology:** Slices only the 12 ARC-specific vocabulary logits from the 151,936 Qwen vocabulary directly on GPU, cutting CPU transfer bottlenecks and accelerating DFS by ~3x.
- **Community Discovery:** Uses unseeded `hash(bk) % 1024**2`, creating run-to-run score volatility of $\pm 6\%$ across multiprocessing workers. Fixed in our repo via `zlib.crc32` deterministic hashing.
