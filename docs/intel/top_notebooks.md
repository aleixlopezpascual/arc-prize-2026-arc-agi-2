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

---

## Recent Public Contribution Snapshot (2026-10-24 13:43 UTC)

**Scope:** Selective review of the 100 most recently active public competition kernels returned by Kaggle, plus recent competition discussions. “Recently run” does not mean newly published or newly scored. I pulled and inspected six notebook sources/metadata and inspected four additional notebook pages; this is not an exhaustive survey. Claims below are labeled by the evidence actually visible on Kaggle and were not independently rerun here.

### Notable additions and updates

- **Dale Zhong — NVARC+ v6s/controller update.** The current page describes a three-pass pooled NVARC+ system, a footprint-matched runtime controller, and a P4 structurally distinct second-attempt rule. Kaggle still displays **31.94% as the best score from V3**; the recent V6s update is not evidence of a new score gain, and its several changes are not an isolated ablation.[9]
- **New multi-pass scheduling variants.** Yuvraj Lamba's “Apex Triple-Pass” title advertises “36.5%+,” but the page does not display a public score; its visible successful run is about 28 minutes on 4×L4. Treat the title as an author claim, not a measured result.[1] Woguo's recent notebook describes cheap-first work ordering, online actual/predicted runtime (“kappa”) calibration, and a fresh-seed leftover pass, but no public score is displayed on the page.[2] VLaDee's two-pass variant explicitly reruns the least-confident 50% with a new seed and a 3.5-hour reserve; the page's **28.19% is V1's best score**, while the latest V2 run is a 10-second run with no outputs.[7]
- **Candidate-score negative result — Candice Williams.** Her reproducibility notebook reports that the pre-registered demonstration-consistency rerank lost to the baseline selector on actual candidates (S1-base: 3 wins/12 losses; S1-TTT: 2/5). A separate planted-near-miss test reports that the score preferred truth over 95%-pixel-accurate random near-misses in 98% of outputs (R=0.85, 95% CI 0.78–0.93). This is a carefully specified, author-reported public-evaluation analysis—not an independent rerun—and the candidate-reranking result is not the same intervention as our task-priority policy.[4]
- **Neuro-symbolic stacks.** Efthimios's NVARC v3 page shows **30.14% (V5)** and describes a symbolic LOO prepass, SOAR-7B, NVARC, and a targeted Pass-B rescue. Source inspection shows the SOAR stage is gated off by default (`ARC_ENABLE_SOAR=0`), and the kernel metadata lists Qwen3-4B but no SOAR model source; do not treat the title as proof that SOAR ran in the scored path.[5] CPRS v7 shows **30.56% (V1)**; the latest run excerpt says zero verified symbolic programs were injected, so that run does not demonstrate a symbolic gain.[8]
- **Small evaluation prototype — Robina Mirbahar.** The notebook describes a topological DSL plus NVARC portfolio and reports 2.5/4 on a four-task evaluation smoke; its Kaggle page shows a 0.00 public score. Treat the smoke result as a prototype check, not full-set validation.[10]
- **CPU DSL baselines.** Masum's v9 adds identity/color operations and a greedy depth-4 beam; Fahd's baseline brute-forces DSL programs. Both pages currently display **0.00 public score**.[3][6] Our `src/dsl/search.py` already includes identity and one-/two-step candidate generation, so the distinct hypothesis is deeper guided search, not simply adding identity.
- **Discussion / infrastructure signal.** The newest “Three Consecutive Days of Kaggle Error” thread has three replies; its latest response says the similar issue being discussed was on ARC3, so it is not an ARC2-specific diagnosis.[11] The older ARC2 cache-instability thread has 32 replies; a commenter said Kaggle believed cache instability in some regions had stabilized, while later replies still describe problems. Treat this as community/operational evidence, not a guarantee about any specific run.[12]

### Ordered use plan for this project (hypotheses only)

These ideas must remain separate from the already planned broad-vs-uncertainty Pass-B pair. Do not change that pair's code, seed, frozen Pass-A set, or budgets to incorporate any of them.

1. **After the paired Pass-B comparison, replay online-kappa scheduling offline first.** Use existing per-task training/decode timing logs to estimate actual/predicted cost ratios and simulate remaining-budget reallocation against the current cheap-first order. Only if the replay predicts improved coverage without starving tasks should it become a one-factor validation-kernel experiment. This is the most direct scheduling idea from the recent NVARC variants.[1][2]
2. **Test structural diversity for attempt 2 as a separate selection ablation.** On the same frozen validation candidate pools, hold attempt 1 fixed and compare the current selector with a rule that prefers a sufficiently different second grid when the normal runner-up is near-duplicate. Score with `src/metrics.py` on identical task-ID folds; this adds no inference cost if the candidate pools already exist. The Dale/CPRS material is a hypothesis source, not proof of gain.[8][9]
3. **Screen depth-4 DSL search on CPU before any GPU work.** Compare it against the current one-/two-step search on training tasks, with a strict time cap and exact-match metric. Promote only if the added depth solves genuinely new tasks at acceptable cost; the public DSL pages' 0.00 scores provide no positive validation evidence.[3][6]
4. **Deprioritize demonstration-consistency reranking.** The public analysis reports fewer wins than losses on the real candidate set. Do not replace `score_ensemble` or fold this score into task priority without a new, isolated result on our exact candidate stream.[4]
5. **Defer SOAR-7B integration.** First establish that the intended checkpoint is available and attached, the environment gate is enabled in the scored path, and the stage completes under the time budget. The current public notebook's metadata/code do not establish that.[5]

All items above are hypotheses/backlog entries, not measured improvements in our repository. Any implementation must be tested separately, compared against the exact v25 path on matching task-ID folds, and scored through `src/metrics.py`; no public score alone promotes a change.

## Sources

[1] https://www.kaggle.com/code/yuvrajlamba01/arc-agi2-apex-triple-pass-sota — Apex Triple-Pass NVARC+ SOTA notebook
[2] https://www.kaggle.com/code/woguoat/arc2-nvarc-full-cost-sota-triple-pass — NVARC Full-Cost SOTA Triple Pass notebook
[3] https://www.kaggle.com/code/masumhuq/arc2-v9-dsl-solver — ARC2 v9 DSL solver
[4] https://www.kaggle.com/code/candiceevawilliams/arc2-a-correct-answer-is-not-a-better-demo — A Correct Answer Is Not a Better Demo reproducibility notebook
[5] https://www.kaggle.com/code/efthimiosfousekis/arc2-neurosymbolic-nvarc-v3 — Neuro-Symbolic NVARC v3
[6] https://www.kaggle.com/code/fahdmtidam/arc-agi-2-dsl-brute-force-baseline — DSL brute-force baseline
[7] https://www.kaggle.com/code/vladee/nvarc-2pass — NVARC 2-pass notebook
[8] https://www.kaggle.com/code/rajrajak99/cprs-arc-agi-2-apex-sota-engine-v7 — CPRS Apex SOTA Engine v7
[9] https://www.kaggle.com/code/dalezhong/arc-agi2-nvarc-plus-ckpt — NVARC+ checkpoint notebook
[10] https://www.kaggle.com/code/robinamirbahar/eda-gemini-4-argon-arc-agi-2-solver-suite — Gemini 4 Argon ARC-AGI-2 solver suite
[11] https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/752284 — Three Consecutive Days of Kaggle Error
[12] https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/751344 — Submission scored 0.00 after <2h: infra issue or true zero?
