# Deep-Dive Analysis: 1st-Place NVARC 2025 Winning Architecture

## 1. Executive Summary
The **NVARC (NVIDIA ARC)** solution took 1st place in ARC Prize 2025 by marrying **domain-specific test-time training (TTT)** with an ultra-efficient **constrained tree-search decoder (`turbo_dfs`)** over open-source LLMs (specifically Qwen-based architectures like `Qwen2.5-Coder` and fine-tuned `qwen3_4b_grids15_sft139`).

Rather than treating ARC as general code generation or pure few-shot prompting, NVARC approaches ARC as a **per-task visual adapter learning problem**:
1. Load a pre-trained ARC foundation model in 4-bit precision with LoRA.
2. For each task, augment the demonstration pairs (D4 dihedral group $\times$ color permutations) and perform rapid Test-Time Training (TTT) on the fly.
3. Fast in-memory LoRA weight reset between tasks (takes ~1 millisecond).
4. Run constrained Depth-First Search (`turbo_dfs`) over the valid token space ($0 \dots 9$, `\n`, `<|im_end|>`), aggressively pruning invalid prefixes.
5. Cross-score generated candidates across multiple geometric augmentations and select the top 2 attempts based on composite likelihood and consistency.

---

## 2. Grid Serialization & Tokenization Contract

### Token Format
Unlike naive formats that introduce spaces or commas (`[[1, 2], [3, 4]]`), NVARC serializes grids as dense strings where each row is a line of bare digits:
```text
012
345
678
```
This minimizes token count, maximizes autoregressive context capacity, and creates uniform token alignments.

### Prompt Template (ChatML)
Tasks are formatted using Qwen's ChatML template:
```text
<|im_start|>user
001
010
<|im_end|>
<|im_start|>assistant
100
010
<|im_end|>
<|im_start|>user
020
200
<|im_end|>
<|im_start|>assistant
```
A custom data collator (`QwenDataCollatorForCompletionOnlyLM`) masks out the user prompt tokens (`labels = -100`) so that backpropagation is applied exclusively to the output grid tokens.

---

## 3. Test-Time Training (TTT) Protocol

### High-Rank LoRA Setup
```python
peft_params = dict(
    r=256,
    lora_alpha=32,
    use_rslora=True,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head"
    ],
    lora_dropout=0.0,
    bias="none",
)
```
- **Rank $r = 256$:** High expressive capacity to absorb task-specific rules in few gradient steps.
- **rsLoRA (`use_rslora=True`):** Stabilizes scaling factor $\frac{\alpha}{\sqrt{r}}$ instead of standard $\frac{\alpha}{r}$.
- **All Linear Modules + Embeddings + LM Head:** Full attention and MLP projection adaptation.

### Instant In-Memory Adapter Reset
Reloading models from disk or reinitializing adapters takes seconds. NVARC avoids this completely:
```python
# Before starting inference loop
default_weights = get_peft_model_state_dict(model, adapter_name="default")
default_weights = {k: v.clone().detach() for k, v in default_weights.items()}

# In per-task loop (runs in ~1ms!)
set_peft_model_state_dict(model, default_weights.copy(), adapter_name="default")
```

### Demonstration Augmentation during TTT
To prevent overfitting on the 2–5 demonstration pairs, NVARC generates 40–80 synthetic training pairs per task:
- **Dihedral Group $D_4$:** 4 rotations (`0°`, `90°`, `180°`, `270°`) $\times$ 2 reflections (horizontal, vertical).
- **Color Permutations:** Random bijective permutations of the 10 color tokens ($0 \dots 9$).

---

## 4. Constrained Turbo-DFS Decoding

Standard autoregressive sampling often hallucinates malformed grids or invalid syntax. NVARC replaces beam search with **`turbo_dfs` (Turbo Depth-First Search)**:

```python
ARC_VOCAB = {"0": 0, "1": 1, ..., "9": 9, "Ċ": 10, "<|im_end|>": 15}
ARC_TOKENS = list(ARC_VOCAB.values())
max_score = -np.log(0.2)  # Cumulative NLL cutoff
```
- At each token position, only tokens from `ARC_TOKENS` are evaluated.
- Cumulative Negative Log-Likelihood ($\text{NLL}$) is tracked along the prefix tree.
- Branches exceeding `max_score` are pruned immediately.
- Completed hypotheses are validated for strict 2D rectangularity with dimensions $\le 30 \times 30$.

---

## 5. Candidate Ensembling & Ranking (`ArcDecoder`)

From multiple samples and augmentations, candidate grids are accumulated and ranked:
- **`score_full_probmul_3`**:
  $$\text{Score}(g) = \sum_{\text{beams}} (3 - \text{NLL}) + \frac{1}{|A|} \sum_{a \in A} (3 - \text{NLL}_a)$$
  Rewards candidates with both low inference beam loss and high cross-augmentation consistency.
- Top candidate $\to$ `attempt_1`.
- Highest-ranking distinct candidate $\to$ `attempt_2`.

---

## 6. Runtime, Concurrency & Kaggle Resource Budget

In Kaggle's 12-hour rerun environment on dual GPUs (e.g. 2x T4 or 2x L4):
- **Multiprocessing Worker Pool:**
  `torch.multiprocessing` spawns 1 worker per GPU. A thread-safe queue distributes `task_id` items dynamically.
- **Memory Optimization:**
  `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`, `FastLanguageModel` in 4-bit (`load_in_4bit=True`), peak memory resets between tasks.
- **Global Deadline Sentinel:**
  ```python
  global_end_time = time.time() + 12 * 3600 - 1200  # 20 minute safety buffer
  ```
  If `time.time() > global_end_time`, workers terminate cleanly and output the best cached predictions to `submission.json`.

---

## 7. Actionable Architectural Synthesis for Our Workspace

We will adopt a **Hybrid Dual-Engine Architecture**:
1. **Tier 1 (Instant Symbolic Filter):**
   Run `src/dsl/solver.py` on each task first. If a candidate symbolic program achieves 100% training score, emit its prediction in $<0.01$s.
2. **Tier 2 (Neural TTT Engine):**
   For tasks unsolved by symbolic primitives, route to the neural TTT solver with 4-bit Qwen2.5/Qwen3, high-rank LoRA, and `turbo_dfs`.
3. **Composite Submission:**
   Guarantee 100% valid schema compliance and maximize test coverage within the 12-hour compute budget.
