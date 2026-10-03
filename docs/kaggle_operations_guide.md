# Kaggle Operations & Deployment Guide: ARC-AGI-2

This guide documents the exact operational protocols for developing, building, packaging, deploying, and submitting kernels to the Kaggle platform for the ARC Prize 2026 competition.

---

## 1. Environment & Container Configuration

### The Python 3.11 ABI Compatibility Trap
In mid-2026, Kaggle upgraded its default container environment from Python 3.11 to Python 3.12 (`/usr/local/lib/python3.12/dist-packages`).
However, high-performance CUDA extensions—specifically `flash-attn` and `unsloth`—are bundled as precompiled binary wheels for **Python 3.11 (`cp311`)**.

If a kernel is pushed without explicit container pinning, Kaggle runs the notebook in the Python 3.12 container. When the script imports `torch` or `numpy`, Python attempts to load the Python 3.11 C-extensions injected by utility kernels, resulting in immediate fatal errors:
```text
ImportError: Failed to load PyTorch C extensions
ModuleNotFoundError: No module named 'numpy._core._multiarray_umath'
```

### The Fix: Pinned BYOD Docker Image
Every submission `kernel-metadata.json` must explicitly specify the proven Python 3.11 BYOD image:
```json
{
  "id": "aleixlopez/arc-agi2-hybrid-sota-symbolic-ttt",
  "title": "ARC AGI2 Hybrid SOTA Symbolic TTT",
  "code_file": "submission_kernel.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": false,
  "keywords": ["gpu"],
  "docker_image_pinning_type": "original",
  "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868",
  "dataset_sources": [],
  "kernel_sources": [
    "sorokin/pip-install-unsloth-flash-patch"
  ],
  "competition_sources": [
    "arc-prize-2026-arc-agi-2"
  ],
  "model_sources": [
    "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
  ],
  "machine_shape": "NvidiaL4"
}
```

---

## 2. Standalone Notebook Builder with Compile Gate

To prevent syntax, indentation, or import collisions in cloud runs, all submission notebooks are compiled locally using `src/tools/build_submission_notebook.py`:

```bash
python3 src/tools/build_submission_notebook.py
```

### Safety Features of the Builder
1. **Dynamic Module Inlining:** Inlines `src/dsl/primitives.py`, `src/dsl/cellular_automata.py`, `src/dsl/overlay.py`, `src/dsl/search.py`, and `src/dsl/solver.py` into a unified `arc_dsl.py` cell.
2. **Local Import Stripping:** Removes all `from src.` package imports while strictly preserving indentation, control flows, and exception handlers (`except Exception: pass`).
3. **Pre-Compilation Verification Gate:**
   ```python
   compile(code_to_verify, "arc_dsl.py", "exec")
   ```
   Compiles the complete inlined payload and merge cell in an isolated namespace before writing the `.ipynb` file. Any `IndentationError` or `SyntaxError` fails immediately locally before touching Kaggle cloud GPUs.

---

## 3. Managing Kaggle Resource Quotas

Kaggle enforces strict limits in code competitions:
1. **Concurrent GPU Session Limit:** A maximum of **2 active batch GPU sessions** can be running simultaneously across your account. Attempting to push a 3rd kernel returns:
   `Kernel push error: Maximum batch GPU session count of 2 reached.`
2. **Daily Competition Submission Quota:** Code competitions limit accounts to **1 submission per day** (resets at 00:00 UTC). Check remaining daily quota via:
   ```bash
   kaggle competitions submission-limits -c arc-prize-2026-arc-agi-2
   ```

---

## 4. Autonomous Background Submitter (`src/tools/auto_submit.py`)

To eliminate manual polling and timing of 30-minute GPU runs, launch the unbuffered auto-submit watcher in the background:

```bash
python3 -u -m src.tools.auto_submit --version 5
```

### Watcher Lifecycle
1. Polls `kaggle kernels status <kernel_slug>` every 45 seconds.
2. When status reaches `COMPLETE`:
   - Downloads the generated output to `/tmp/auto_sub_output/`.
   - Validates `submission.json` against `src/validator.py`.
   - Queries `submission-limits` to confirm quota $> 0$.
   - Executes `kaggle competitions submit -c arc-prize-2026-arc-agi-2 -k <kernel_slug> -v 5 -f submission.json -m "..."`.
   - Logs the submission reference ID and exits.

---

## 5. Live Leaderboard Verification

To download and query your exact position on the global competition leaderboard:
```bash
kaggle competitions leaderboard -c arc-prize-2026-arc-agi-2 --download -p /tmp/leaderboard
unzip -o /tmp/leaderboard/arc-prize-2026-arc-agi-2.zip -d /tmp/leaderboard
python3 -c "
import glob, pandas as pd
df = pd.read_csv(glob.glob('/tmp/leaderboard/*.csv')[0])
row = df[df['TeamMemberUserNames'].str.contains('aleixlopez', na=False)]
print(row[['Rank', 'TeamName', 'Score', 'SubmissionCount']])
"
```
*(Currently: Rank 1,043 / 2,377 competitors, Score: 29.31%)*
