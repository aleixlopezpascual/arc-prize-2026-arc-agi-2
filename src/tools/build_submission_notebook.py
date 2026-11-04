from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[2]


def build(
    output_path: Path | None = None,
    validation_mode: Literal["submission", "train", "eval"] = "submission",
    pass_b_policy: Literal["broad", "uncertainty", "paired"] = "broad",
    symbolic_merge: bool = True,
    task_ids: list[str] | None = None,
) -> Path:
    if task_ids is not None:
        if validation_mode == "submission":
            raise ValueError("Task subsets are a validation artifact; they cannot be used for a submission.")
        if not task_ids or len(set(task_ids)) != len(task_ids) or not all(isinstance(t, str) and t for t in task_ids):
            raise ValueError("task_ids must be a non-empty list of unique task ID strings.")
    if pass_b_policy == "paired" and validation_mode != "train":
        raise ValueError("The paired Pass-B validation mode is only supported for train CV.")
    if not symbolic_merge and validation_mode == "submission":
        raise ValueError("Neural-only output is a validation artifact; it cannot be used for a submission.")
    if not symbolic_merge and pass_b_policy == "paired":
        raise ValueError("Neural-only output is not supported with the paired Pass-B policy.")

    base_nb_path = REPO_ROOT / "notebooks/public_baselines/lb33_89/arc-agi2-lb33-89-minimal-perfpatch.ipynb"
    base_nb = json.loads(base_nb_path.read_text())

    primitives_py = (REPO_ROOT / "src/dsl/primitives.py").read_text()
    overlay_py = (REPO_ROOT / "src/dsl/overlay.py").read_text()
    ca_py = (REPO_ROOT / "src/dsl/cellular_automata.py").read_text()
    search_py = (REPO_ROOT / "src/dsl/search.py").read_text()
    solver_py = (REPO_ROOT / "src/dsl/solver.py").read_text()

    # Create arc_dsl.py content
    arc_dsl_content = "%%writefile arc_dsl.py\n"
    arc_dsl_content += "# Combined ARC DSL primitives, search, and solver\n"
    arc_dsl_content += "import copy\nimport numpy as np\nfrom collections import Counter, deque\nfrom dataclasses import dataclass, field\nfrom typing import Any, Callable, Sequence, Union\n\n"
    arc_dsl_content += "Grid = list[list[int]]\n\n"

    # Add grid_equal helper
    arc_dsl_content += """
def grid_equal(g1, g2):
    if g1 is None or g2 is None:
        return False
    a1 = np.asarray(g1, dtype=int)
    a2 = np.asarray(g2, dtype=int)
    return a1.shape == a2.shape and bool(np.array_equal(a1, a2))
\n"""

    def clean_code(code_str: str) -> str:
        lines = []
        for line in code_str.splitlines():
            s = line.strip()
            if s.startswith("from __future__") or s == "Grid = list[list[int]]":
                continue
            if s.startswith("from src.") or s.startswith("import src."):
                continue
            lines.append(line.replace("P.", ""))
        return "\n".join(lines)

    cleaned_prims = clean_code(primitives_py)
    arc_dsl_content += cleaned_prims + "\n\n"

    cleaned_overlay = clean_code(overlay_py)
    arc_dsl_content += cleaned_overlay + "\n\n"

    cleaned_ca = clean_code(ca_py)
    arc_dsl_content += cleaned_ca + "\n\n"

    cleaned_search = clean_code(search_py)
    arc_dsl_content += cleaned_search + "\n\n"

    cleaned_solver = clean_code(solver_py)
    arc_dsl_content += cleaned_solver + "\n"

    # Pre-compile verification
    code_to_verify = arc_dsl_content.replace("%%writefile arc_dsl.py\n", "")
    try:
        compile(code_to_verify, "arc_dsl.py", "exec")
        print("[+] Verified: arc_dsl.py compiles cleanly with zero syntax/indentation errors!")
    except Exception as e:
        raise RuntimeError(f"Syntax verification failed for arc_dsl.py: {e}")

    cells = []

    # Cell 0: Markdown
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ARC-AGI-2 Hybrid SOTA (Tier 1 Symbolic Filter + Tier 2 Qwen 4B rsLoRA TTT)\n",
                "\n",
                "Architecture:\n",
                "1. **Tier 1 (Instant Symbolic Program Induction):** Evaluates 1- and 2-step DSL primitives (connected components, enclosed region fills, dot connections, periodic room repairs, and geometric symmetries). If 100% verified on train demonstrations, pre-empts attempt_1.\n",
                "2. **Tier 2 (Test-Time Training Transduction):** 4B Qwen pre-trained model with 128 augmentations, high-rank rsLoRA (r=256), and GPU-accelerated constrained turbo_dfs.\n",
                "3. **Monotonic Merging:** Ensures every verified task improves accuracy without regression on unsolved tasks.\n",
            ],
        }
    )

    # Cell 1: Buffer deadline and deterministic seeding
    env_cell = """# Keep the original global 10-minute submission/write buffer and enforce deterministic seeding.
import os
import time
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
global_end_time = time.time() + 12 * 3600 - 600
"""
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [env_cell],
        }
    )

    # Cell 2: pip uninstall tensorflow
    cells.append(base_nb["cells"][2])

    # Cell 3: arc_loader.py
    loader_cell = base_nb["cells"][3]
    if task_ids is not None:
        # Restrict every ArcDataset.from_file() call (workers, uncertainty map, merge) to the chosen tasks.
        loader_cell = json.loads(json.dumps(loader_cell))
        loader_source = "".join(loader_cell["source"])
        old_from_file = (
            "        return cls(\n"
            "            queries=json.loads(queries),\n"
            "            is_orig=True,\n"
            "            keys=keys,\n"
            "        )\n"
        )
        new_from_file = (
            "        import os\n"
            "        queries = json.loads(queries)\n"
            '        ids_file = os.getenv("ARC_TASK_IDS_FILE")\n'
            "        if ids_file and keys is None:\n"
            "            with open(ids_file) as f:\n"
            "                wanted = set(json.load(f))\n"
            "            keys = sorted(k for k in queries if k in wanted)\n"
            "        return cls(\n"
            "            queries=queries,\n"
            "            is_orig=True,\n"
            "            keys=keys,\n"
            "        )\n"
        )
        if loader_source.count(old_from_file) != 1:
            raise ValueError("Could not find the expected ArcDataset.from_file block.")
        loader_cell["source"] = loader_source.replace(old_from_file, new_from_file).splitlines(keepends=True)
    cells.append(loader_cell)
    if task_ids is not None:
        ids_source = (
            "import json\n"
            "import os\n"
            f"TASK_IDS = {sorted(task_ids)!r}\n"
            'with open("/kaggle/working/task_ids.json", "w") as f:\n'
            "    json.dump(TASK_IDS, f)\n"
            'os.environ["ARC_TASK_IDS_FILE"] = "/kaggle/working/task_ids.json"\n'
            'print(f"*** Restricting run to {len(TASK_IDS)} task IDs")\n'
        )
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [ids_source],
            }
        )

    # Cell 4: arc_dsl.py
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [arc_dsl_content],
        }
    )

    # Cell 5: arc_decoder.py with score_ensemble and physical invariant filter
    decoder_source = "".join(base_nb["cells"][4]["source"])
    if "def score_ensemble" not in decoder_source:
        filter_code = """
def is_physically_plausible(solution):
    if solution is None:
        return False
    arr = np.asarray(solution)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[0] > 30 or arr.shape[1] < 1 or arr.shape[1] > 30:
        return False
    if np.any((arr < 0) | (arr > 9)):
        return False
    return True

def filter_valid_guesses(guesses):
    valid = {k: g for k, g in guesses.items() if is_physically_plausible(g.get("solution"))}
    return valid if valid else guesses
"""
        ensemble_code = """
def candidate_signature(grid):
    arr = np.asarray(grid)
    return (arr.shape, frozenset(np.unique(arr).tolist()))

def score_ensemble(guesses):
    # Dale Zhong Rule C: Cross-Family Pass@2 Selection
    # Family 1: Maximum Likelihood (Token probability)
    # Family 2: Dihedral Consensus Voting (Symmetry invariance)
    ranked_prob = score_full_probmul_3(guesses)
    ranked_vote = score_kgmon(guesses)
    
    if not ranked_prob:
        return ranked_vote
    if not ranked_vote:
        return ranked_prob
        
    y1 = ranked_prob[0]
    h1 = hashable(y1)
    sig1 = candidate_signature(y1)
    
    result = [y1]
    seen = {h1}
    
    # Rule C: If top consensus vote differs from y1, pair it immediately as attempt 2
    hv0 = hashable(ranked_vote[0])
    if hv0 not in seen:
        seen.add(hv0)
        result.append(ranked_vote[0])
        
    # Interleave remaining candidates
    all_cands = []
    for a, b in zip(ranked_prob[1:], ranked_vote[1:]):
        for cand in (b, a):
            h = hashable(cand)
            if h not in seen:
                seen.add(h)
                all_cands.append(cand)
                
    # If attempt 2 is still needed, prioritize candidate with divergent signature (cross-family)
    if len(result) < 2:
        for cand in all_cands:
            if candidate_signature(cand) != sig1:
                result.append(cand)
                seen.add(hashable(cand))
                break
                
    # Fill remaining guesses
    for cand in all_cands:
        h = hashable(cand)
        if h not in seen:
            seen.add(h)
            result.append(cand)
            
    for cand in ranked_prob + ranked_vote:
        h = hashable(cand)
        if h not in seen:
            seen.add(h)
            result.append(cand)
            
    return result
"""
        decoder_source = decoder_source.replace(
            "def score_sum(guesses, getter):\n    guess_list = list(guesses.values())",
            f"{filter_code}\ndef score_sum(guesses, getter):\n    guesses = filter_valid_guesses(guesses)\n    guess_list = list(guesses.values())",
        )
        decoder_source = decoder_source.replace(
            "selection_algorithms = [\n    score_full_probmul_3,\n    score_kgmon,\n]",
            f"{ensemble_code}\nselection_algorithms = [\n    score_full_probmul_3,\n    score_kgmon,\n    score_ensemble,\n]",
        )
        decoder_source = decoder_source.replace(
            "def run_selection_algo(self, selection_algorithm=score_kgmon):",
            "def run_selection_algo(self, selection_algorithm=score_ensemble):",
        )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [decoder_source],
        }
    )

    # Cell 6: arc_solver.py (with deterministic CRC32 seeding and dynamic task timeouts)
    solver_source = "".join(base_nb["cells"][5]["source"])
    worker_dataset_selection = (
        '    if rerun_mode:\n'
        '        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"\n'
        '    else:\n'
        '        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"'
    )
    worker_validation_selection = (
        '    validation_mode = os.getenv("ARC_VALIDATION_MODE", "submission")\n'
        '    if validation_mode == "train":\n'
        '        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json"\n'
        '        if not os.path.exists(test_path):\n'
        '            test_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json"\n'
        '    elif validation_mode == "eval":\n'
        '        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"\n'
        '        if not os.path.exists(test_path):\n'
        '            test_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"\n'
        '    elif rerun_mode:\n'
        '        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"\n'
        '        if not os.path.exists(test_path):\n'
        '            test_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"\n'
        '    else:\n'
        '        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"\n'
        '        if not os.path.exists(test_path):\n'
        '            test_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"'
    )
    if worker_dataset_selection not in solver_source:
        raise ValueError("Could not find the expected worker dataset selection block.")
    solver_source = solver_source.replace(worker_dataset_selection, worker_validation_selection)
    if "import zlib" not in solver_source:
        solver_source = solver_source.replace(
            "import bz2\nimport pickle\n", "import bz2\nimport pickle\nimport zlib\n"
        )
    solver_source = solver_source.replace(
        "aug_dataset = aug_dataset.augment(seed=hash(bk) % 1024**2)",
        "aug_dataset = aug_dataset.augment(seed=zlib.crc32(bk.encode('utf-8')) % 1024**2)",
    )

    # Dynamic task timeout in turbo_dfs
    solver_source = solver_source.replace(
        'dir_outputs = "/kaggle/inference_outputs"',
        'dir_outputs = os.getenv("ARC_OUT_DIR", "/kaggle/inference_outputs")',
    )
    solver_source = solver_source.replace(
        "train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)",
        """n_train_aug = int(os.getenv("ARC_N_TRAIN_AUG", "8"))
        train_seed = int(os.getenv("ARC_SEED", "1"))
        train_ds = puzzle_ds.augment(n=n_train_aug, shfl_keys=True, seed=train_seed)""",
    )
    solver_source = solver_source.replace(
        "def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time) -> dict:",
        "def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time, task_timeout: float = 540.0) -> dict:",
    )
    solver_source = solver_source.replace(
        "            start_time=start_time,\n            end_time=end_time,\n        )",
        "            start_time=start_time,\n            end_time=end_time,\n            task_timeout=task_timeout,\n        )",
    )
    solver_source = solver_source.replace(
        "    while time.time() - start_time < 540 and time.time() < end_time:",
        "    while time.time() - start_time < task_timeout and time.time() < end_time:",
    )

    # Dynamic task timeout in inference_turbo_dfs
    solver_source = solver_source.replace(
        "def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):",
        "def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time, task_timeout: float = 540.0):",
    )
    solver_source = solver_source.replace(
        "        start_time=time.time(),\n        end_time=end_time,\n    )",
        "        start_time=time.time(),\n        end_time=end_time,\n        task_timeout=task_timeout,\n    )",
    )

    # Worker unpacking dynamic timeout
    solver_source = solver_source.replace(
        "        key = queue.get()\n        if key is None:\n            break",
        """        item = queue.get()
        if item is None:
            break
        if isinstance(item, (list, tuple)):
            key, task_timeout = item
        else:
            key, task_timeout = item, 540.0""",
    )
    solver_source = solver_source.replace(
        "dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time)",
        "dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time, task_timeout=task_timeout)",
    )
    solver_source = solver_source.replace(
        "if spend_time > 1200 or time.time() > end_time:",
        "if spend_time > max(1200.0, task_timeout * 2) or time.time() > end_time:",
    )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [solver_source],
        }
    )

    if pass_b_policy in ('uncertainty', 'paired'):
        pass_priority_py = (REPO_ROOT / "src/neural/pass_priority.py").read_text()
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["%%writefile pass_priority.py\n" + pass_priority_py],
            }
        )

    if pass_b_policy in ('uncertainty', 'paired'):
        uncertainty_routing = """    if os.getenv("ARC_PASS_B_POLICY", "broad") == "uncertainty" and os.path.exists("/kaggle/working/pass_b_uncertainty.json"):
        import pass_priority
        with open("/kaggle/working/pass_b_uncertainty.json", "r") as f:
            unc_map = json.load(f)

        # Rank hard tasks
        work_map = {k: estimated_work(data[k]) for k in hard_tasks}
        hard_tasks = pass_priority.rank_tasks_by_uncertainty({k: unc_map.get(k, 1.0) for k in hard_tasks}, work_map)

        # Rank easy tasks
        easy_work_map = {k: estimated_work(data[k]) for k in easy_tasks}
        easy_tasks = pass_priority.rank_tasks_by_uncertainty({k: unc_map.get(k, 1.0) for k in easy_tasks}, easy_work_map)
    else:
        hard_tasks = sorted(hard_tasks, key=lambda k: estimated_work(data[k]))
        """
    else:
        uncertainty_routing = """    hard_tasks = sorted(hard_tasks, key=lambda k: estimated_work(data[k]))
"""

    # Cell 7: starter.py (with Deficit-First Routing)
    starter_source = f"""%%writefile starter.py
import os
import time
import json
import torch
import argparse
import torch.multiprocessing as mp


def cleanup_barriers(barrier_dir=None, barrier_tag=None):
    if barrier_dir is None:
        barrier_dir = os.environ.get("ARC_BARRIER_DIR", "/kaggle")
    if barrier_tag is None:
        barrier_tag = os.environ.get("ARC_BARRIER_TAG", "")
    if not os.path.isdir(barrier_dir):
        return
    prefix = f"worker_{{barrier_tag}}_" if barrier_tag else "worker"
    for r in range(4):
        bf = os.path.join(barrier_dir, f"{{prefix}}{{r}}")
        try:
            if os.path.isfile(bf):
                os.remove(bf)
        except OSError:
            pass
    if barrier_tag:
        for r in range(4):
            bf = os.path.join(barrier_dir, f"worker{{r}}")
            try:
                if os.path.isfile(bf):
                    os.remove(bf)
            except OSError:
                pass


def local_worker(rank, queue, end_time):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    barrier_dir = os.environ.get("ARC_BARRIER_DIR", "/kaggle")
    barrier_tag = os.environ.get("ARC_BARRIER_TAG", "")
    prefix = f"worker_{{barrier_tag}}_" if barrier_tag else "worker"

    # Fix Unsloth patching issue: stagger worker imports so rank 0 patches cleanly first
    if rank > 0:
        prev_barrier = os.path.join(barrier_dir, f"{{prefix}}{{rank-1}}")
        wait_start = time.time()
        while not os.path.exists(prev_barrier):
            if time.time() - wait_start > 300:
                raise TimeoutError(f"[Rank {{rank}}] Timed out waiting for barrier {{prev_barrier}}")
            time.sleep(5)
    
    from arc_solver import worker

    barrier_file = os.path.join(barrier_dir, f"{{prefix}}{{rank}}")
    with open(barrier_file, "w") as f:
        f.write("Ok")
    
    print(f"[Rank {{rank}}] start!")
    
    worker(rank, queue, end_time)
    
    print(f"[Rank {{rank}}] done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if {validation_mode!r} == 'train':
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json"
        if not os.path.exists(test_path):
            test_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json"
    elif {validation_mode!r} == 'eval':
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
        if not os.path.exists(test_path):
            test_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
    else:
        if rerun_mode:
            test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
            if not os.path.exists(test_path):
                test_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
        else:
            test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
            if not os.path.exists(test_path):
                test_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    # Deficit-First Routing: detect tasks already solved 100% by Tier 1 Symbolic Induction
    import arc_dsl
    
    easy_tasks = []
    hard_tasks = []

    for key in sorted(data.keys()):
        if {validation_mode!r} == 'submission' and not rerun_mode:
            if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:
                continue
        task_train = data[key]["train"]
        try:
            progs = arc_dsl.search_dsl_programs(task_train, max_level=1, max_perfect_programs=2)
        except Exception:
            progs = []
        if progs:
            easy_tasks.append(key)
        else:
            hard_tasks.append(key)

    print(f"*** Deficit-First Routing: {{len(easy_tasks)}} Tier-1 Solved | {{len(hard_tasks)}} Hard (Expanded-TTT)")

    # Cheap-First Work Estimator: sort hard tasks so compact, highly solvable grids are processed first!
    def estimated_work(task):
        def ntok(g):
            return len(g) * (len(g[0]) + 1)
        train_tokens = sum(ntok(p["input"]) + ntok(p["output"]) for p in task["train"])
        ratios = [ntok(p["output"]) / max(1, ntok(p["input"])) for p in task["train"]]
        ratios.sort()
        ratio = ratios[len(ratios) // 2]
        test_tokens = sum(ntok(t["input"]) * (1 + ratio) for t in task["test"])
        return train_tokens * 16 + test_tokens * 8 * len(task["test"])

{uncertainty_routing}

    # Hard tasks get expanded 720s timeout (+33%), easy tasks get fast 90s timeout
    hard_timeout = 720.0
    easy_timeout = 90.0

    queue = mp.Manager().Queue()

    # Queue hard tasks FIRST in Cheap-First order to maximize solved puzzles before deadline
    for key in hard_tasks:
        queue.put((key, hard_timeout))
        
    for key in easy_tasks:
        queue.put((key, easy_timeout))

    for _ in range(4):
        queue.put(None)
    
    barrier_dir = os.environ.get("ARC_BARRIER_DIR", "/kaggle")
    barrier_tag = os.environ.get("ARC_BARRIER_TAG", "")
    cleanup_barriers(barrier_dir, barrier_tag)

    try:
        mp.spawn(local_worker, args=(queue, args.end_time), nprocs=4)
    finally:
        cleanup_barriers(barrier_dir, barrier_tag)
"""
    if task_ids is not None:
        # The deficit-first router reads the raw JSON, so filter it here too.
        old_route = '    with open(test_path, "r") as f:\n        data = json.load(f)\n'
        new_route = (
            old_route
            + '    _ids_file = os.getenv("ARC_TASK_IDS_FILE")\n'
            + "    if _ids_file:\n"
            + "        with open(_ids_file) as _f:\n"
            + "            _wanted = set(json.load(_f))\n"
            + "        data = {k: v for k, v in data.items() if k in _wanted}\n"
        )
        if starter_source.count(old_route) != 1:
            raise ValueError("Could not find the expected starter data-load block.")
        starter_source = starter_source.replace(old_route, new_route)
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [starter_source],
        }
    )

    # Cell 8: Multi-Pass TTT Runner (Pass A + Pass B Candidate Generation)
    multi_pass_runner = """import os
import sys
import time
import glob
import subprocess


def cleanup_kaggle_barriers():
    barrier_dir = os.environ.get("ARC_BARRIER_DIR", "/kaggle")
    if os.path.isdir(barrier_dir):
        for pattern in ("worker*", "worker_*"):
            for bf in glob.glob(os.path.join(barrier_dir, pattern)):
                if os.path.isfile(bf):
                    try:
                        os.remove(bf)
                    except OSError:
                        pass


cleanup_kaggle_barriers()
print("=== [Pass A] Starting Primary Half-Cost TTT Run (Seed 1, ARC_N_TRAIN_AUG=8) ===")
env_a = dict(
    os.environ,
    UNSLOTH_DISABLE_STATISTICS="1",
    TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas",
    OMP_NUM_THREADS="12",
    ARC_N_TRAIN_AUG="8",
    ARC_SEED="1",
    ARC_OUT_DIR="/kaggle/inference_outputs",
)
subprocess.call([sys.executable, "starter.py", "--end-time", f"{global_end_time}"], env=env_a)

time_remaining = global_end_time - time.time()
print(f"[*] Pass A finished. Remaining wall-clock time: {time_remaining/60:.1f} minutes")
"""

    if pass_b_policy == "paired":
        pass_a_log_marker = 'print("=== [Pass A] Starting Primary Half-Cost TTT Run (Seed 1, ARC_N_TRAIN_AUG=8) ===")\n'
        pass_a_call_marker = 'subprocess.call([sys.executable, "starter.py", "--end-time", f"{global_end_time}"], env=env_a)'
        if multi_pass_runner.count(pass_a_log_marker) != 1 or multi_pass_runner.count(pass_a_call_marker) != 1:
            raise ValueError("Could not locate the standard Pass-A runner template.")
        multi_pass_runner = multi_pass_runner.replace(
            pass_a_log_marker,
            "paired_run_start = time.time()\n"
            "paired_inference_deadline = global_end_time - 600\n"
            "pass_a_end_time = paired_run_start + max(0.0, (paired_inference_deadline - paired_run_start) * 0.5)\n"
            + pass_a_log_marker,
        )
        multi_pass_runner = multi_pass_runner.replace(
            pass_a_call_marker,
            'subprocess.run([sys.executable, "starter.py", "--end-time", str(pass_a_end_time)], env=env_a, check=True)',
        )

    if pass_b_policy in ('uncertainty', 'paired'):
        multi_pass_runner += f"""
print("=== Computing Pass-A Uncertainty Map ===")
import json
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder, score_full_probmul_3, score_kgmon, hashable, filter_valid_guesses
from pass_priority import uncertainty_from_candidate_evidence

if {validation_mode!r} == 'train':
    test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json"
    if not os.path.exists(test_path):
        test_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json"
elif {validation_mode!r} == 'eval':
    test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
    if not os.path.exists(test_path):
        test_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
else:
    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
        if not os.path.exists(test_path):
            test_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
        if not os.path.exists(test_path):
            test_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

try:
    data = ArcDataset.from_file(test_path)
    decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
    decoder.load_decoded_results("/kaggle/inference_outputs")

    unc_map = {{}}
    task_unc_list = {{}}
    for output_key, guesses_dict in decoder.decoded_results.items():
        task_id = output_key.rsplit('_', 1)[0]
        valid_guesses = filter_valid_guesses(guesses_dict)

        ranked_prob = score_full_probmul_3(guesses_dict)
        prob_strs = [str(hashable(g)) for g in ranked_prob]

        ranked_vote = score_kgmon(guesses_dict)
        vote_strs = [str(hashable(g)) for g in ranked_vote]

        guess_list = list(valid_guesses.values())
        vote_counts = {{}}
        for g in guess_list:
            h_str = str(hashable(g["solution"]))
            vote_counts[h_str] = vote_counts.get(h_str, 0) + 1

        try:
            unc = uncertainty_from_candidate_evidence(prob_strs, vote_strs, vote_counts)
        except Exception as e:
            print(f"Error computing uncertainty for output {{output_key}}: {{e}}")
            unc = 1.0

        if task_id not in task_unc_list:
            task_unc_list[task_id] = []
        task_unc_list[task_id].append(unc)

    for task_id, uncs in task_unc_list.items():
        unc_map[task_id] = sum(uncs) / len(uncs)

    if not unc_map:
        raise ValueError("Pass A produced no candidate uncertainty evidence.")

    with open("/kaggle/working/pass_b_uncertainty.json", "w") as f:
        json.dump(unc_map, f)
    print(f"Successfully wrote Pass-A Uncertainty Map for {{len(unc_map)}} task IDs.")
except Exception as main_e:
    print(f"Failed to generate uncertainty map: {{main_e}}")
    raise RuntimeError("Pass-A uncertainty map generation failed") from main_e

"""

    if pass_b_policy == "paired":
        multi_pass_runner += """
# Both Pass-B arms reuse the exact same frozen Pass-A output directory.
pass_b_arm_budget = max(0.0, (paired_inference_deadline - time.time()) / 2.0)
if pass_b_arm_budget < 900:
    raise RuntimeError("Insufficient time to give both paired Pass-B arms a minimum budget.")

print("=== [Paired Pass B 1/2] Broad ordering, Seed 137 ===")
cleanup_kaggle_barriers()
broad_end_time = min(paired_inference_deadline, time.time() + pass_b_arm_budget)
env_broad = dict(
    os.environ,
    UNSLOTH_DISABLE_STATISTICS="1",
    TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas",
    OMP_NUM_THREADS="12",
    ARC_N_TRAIN_AUG="8",
    ARC_SEED="137",
    ARC_OUT_DIR="/kaggle/inference_outputs_broad",
)
subprocess.run([sys.executable, "starter.py", "--end-time", str(broad_end_time)], env=env_broad, check=True)
if not any(os.scandir("/kaggle/inference_outputs_broad")):
    raise RuntimeError("Broad Pass B produced no candidate files.")

print("=== [Paired Pass B 2/2] Uncertainty ordering, Seed 137 ===")
cleanup_kaggle_barriers()
uncertainty_time_remaining = paired_inference_deadline - time.time()
if uncertainty_time_remaining < 900:
    raise RuntimeError("The broad arm overran the paired budget; refusing an unmatched uncertainty arm.")
uncertainty_end_time = min(paired_inference_deadline, time.time() + pass_b_arm_budget)
env_uncertainty = dict(
    os.environ,
    UNSLOTH_DISABLE_STATISTICS="1",
    TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas",
    OMP_NUM_THREADS="12",
    ARC_N_TRAIN_AUG="8",
    ARC_SEED="137",
    ARC_OUT_DIR="/kaggle/inference_outputs_uncertainty",
)
subprocess.run([sys.executable, "starter.py", "--end-time", str(uncertainty_end_time)], env=env_uncertainty, check=True)
if not any(os.scandir("/kaggle/inference_outputs_uncertainty")):
    raise RuntimeError("Uncertainty Pass B produced no candidate files.")
"""
    else:
        multi_pass_runner += """
# Pass B: Secondary independent pass with Seed 137 if remaining time > 15 minutes
if time_remaining > 900:
    cleanup_kaggle_barriers()
    print("=== [Pass B] Starting Secondary Independent TTT Run (Seed 137, ARC_N_TRAIN_AUG=8) ===")
    env_b = dict(
        os.environ,
        UNSLOTH_DISABLE_STATISTICS="1",
        TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas",
        OMP_NUM_THREADS="12",
        ARC_N_TRAIN_AUG="8",
        ARC_SEED="137",
        ARC_OUT_DIR="/kaggle/inference_outputs_b",
    )
    subprocess.call([sys.executable, "starter.py", "--end-time", f"{global_end_time}"], env=env_b)
"""

    validation_mode_env = f'    ARC_VALIDATION_MODE="{validation_mode}",\n'
    output_policies = {
        "/kaggle/inference_outputs": ("broad", "pass_a"),
        "/kaggle/inference_outputs_b": ("uncertainty" if pass_b_policy == "uncertainty" else "broad", "pass_b"),
        "/kaggle/inference_outputs_broad": ("broad", "pass_b_broad"),
        "/kaggle/inference_outputs_uncertainty": ("uncertainty", "pass_b_uncertainty"),
    }
    for output_dir, (route_policy, barrier_tag) in output_policies.items():
        marker = f'    ARC_OUT_DIR="{output_dir}",\n'
        occurrences = multi_pass_runner.count(marker)
        if occurrences > 1:
            raise ValueError(f"Expected at most one environment block for {output_dir}.")
        if occurrences == 1:
            route_env = (
                validation_mode_env
                + f'    ARC_PASS_B_POLICY="{route_policy}",\n'
                + f'    ARC_BARRIER_TAG="{barrier_tag}",\n'
            )
            multi_pass_runner = multi_pass_runner.replace(marker, marker + route_env)

    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [multi_pass_runner],
        }
    )

    # Cell 9: Hybrid merge and submission
    if validation_mode == 'train':
        cell9_paths = """    test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json"
    if not os.path.exists(test_path):
        test_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json"
    data = ArcDataset.from_file(test_path)
    sol_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_solutions.json"
    if not os.path.exists(sol_path):
        sol_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_training_solutions.json"
    data = data.load_replies(sol_path)"""
    elif validation_mode == 'eval':
        cell9_paths = """    test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
    if not os.path.exists(test_path):
        test_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
    data = ArcDataset.from_file(test_path)
    sol_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json"
    if not os.path.exists(sol_path):
        sol_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json"
    data = data.load_replies(sol_path)"""
    else: # submission
        cell9_paths = """    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
        if not os.path.exists(test_path):
            test_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
        data = ArcDataset.from_file(test_path)
    else:
        eval_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
        if not os.path.exists(eval_path):
            eval_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
        data = ArcDataset.from_file(eval_path)
        sol_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json"
        if not os.path.exists(sol_path):
            sol_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json"
        data = data.load_replies(sol_path)"""

    cell9_paths = textwrap.dedent(cell9_paths)
    output_filename = "validation_predictions.json" if validation_mode in ('train', 'eval') else "submission.json"
    if not symbolic_merge:
        # Distinct name so a symbolic-free file is never mistaken for the hybrid output.
        output_filename = "validation_predictions_neural_only.json"

    merge_lines = [
        "import os\n",
        "import json\n",
        "import numpy as np\n",
        "from arc_loader import ArcDataset\n",
        "from arc_decoder import ArcDecoder, score_ensemble\n",
        "import arc_dsl\n",
        "\n",
        'rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")\n',
        "\n",
        cell9_paths + "\n",
        "\n",
        "decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)\n",
        'print("[*] Loading Pass A decoded candidates from /kaggle/inference_outputs...")\n',
        'decoder.load_decoded_results("/kaggle/inference_outputs")\n',
        'if os.path.exists("/kaggle/inference_outputs_b"):\n',
        '    print("[*] Pooling Pass B candidates from /kaggle/inference_outputs_b into ArcDecoder...")\n',
        '    decoder.load_decoded_results("/kaggle/inference_outputs_b", run_name=".b")\n',
        "\n",
        "# Diversity Selection Helper: ensures Attempt 2 is structurally distinct from Attempt 1\n",
        "def sanitize_grid(g):\n",
        "    if g is None:\n",
        "        return [[0]]\n",
        "    try:\n",
        "        arr = np.asarray(g, dtype=int)\n",
        "        if arr.ndim != 2 or arr.size == 0:\n",
        "            return [[0]]\n",
        "        arr = np.clip(arr, 0, 9)\n",
        "        return arr.tolist()\n",
        "    except Exception:\n",
        "        return [[0]]\n",
        "\n",
        "def select_diverse_pair(cand_list, fallback=None):\n",
        "    clean = [c for c in cand_list if c is not None]\n",
        "    if not clean:\n",
        "        fb = sanitize_grid(fallback)\n",
        "        return fb, fb\n",
        "    a1 = sanitize_grid(clean[0])\n",
        "    for c in clean[1:]:\n",
        "        c_clean = sanitize_grid(c)\n",
        "        if np.shape(c_clean) != np.shape(a1):\n",
        "            return a1, c_clean\n",
        "        diff = int(np.sum(np.asarray(c_clean) != np.asarray(a1)))\n",
        "        if diff >= 2 and (diff / np.asarray(a1).size) >= 0.05:\n",
        "            return a1, c_clean\n",
        "    for c in clean[1:]:\n",
        "        c_clean = sanitize_grid(c)\n",
        "        if not np.array_equal(np.asarray(c_clean), np.asarray(a1)):\n",
        "            return a1, c_clean\n",
        "    return a1, a1\n",
        "\n",
        "# Base neural predictions with interleaved probability and voting ensemble\n",
        "neural_raw_results = decoder.run_selection_algo(score_ensemble)\n",
        "submission = data.get_submission(neural_raw_results)\n",
        "\n",
        "# Apply structural diversity reranking across base neural predictions\n",
        "for base_key, ordered_cands in neural_raw_results.items():\n",
        "    if '_' in base_key:\n",
        "        t_id, sub_idx = base_key.rsplit('_', 1)\n",
        "        sub_i = int(sub_idx)\n",
        "        if t_id in submission and sub_i < len(submission[t_id]):\n",
        "            c1, c2 = select_diverse_pair(ordered_cands, fallback=submission[t_id][sub_i].get('attempt_1'))\n",
        "            submission[t_id][sub_i]['attempt_1'] = c1\n",
        "            submission[t_id][sub_i]['attempt_2'] = c2\n",
        "\n",
        "# Tier 1 Symbolic Induction Filter\n",
        'print("*** Running Tier 1 Symbolic Verification pre-pass...")\n',
        "try:\n",
        "    symbolic_solver = arc_dsl.DSLSolver(max_level=2)\n",
        "    tier1_hits = 0\n",
        "\n",
        "    for task_id, task in data.queries.items():\n",
        "        try:\n",
        '            train_pairs = task.get("train", [])\n',
        "            perfect_progs = arc_dsl.search_dsl_programs(train_pairs, max_level=2, max_perfect_programs=2)\n",
        "            if perfect_progs:\n",
        "                best_name = perfect_progs[0][0].name\n",
        '                print(f"[Tier 1 Verified] Task {task_id}: {best_name}")\n',
        "                sym_attempts = symbolic_solver.solve_task(task)\n",
        "                if task_id in submission and len(submission[task_id]) == len(sym_attempts):\n",
        "                    tier1_hits += 1\n",
        "                    for i, sym_item in enumerate(sym_attempts):\n",
        '                        base_cand_1 = submission[task_id][i].get("attempt_1")\n',
        '                        base_cand_2 = submission[task_id][i].get("attempt_2")\n',
        '                        sym_cand = sym_item["attempt_1"]\n',
        "                        submission[task_id][i]['attempt_1'] = sanitize_grid(sym_cand)\n",
        "                        _, div_cand = select_diverse_pair([sym_cand, base_cand_1, base_cand_2], fallback=sym_cand)\n",
        "                        submission[task_id][i]['attempt_2'] = sanitize_grid(div_cand)\n",
        "        except Exception as e_task:\n",
        '            print(f"Warning: Symbolic verification error on task {task_id}: {e_task}")\n',
        "\n",
        '    print(f"*** Tier 1 Symbolic Verified Hits: {tier1_hits}/{len(data.queries)} tasks")\n',
        "except Exception as e_sym:\n",
        '    print(f"Warning: Global symbolic verification error: {e_sym}")\n',
        "\n",
        "# Defensive validation of all submission entries\n",
        "for k in list(submission.keys()):\n",
        "    for item in submission[k]:\n",
        "        item['attempt_1'] = sanitize_grid(item.get('attempt_1'))\n",
        "        item['attempt_2'] = sanitize_grid(item.get('attempt_2'))\n",
        "\n",
        f'with open("{output_filename}", "w") as f:\n',
        "    json.dump(submission, f)\n",
        "\n",
        f'with open("{output_filename}", "r") as f:\n',
        "    reload_submission = json.load(f)\n",
        "\n",
        "if not rerun_mode:\n",
        "    decoder.benchmark_selection_algos()\n",
        '    print("*** Final Hybrid Validation score:", data.validate_submission(reload_submission))\n',
    ]

    if not symbolic_merge:
        # Drop the Tier 1 pre-pass so attempt_1/attempt_2 stay purely neural (for held-out A/B proxies).
        start = merge_lines.index("# Tier 1 Symbolic Induction Filter\n")
        end = next(i for i, line in enumerate(merge_lines) if "Global symbolic verification error" in line)
        del merge_lines[start : end + 1]

    if pass_b_policy == "paired":
        # Skip the notebook's internal benchmark; the paired outputs are scored locally via src.metrics.
        merge_lines = merge_lines[:-3]
        for pass_b_dir, prediction_name in (
            ("/kaggle/inference_outputs_broad", "validation_predictions_broad.json"),
            ("/kaggle/inference_outputs_uncertainty", "validation_predictions_uncertainty.json"),
        ):
            paired_merge_lines = [
                line.replace("/kaggle/inference_outputs_b", pass_b_dir)
                .replace("validation_predictions.json", prediction_name)
                for line in merge_lines
            ]
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": paired_merge_lines,
                }
            )
    else:
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": merge_lines,
            }
        )

    out_nb = {
        "cells": cells,
        "metadata": base_nb.get("metadata", {}),
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    if output_path is None:
        dst_path = (
            REPO_ROOT
            / "notebooks/submissions/hybrid_sota_v1/submission_kernel.ipynb"
        )
    else:
        dst_path = output_path

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(out_nb, indent=1))
    print(f"Successfully generated hybrid submission notebook at {dst_path}")
    return dst_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None, help="Output path for the notebook")
    parser.add_argument("--validation-mode", type=str, choices=["submission", "train", "eval"], default="submission")
    parser.add_argument("--pass-b-policy", type=str, choices=["broad", "uncertainty", "paired"], default="broad")
    parser.add_argument(
        "--task-ids-file",
        type=str,
        default=None,
        help="JSON list of task IDs; restricts a validation run to those tasks.",
    )
    parser.add_argument(
        "--no-symbolic-merge",
        action="store_true",
        help="Skip the Tier 1 symbolic override; write neural-only predictions (validation modes only).",
    )
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else None
    build(
        output_path=out_path,
        validation_mode=args.validation_mode,
        pass_b_policy=args.pass_b_policy,
        symbolic_merge=not args.no_symbolic_merge,
        task_ids=json.loads(Path(args.task_ids_file).read_text()) if args.task_ids_file else None,
    )
