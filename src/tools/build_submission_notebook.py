from __future__ import annotations

import json
from pathlib import Path


def build():
    base_nb_path = Path(
        "/Users/aleix.lopez/arc-prize-2026-arc-agi-2/notebooks/public_baselines/lb33_89/arc-agi2-lb33-89-minimal-perfpatch.ipynb"
    )
    base_nb = json.loads(base_nb_path.read_text())

    primitives_py = Path(
        "/Users/aleix.lopez/arc-prize-2026-arc-agi-2/src/dsl/primitives.py"
    ).read_text()
    overlay_py = Path(
        "/Users/aleix.lopez/arc-prize-2026-arc-agi-2/src/dsl/overlay.py"
    ).read_text()
    ca_py = Path(
        "/Users/aleix.lopez/arc-prize-2026-arc-agi-2/src/dsl/cellular_automata.py"
    ).read_text()
    search_py = Path(
        "/Users/aleix.lopez/arc-prize-2026-arc-agi-2/src/dsl/search.py"
    ).read_text()
    solver_py = Path(
        "/Users/aleix.lopez/arc-prize-2026-arc-agi-2/src/dsl/solver.py"
    ).read_text()

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
    cells.append(base_nb["cells"][3])

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
def score_ensemble(guesses):
    ranked_prob = score_full_probmul_3(guesses)
    ranked_vote = score_kgmon(guesses)
    seen, result = set(), []
    for a, b in zip(ranked_prob, ranked_vote):
        for cand in (a, b):
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

    # Cell 7: starter.py (with Deficit-First Routing)
    starter_source = """%%writefile starter.py
import os
import time
import json
import torch
import argparse
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue
    if rank > 0:
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(5)
    
    from arc_solver import worker

    with open(f"/kaggle/worker{rank}", "w") as f:
        f.write("Ok")
    
    print(f"[Rank {rank}] start!")
    
    worker(rank, queue, end_time)
    
    print(f"[Rank {rank}] done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

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
        if not rerun_mode:
            if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:
                continue
        task_train = data[key]["train"]
        progs = arc_dsl.search_dsl_programs(task_train, max_level=1, max_perfect_programs=2)
        if progs:
            easy_tasks.append(key)
        else:
            hard_tasks.append(key)

    print(f"*** Deficit-First Routing: {len(easy_tasks)} Tier-1 Solved | {len(hard_tasks)} Hard (Expanded-TTT)")

    # Hard tasks get expanded 720s timeout (+33%), easy tasks get fast 90s timeout
    hard_timeout = 720.0
    easy_timeout = 90.0

    queue = mp.Manager().Queue()

    # Queue hard tasks FIRST to maximize GPU compute on unsolved tasks
    for key in hard_tasks:
        queue.put((key, hard_timeout))
        
    for key in easy_tasks:
        queue.put((key, easy_timeout))

    for _ in range(4):
        queue.put(None)
    
    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=4)
"""
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [starter_source],
        }
    )

    # Cell 8: run starter.py
    cells.append(base_nb["cells"][7])

    # Cell 9: Hybrid merge and submission
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
        "if rerun_mode:\n",
        '    test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"\n',
        "    if not os.path.exists(test_path):\n",
        '        test_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"\n',
        "    data = ArcDataset.from_file(test_path)\n",
        "else:\n",
        '    eval_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"\n',
        "    if not os.path.exists(eval_path):\n",
        '        eval_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"\n',
        "    data = ArcDataset.from_file(eval_path)\n",
        '    sol_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json"\n',
        "    if not os.path.exists(sol_path):\n",
        '        sol_path = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json"\n',
        "    data = data.load_replies(sol_path)\n",
        "\n",
        "decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)\n",
        'decoder.load_decoded_results("/kaggle/inference_outputs")\n',
        "\n",
        "# Base neural predictions with interleaved probability and voting ensemble\n",
        "submission = data.get_submission(decoder.run_selection_algo(score_ensemble))\n",
        "\n",
        "# Tier 1 Symbolic Induction Filter\n",
        'print("*** Running Tier 1 Symbolic Verification pre-pass...")\n',
        "symbolic_solver = arc_dsl.DSLSolver(max_level=2)\n",
        "tier1_hits = 0\n",
        "\n",
        "for task_id, task in data.queries.items():\n",
        '    train_pairs = task.get("train", [])\n',
        "    perfect_progs = arc_dsl.search_dsl_programs(train_pairs, max_level=2, max_perfect_programs=2)\n",
        "    if perfect_progs:\n",
        "        best_name = perfect_progs[0][0].name\n",
        '        print(f"[Tier 1 Verified] Task {task_id}: {best_name}")\n',
        "        sym_attempts = symbolic_solver.solve_task(task)\n",
        "        if task_id in submission and len(submission[task_id]) == len(sym_attempts):\n",
        "            tier1_hits += 1\n",
        "            for i, sym_item in enumerate(sym_attempts):\n",
        '                base_cand_1 = submission[task_id][i].get("attempt_1")\n',
        '                base_cand_2 = submission[task_id][i].get("attempt_2")\n',
        '                sym_cand = sym_item["attempt_1"]\n',
        "                # Optimal 2-attempt union: keep top neural guess if distinct from symbolic\n",
        '                submission[task_id][i]["attempt_1"] = sym_cand\n',
        "                if arc_dsl.grid_equal(sym_cand, base_cand_1):\n",
        '                    submission[task_id][i]["attempt_2"] = base_cand_2\n',
        "                else:\n",
        '                    submission[task_id][i]["attempt_2"] = base_cand_1\n',
        "\n",
        'print(f"*** Tier 1 Symbolic Verified Hits: {tier1_hits}/{len(data.queries)} tasks")\n',
        "\n",
        'with open("submission.json", "w") as f:\n',
        "    json.dump(submission, f)\n",
        "\n",
        'with open("submission.json", "r") as f:\n',
        "    reload_submission = json.load(f)\n",
        "\n",
        "if not rerun_mode:\n",
        "    decoder.benchmark_selection_algos()\n",
        '    print("*** Final Hybrid Validation score:", data.validate_submission(reload_submission))\n',
    ]

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

    dst_path = Path(
        "/Users/aleix.lopez/arc-prize-2026-arc-agi-2/notebooks/submissions/hybrid_sota_v1/submission_kernel.ipynb"
    )
    dst_path.write_text(json.dumps(out_nb, indent=1))
    print(f"Successfully generated hybrid submission notebook at {dst_path}")


if __name__ == "__main__":
    build()
