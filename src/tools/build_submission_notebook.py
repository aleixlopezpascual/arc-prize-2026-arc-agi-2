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

    cleaned_prims = "\n".join(
        line
        for line in primitives_py.splitlines()
        if not line.startswith("from ")
        and not line.startswith("import ")
        and not line.startswith("Grid =")
    )
    arc_dsl_content += cleaned_prims + "\n\n"

    cleaned_overlay = "\n".join(
        line
        for line in overlay_py.splitlines()
        if not line.startswith("from ")
        and not line.startswith("import ")
        and not line.startswith("Grid =")
    )
    arc_dsl_content += cleaned_overlay + "\n\n"

    cleaned_ca = "\n".join(
        line
        for line in ca_py.splitlines()
        if not line.startswith("from ")
        and not line.startswith("import ")
        and not line.startswith("Grid =")
    )
    arc_dsl_content += cleaned_ca + "\n\n"

    cleaned_search = "\n".join(
        line
        for line in search_py.splitlines()
        if not line.strip().startswith("from ")
        and not line.strip().startswith("import ")
        and not line.strip().startswith("Grid =")
        and not line.strip().startswith("try:")
        and not line.strip().startswith("except ")
        and not line.strip().startswith("pass")
    )
    cleaned_search = cleaned_search.replace("P.", "")
    arc_dsl_content += cleaned_search + "\n\n"

    cleaned_solver = "\n".join(
        line
        for line in solver_py.splitlines()
        if not line.startswith("from ")
        and not line.startswith("import ")
        and not line.startswith("Grid =")
    )
    cleaned_solver = cleaned_solver.replace("P.", "")
    arc_dsl_content += cleaned_solver + "\n"

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

    # Cell 1: Buffer deadline
    cells.append(base_nb["cells"][1])

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

    # Cell 5: arc_decoder.py
    cells.append(base_nb["cells"][4])

    # Cell 6: arc_solver.py
    cells.append(base_nb["cells"][5])

    # Cell 7: starter.py
    cells.append(base_nb["cells"][6])

    # Cell 8: run starter.py
    cells.append(base_nb["cells"][7])

    # Cell 9: Hybrid merge and submission
    merge_lines = [
        "import os\n",
        "import json\n",
        "import numpy as np\n",
        "from arc_loader import ArcDataset\n",
        "from arc_decoder import ArcDecoder\n",
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
        "# Base neural predictions\n",
        "submission = data.get_submission(decoder.run_selection_algo())\n",
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
        "nbformat_minor": 5,
    }

    dst_path = Path(
        "/Users/aleix.lopez/arc-prize-2026-arc-agi-2/notebooks/submissions/hybrid_sota_v1/submission_kernel.ipynb"
    )
    dst_path.write_text(json.dumps(out_nb, indent=1))
    print(f"Successfully generated hybrid submission notebook at {dst_path}")


if __name__ == "__main__":
    build()
