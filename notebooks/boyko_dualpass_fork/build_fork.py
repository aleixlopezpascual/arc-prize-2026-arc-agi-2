"""Build our fork of evgendvorkin/arc-2-2026-4-4-local-boyko-nvarc (dual-pass, public LB 31.81-32.64 over V17-V20).

Upstream notebook is copied unchanged except for two inserted cells right after the budget cell:
  1. ARC_ASTRA_PROMOTE defaults to 0: the V21/V22 Astra expert was never scored, so we run the
     proven V17-V20 neural-only selection (expert still runs in shadow mode for diagnostics).
  2. An early identity-grid submission.json, so a run killed before the finalizer never scores 0.00.
Usage: python build_fork.py <upstream.ipynb> <upstream kernel-metadata.json>
"""
import copy
import json
import sys
from pathlib import Path

OUT = Path(__file__).parent

PROMOTE_CELL = """# [fork] Run the scored V17-V20 configuration: Astra expert in shadow mode only.
import os
os.environ.setdefault('ARC_ASTRA_PROMOTE', '0')
print('[fork] ARC_ASTRA_PROMOTE =', os.environ['ARC_ASTRA_PROMOTE'])
"""

FALLBACK_CELL = """# [fork] Early fallback: a complete, valid submission.json before any GPU work.
# The finalizer overwrites it; if the run is killed first, identity grids still score > 0.00.
import json, os
_root = '/kaggle/input/competitions/arc-prize-2026-arc-agi-2'
if not os.path.isdir(_root):
    _root = '/kaggle/input/arc-prize-2026-arc-agi-2'
_rerun = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))
with open(os.path.join(_root, 'arc-agi_test_challenges.json' if _rerun else 'arc-agi_evaluation_challenges.json')) as _f:
    _chal = json.load(_f)
_fallback = {k: [{'attempt_1': t['input'], 'attempt_2': t['input']} for t in v['test']] for k, v in _chal.items()}
os.makedirs('/kaggle/working', exist_ok=True)
with open('/kaggle/working/submission.json', 'w') as _f:
    json.dump(_fallback, _f)
print(f'[fork] fallback submission.json written: {len(_fallback)} tasks')
"""


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def main(nb_path, meta_path):
    nb = json.loads(Path(nb_path).read_text())
    budget_idx = next(i for i, c in enumerate(nb["cells"])
                      if c["cell_type"] == "code" and "global_end_time = RUN_START + MAX_SECONDS" in "".join(c["source"]))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
    nb["cells"][budget_idx + 1:budget_idx + 1] = [code_cell(PROMOTE_CELL), code_cell(FALLBACK_CELL)]
    nb.setdefault("metadata", {}).setdefault("kaggle", {})["isGpuEnabled"] = True
    (OUT / "submission_kernel.ipynb").write_text(json.dumps(nb, indent=1))

    meta = json.loads(Path(meta_path).read_text())
    fork = copy.deepcopy(meta)
    for k in ("id_no",):
        fork.pop(k, None)
    fork.update(id="aleixlopez/arc-agi2-boyko-dualpass-fork", title="ARC AGI2 Boyko DualPass Fork",
                code_file="submission_kernel.ipynb", is_private=True,
                docker_image_pinning_type="original")
    (OUT / "kernel-metadata.json").write_text(json.dumps(fork, indent=2))
    print("fork written; inserted cells after index", budget_idx)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
