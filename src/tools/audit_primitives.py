"""Audit how well DSL programs generalize across ARC splits.

For every task, collects ALL programs that fit 100% of the demonstration pairs
(not just the top-ranked ones), runs them on the test inputs and records whether
each one reproduces the ground-truth test outputs. Results are dumped as JSON so
`summarize()` can group them per program family and per split.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import re
import signal
import time
from pathlib import Path
from typing import Any

from src import config
from src.dsl.search import search_dsl_programs
from src.metrics import grid_equal

TASK_TIMEOUT_S = 30
MAX_PERFECT = 200

SPLITS = {
    "train": ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json"),
    "eval": ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"),
    "test": ("arc-agi_test_challenges.json", None),
}


class _Timeout(Exception):
    pass


def _on_alarm(signum: int, frame: Any) -> None:
    raise _Timeout()


def family_of(name: str) -> str:
    """Strips numeric/parameter suffixes so parameterised variants collapse."""
    parts = [p.strip() for p in name.split(">>")]
    # D4 wrapper: "rot90 >> inner >> inv_rot90" -> keep inner programs only
    parts = [p for p in parts if not p.startswith("inv_") and p not in {"rot90", "rot180", "rot270", "fliplr", "flipud"}] or parts
    fams = []
    for p in parts:
        p = re.sub(r"(_(bg|c|k|n|d|t|m|s)?-?\d+)+", "", p)
        p = re.sub(r"\(.*\)", "", p)
        fams.append(p)
    return " >> ".join(fams)


def _audit_task(job: tuple[str, dict[str, Any], list[list[list[int]]] | None]) -> dict[str, Any]:
    task_id, task, solutions = job
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(TASK_TIMEOUT_S)
    rec: dict[str, Any] = {"task_id": task_id, "programs": [], "error": None, "elapsed": 0.0}
    t0 = time.time()
    try:
        progs = search_dsl_programs(task["train"], max_level=3, max_perfect_programs=MAX_PERFECT)
        for prog, _ in progs:
            outs_ok: list[bool | None] = []
            for i, t in enumerate(task["test"]):
                try:
                    pred = prog.execute(t["input"])
                    outs_ok.append(None if solutions is None else bool(grid_equal(pred, solutions[i])))
                except Exception:
                    outs_ok.append(False if solutions is not None else None)
            rec["programs"].append(
                {
                    "name": prog.name,
                    "family": family_of(prog.name),
                    "steps": len(prog.steps),
                    "test_ok": outs_ok,
                }
            )
    except _Timeout:
        rec["error"] = "timeout"
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        signal.alarm(0)
        rec["elapsed"] = round(time.time() - t0, 3)
        gc.collect()
    return rec


def run_split(split: str, limit: int | None, workers: int) -> list[dict[str, Any]]:
    chal_f, sol_f = SPLITS[split]
    chal = json.loads((config.INPUT_DIR / chal_f).read_text())
    sols = json.loads((config.INPUT_DIR / sol_f).read_text()) if sol_f else {}
    ids = sorted(chal)[:limit] if limit else sorted(chal)
    jobs = [(tid, chal[tid], sols.get(tid)) for tid in ids]
    with mp.Pool(workers, maxtasksperchild=25) as pool:
        return list(pool.imap_unordered(_audit_task, jobs, chunksize=1))


def summarize(records_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Per-family stats: fires/correct tasks per split."""
    fam: dict[str, dict[str, Any]] = {}
    for split, recs in records_by_split.items():
        for rec in recs:
            seen: dict[str, tuple[bool, bool]] = {}
            for p in rec["programs"]:
                oks = p["test_ok"]
                full = all(o is True for o in oks) if oks and oks[0] is not None else False
                prev = seen.get(p["family"], (True, False))
                seen[p["family"]] = (True, prev[1] or full)
            for f, (fired, correct) in seen.items():
                d = fam.setdefault(f, {})
                s = d.setdefault(split, {"fires": [], "correct": []})
                s["fires"].append(rec["task_id"])
                if correct:
                    s["correct"].append(rec["task_id"])
    return fam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "eval", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("docs/intel/primitive_audit_raw.json"))
    args = ap.parse_args()

    out: dict[str, list[dict[str, Any]]] = {}
    for split in args.splits:
        t0 = time.time()
        out[split] = run_split(split, args.limit, args.workers)
        errs = sum(1 for r in out[split] if r["error"])
        print(f"[{split}] {len(out[split])} tasks, {errs} errors/timeouts, {time.time() - t0:.1f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
