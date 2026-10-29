"""Held-out A/B proxy: does letting the symbolic DSL take an attempt slot help?

Local eval is contaminated (see docs/intel/primitive_audit.md), so this tool
scores on the *train* split, which the v14+ primitives never targeted, using the
cached neural Pass-B 5-fold predictions (`oof/val_broad/validation_predictions.json`).

For every test output it compares attempt policies on identical tasks:
  neural        (N1, N2)                                   - no symbolic
  production    select_diverse_attempt_pair([S, N1, N2])   - what the notebook ships
  union         (S, N1) or (S, N2) if S == N1              - documented union protocol
  union_no_ca   union, but cellular-automata programs never fire
  union_loo     union only if leave-one-task-out family precision >= threshold

`S` is the top-ranked symbolic program (max_level=2, matching the notebook), or
absent when no program fits the demos. Scoring uses `src.metrics` only.

The A/B needs a *symbolic-free* neural reference. `check_neural_reference` refuses
files whose attempt_1 already equals S (the hybrid output): `oof/val_broad/
validation_predictions.json` is such a file (200/200 fired outputs), so every
policy would trivially tie. Use `--split eval --generic-only` with a neural-only
eval run instead; `--generic-only` drops families that never fire on train.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import signal
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from src import config
from src.dsl.search import search_dsl_programs
from src.metrics import compute_competition_metric, grid_equal
from src.models.diversity_reranker import select_diverse_attempt_pair
from src.tools.audit_primitives import family_of

Grid = list[list[int]]
Attempts = dict[str, Grid]

TASK_TIMEOUT_S = 30
MAX_LEVEL = 2
LOO_THRESHOLD = 0.5
DEFAULT_NEURAL = config.OOF_DIR / "val_broad" / "validation_predictions.json"
DEFAULT_CACHE = config.OOF_DIR / "heldout_symbolic_{split}.json"


class _Timeout(Exception):
    pass


def _on_alarm(signum: int, frame: Any) -> None:
    raise _Timeout()


def _symbolic_task(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    """Top-ranked fitting program and its prediction for each test input."""
    task_id, task = job
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(TASK_TIMEOUT_S)
    rec: dict[str, Any] = {"task_id": task_id, "name": None, "family": None, "preds": None, "error": None}
    try:
        progs = search_dsl_programs(task["train"], max_level=MAX_LEVEL, max_perfect_programs=4)
        if progs:
            prog = progs[0][0]
            preds: list[Grid] = []
            for t in task["test"]:
                try:
                    cand = prog.execute(t["input"])
                    ok = len(cand) > 0 and len(cand[0]) > 0
                except Exception:
                    ok = False
                # DSLSolver falls back to the input grid when execution fails.
                preds.append([list(map(int, r)) for r in cand] if ok else [r[:] for r in t["input"]])
            rec.update(name=prog.name, family=family_of(prog.name), preds=preds)
    except _Timeout:
        rec["error"] = "timeout"
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        signal.alarm(0)
        gc.collect()
    return rec


def collect_symbolic(split: str, workers: int, limit: int | None = None) -> dict[str, dict[str, Any]]:
    chal_path = {
        "train": config.TRAIN_CHALLENGES_PATH,
        "eval": config.EVAL_CHALLENGES_PATH,
        "test": config.TEST_CHALLENGES_PATH,
    }[split]
    chal = json.loads(chal_path.read_text())
    ids = sorted(chal)[:limit] if limit else sorted(chal)
    with mp.Pool(workers, maxtasksperchild=25) as pool:
        recs = list(pool.imap_unordered(_symbolic_task, [(i, chal[i]) for i in ids], chunksize=1))
    return {r["task_id"]: r for r in recs}


# ---------------------------------------------------------------- policies


def _is_ca(sym: dict[str, Any]) -> bool:
    return "cellular_automata" in (sym.get("family") or "")


def _union(s: Grid, n1: Grid, n2: Grid) -> tuple[Grid, Grid]:
    return (s, n2) if grid_equal(s, n1) else (s, n1)


def policy_attempts(
    name: str,
    sym: dict[str, Any] | None,
    idx: int,
    n1: Grid,
    n2: Grid,
    loo_precision: float | None = None,
) -> tuple[Grid, Grid]:
    """Returns (attempt_1, attempt_2) for one test output under a named policy."""
    fires = bool(sym and sym.get("preds"))
    if name == "neural" or not fires:
        return n1, n2
    assert sym is not None and sym["preds"] is not None
    s = sym["preds"][idx]
    if name == "production":
        return select_diverse_attempt_pair([s, n1, n2], fallback_input=s)
    if name == "union":
        return _union(s, n1, n2)
    if name == "union_no_ca":
        return (n1, n2) if _is_ca(sym) else _union(s, n1, n2)
    if name == "union_loo":
        ok = loo_precision is not None and loo_precision >= LOO_THRESHOLD
        return _union(s, n1, n2) if ok else (n1, n2)
    raise ValueError(f"unknown policy: {name}")


POLICIES = ("neural", "production", "union", "union_no_ca", "union_loo")


def check_neural_reference(
    symbolic: dict[str, dict[str, Any]], neural: dict[str, list[Attempts]]
) -> tuple[int, int]:
    """Counts fired outputs whose neural attempt_1 already equals the symbolic S.

    A neural-only reference agrees with S only by chance; a hybrid output agrees
    on (nearly) every fired output. Returns (agreeing, fired).
    """
    agree = fired = 0
    for tid, rec in symbolic.items():
        if not rec.get("preds") or tid not in neural:
            continue
        for i, s in enumerate(rec["preds"]):
            fired += 1
            agree += grid_equal(s, neural[tid][i]["attempt_1"])
    return agree, fired


def restrict_to_families(
    symbolic: dict[str, dict[str, Any]], allowed: set[str]
) -> dict[str, dict[str, Any]]:
    """Drops symbolic hits whose family is not in `allowed` (they stop firing)."""
    out = {}
    for tid, rec in symbolic.items():
        keep = rec.get("preds") and rec.get("family") in allowed
        out[tid] = rec if keep else {**rec, "name": None, "family": None, "preds": None}
    return out


def family_loo_precision(
    symbolic: dict[str, dict[str, Any]], solutions: dict[str, list[Grid]]
) -> dict[tuple[str, str], float]:
    """Per (task, family) leave-one-task-out precision of the symbolic top-1.

    Laplace-smoothed: (correct + 0.5) / (n + 1), so an unseen family sits at 0.5.
    """
    per_task: dict[str, tuple[str, int, int]] = {}
    for tid, rec in symbolic.items():
        if not rec.get("preds") or tid not in solutions:
            continue
        sols = solutions[tid]
        correct = sum(grid_equal(p, sols[i]) for i, p in enumerate(rec["preds"]))
        per_task[tid] = (rec["family"], correct, len(rec["preds"]))
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for fam, c, n in per_task.values():
        totals[fam][0] += c
        totals[fam][1] += n
    out: dict[tuple[str, str], float] = {}
    for tid, (fam, c, n) in per_task.items():
        tc, tn = totals[fam][0] - c, totals[fam][1] - n
        out[(tid, fam)] = (tc + 0.5) / (tn + 1)
    return out


def score_policies(
    symbolic: dict[str, dict[str, Any]],
    neural: dict[str, list[Attempts]],
    solutions: dict[str, list[Grid]],
    policies: tuple[str, ...] = POLICIES,
) -> dict[str, dict[str, Any]]:
    """Scores each policy with `src.metrics.compute_competition_metric`."""
    task_ids = [t for t in sorted(solutions) if t in neural]
    sub_solutions = {t: solutions[t] for t in task_ids}
    loo = family_loo_precision({t: symbolic[t] for t in task_ids if t in symbolic}, sub_solutions)
    results: dict[str, dict[str, Any]] = {}
    preds_by_policy: dict[str, dict[str, list[Attempts]]] = {}
    for pol in policies:
        preds: dict[str, list[Attempts]] = {}
        for tid in task_ids:
            sym = symbolic.get(tid)
            lp = loo.get((tid, sym["family"])) if sym and sym.get("family") else None
            per_out = []
            for i, n in enumerate(neural[tid]):
                a1, a2 = policy_attempts(pol, sym, i, n["attempt_1"], n["attempt_2"], lp)
                per_out.append({"attempt_1": a1, "attempt_2": a2})
            preds[tid] = per_out
        preds_by_policy[pol] = preds
        m = compute_competition_metric(preds, sub_solutions)
        results[pol] = {
            "correct": m["correct_test_outputs"],
            "total": m["total_test_outputs"],
            "accuracy": m["accuracy"],
            "per_task": m["per_task_detail"],
        }
    # Paired outcome vs. neural-only: gains and losses per test output.
    base = results["neural"]["per_task"]
    for pol, res in results.items():
        gain = loss = 0
        for tid in task_ids:
            for i, ok in enumerate(res["per_task"][tid]["test_results"]):
                b = base[tid]["test_results"][i]
                gain += ok and not b
                loss += b and not ok
        res.update(gains=gain, losses=loss, net=gain - loss)
    return results


def family_table(
    symbolic: dict[str, dict[str, Any]],
    neural: dict[str, list[Attempts]],
    solutions: dict[str, list[Grid]],
) -> list[dict[str, Any]]:
    """Per symbolic family: how often S, N1 and N2 are right on the tasks it fires on."""
    agg: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for tid, rec in symbolic.items():
        if not rec.get("preds") or tid not in solutions or tid not in neural:
            continue
        for i, s in enumerate(rec["preds"]):
            sol = solutions[tid][i]
            n = neural[tid][i]
            a = agg[rec["family"]]
            a["outputs"] += 1
            a["S"] += grid_equal(s, sol)
            a["N1"] += grid_equal(n["attempt_1"], sol)
            a["N2_only"] += grid_equal(n["attempt_2"], sol) and not grid_equal(n["attempt_1"], sol)
            a["S_only"] += grid_equal(s, sol) and not (
                grid_equal(n["attempt_1"], sol) or grid_equal(n["attempt_2"], sol)
            )
    return sorted(({"family": f, **dict(v)} for f, v in agg.items()), key=lambda r: -r["outputs"])


SPLIT_FILES = {
    "train": (config.TRAIN_CHALLENGES_PATH, config.TRAIN_SOLUTIONS_PATH),
    "eval": (config.EVAL_CHALLENGES_PATH, config.EVAL_SOLUTIONS_PATH),
}


def _load_symbolic(split: str, workers: int, limit: int | None, refresh: bool) -> dict[str, dict[str, Any]]:
    cache = Path(str(DEFAULT_CACHE).format(split=split))
    if cache.exists() and not refresh and not limit:
        return json.loads(cache.read_text())
    t0 = time.time()
    symbolic = collect_symbolic(split, workers, limit)
    errs = sum(1 for r in symbolic.values() if r["error"])
    print(f"[{split}] symbolic search: {len(symbolic)} tasks, {errs} errors/timeouts, {time.time() - t0:.1f}s")
    if not limit:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(symbolic))
    return symbolic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=sorted(SPLIT_FILES), default="train")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--neural", type=Path, default=DEFAULT_NEURAL, help="neural-only predictions, task -> attempts")
    ap.add_argument("--generic-only", action="store_true", help="keep only families that also fire on train")
    ap.add_argument("--refresh", action="store_true", help="recompute the symbolic cache")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    symbolic = _load_symbolic(args.split, args.workers, args.limit, args.refresh)
    if args.generic_only:
        train_sym = _load_symbolic("train", args.workers, None, False)
        generic = {r["family"] for r in train_sym.values() if r.get("preds")}
        before = sum(1 for r in symbolic.values() if r.get("preds"))
        symbolic = restrict_to_families(symbolic, generic)
        print(f"[{args.split}] --generic-only: {before} -> {sum(1 for r in symbolic.values() if r.get('preds'))} fired tasks")

    neural = json.loads(args.neural.read_text())
    solutions = json.loads(SPLIT_FILES[args.split][1].read_text())
    if args.limit:
        solutions = {t: solutions[t] for t in symbolic}

    agree, fired = check_neural_reference(symbolic, neural)
    print(f"neural reference: attempt_1 == S on {agree}/{fired} fired outputs")
    if fired and agree / fired >= 0.95:
        raise SystemExit(
            f"INVALID A/B: {args.neural} already contains the symbolic prediction as attempt_1 "
            f"({agree}/{fired}); every policy would tie. Provide a neural-only run."
        )

    res = score_policies(symbolic, neural, solutions)
    fams = family_table(symbolic, neural, solutions)
    print(f"\n{'policy':<14}{'correct':>9}{'acc':>9}{'gains':>7}{'losses':>8}{'net':>6}")
    for pol in POLICIES:
        r = res[pol]
        print(f"{pol:<14}{r['correct']:>5}/{r['total']:<4}{r['accuracy']:>8.2%}{r['gains']:>7}{r['losses']:>8}{r['net']:>+6}")
    print(f"\n{'family (fires)':<46}{'n':>4}{'S':>4}{'N1':>4}{'S_only':>7}{'N2_only':>8}")
    for f in fams[:15]:
        print(f"{f['family'][:45]:<46}{f['outputs']:>4}{f['S']:>4}{f['N1']:>4}{f['S_only']:>7}{f['N2_only']:>8}")

    out = args.out or config.OOF_DIR / f"heldout_proxy_report_{args.split}.json"
    slim = {p: {k: v for k, v in r.items() if k != "per_task"} for p, r in res.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"policies": slim, "families": fams}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
