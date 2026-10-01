from __future__ import annotations

from typing import Any
import numpy as np

from src.dsl import primitives as P
from src.dsl.search import search_dsl_programs
from src.metrics import grid_equal

Grid = list[list[int]]


class DSLSolver:
    """End-to-end ARC puzzle solver using symbolic DSL search and robust fallbacks."""

    def __init__(self, max_level: int = 2) -> None:
        self.max_level = max_level

    def solve_task(self, task: dict[str, Any]) -> list[dict[str, Grid]]:
        """Solves a single ARC task across all its test pairs."""
        train_pairs = task.get("train", [])
        test_pairs = task.get("test", [])

        # 1. Search for programs solving 100% of demonstration pairs
        perfect_programs = search_dsl_programs(
            train_pairs, max_level=self.max_level, max_perfect_programs=4
        )

        results: list[dict[str, Grid]] = []

        for test_item in test_pairs:
            test_in = test_item["input"]
            attempt_1: Grid | None = None
            attempt_2: Grid | None = None

            # Try generating attempt_1 from top program
            if perfect_programs:
                best_prog, _ = perfect_programs[0]
                try:
                    cand = best_prog.execute(test_in)
                    if len(cand) > 0 and len(cand[0]) > 0:
                        attempt_1 = cand
                except Exception:
                    attempt_1 = None

            # Try generating attempt_2 from subsequent distinct program
            if perfect_programs and len(perfect_programs) > 1:
                for prog, _ in perfect_programs[1:]:
                    try:
                        cand = prog.execute(test_in)
                        if len(cand) > 0 and len(cand[0]) > 0:
                            if attempt_1 is None or not grid_equal(cand, attempt_1):
                                attempt_2 = cand
                                break
                    except Exception:
                        continue

            # Fallbacks if attempts are still missing
            if attempt_1 is None:
                attempt_1 = [row[:] for row in test_in]

            if attempt_2 is None:
                # Try horizontal reflection
                alt = P.reflect_horizontal(test_in)
                if not grid_equal(alt, attempt_1):
                    attempt_2 = alt
                else:
                    # Try vertical reflection
                    alt2 = P.reflect_vertical(test_in)
                    if not grid_equal(alt2, attempt_1):
                        attempt_2 = alt2
                    else:
                        attempt_2 = [row[:] for row in test_in]

            results.append({"attempt_1": attempt_1, "attempt_2": attempt_2})

        return results

    def predict_all(self, challenges: dict[str, Any]) -> dict[str, list[dict[str, Grid]]]:
        """Runs the solver on all tasks in the challenges dictionary."""
        preds: dict[str, list[dict[str, Grid]]] = {}
        for task_id, task in challenges.items():
            preds[task_id] = self.solve_task(task)
        return preds
