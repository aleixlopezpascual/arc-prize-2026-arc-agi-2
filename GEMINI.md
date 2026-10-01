# Gemini CLI Guidelines: ARC Prize 2026 - ARC-AGI-2

See `AGENTS.md` for overall repo architectural rules and validation protocols.

## Key Directives for Gemini CLI
1. **Target Benchmark:** ARC-AGI-2 (`input/arc-agi_evaluation_challenges.json` + `arc-agi_evaluation_solutions.json`).
2. **Local Evaluation:** Run evaluation via `src/metrics.py`. Never estimate accuracy without running the exact metric.
3. **Submission Verification:** Always validate predictions with `src/validator.py` before exporting a submission file.
4. **Time & Memory Bounds:** Always ensure per-task processing has hard timeouts and explicit garbage collection.
