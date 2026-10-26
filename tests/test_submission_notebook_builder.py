import json
from pathlib import Path

import pytest

from src.tools.build_submission_notebook import build


def test_builder_train_mode_broad_policy(tmp_path: Path):
    """Test train mode with broad policy."""
    output_path = tmp_path / "submission_train.ipynb"
    
    # API should accept output_path, validation_mode, pass_b_policy
    returned_path = build(
        output_path=output_path,
        validation_mode='train',
        pass_b_policy='broad'
    )
    
    assert returned_path == output_path
    assert output_path.exists()
    
    with open(output_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code_cells = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    full_code = "\n".join(code_cells)
    
    # (1) train mode loads training datasets
    assert "arc-agi_training_challenges.json" in full_code
    assert "arc-agi_training_solutions.json" in full_code
    
    # Processes full dataset, ensuring validation mode is correct
    assert "if 'train' == 'train':" in full_code
    assert "if 'train' == 'submission' and not rerun_mode:" in full_code
    
    # (3) broad policy preserves existing broad Pass-B behavior (no uncertainty file)
    assert "/kaggle/working/pass_b_uncertainty.json" not in full_code
    assert "pass_priority.py" not in full_code
    
    # (5) generated notebook does not call kaggle competitions submit
    assert "kaggle competitions submit" not in full_code
    
    # Assert starter.py compiles
    starter_cell = None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if source.startswith("%%writefile starter.py"):
                starter_cell = source
                break
    assert starter_cell is not None
    starter_code = "\n".join(starter_cell.splitlines()[1:])
    compile(starter_code, '<string>', 'exec')

def test_builder_eval_mode_uncertainty_policy(tmp_path: Path):
    """Test eval mode with uncertainty policy."""
    output_path = tmp_path / "submission_eval.ipynb"
    
    returned_path = build(
        output_path=output_path,
        validation_mode='eval',
        pass_b_policy='uncertainty'
    )
    
    assert returned_path == output_path
    assert output_path.exists()
    
    with open(output_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code_cells = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    full_code = "\n".join(code_cells)
    
    # (2) eval mode uses the official evaluation challenges/solutions
    assert "arc-agi_evaluation_challenges.json" in full_code
    assert "arc-agi_evaluation_solutions.json" in full_code
    
    # Processes full 120-task route, ensuring validation mode is correct
    assert "if 'eval' == 'eval':" in full_code
    assert "if 'eval' == 'submission' and not rerun_mode:" in full_code
    assert "validation_predictions.json" in full_code
    
    # (4) uncertainty policy computes/writes a Pass-A uncertainty map
    assert "uncertainty" in full_code.lower()
    assert "rsplit('_', 1)" in full_code
    assert "task_unc_list[task_id].append(unc)" in full_code
    assert "sum(uncs) / len(uncs)" in full_code
    assert "ArcDecoder" in full_code
    assert "n_guesses=2" in full_code
    assert "/kaggle/working/pass_b_uncertainty.json" in full_code
    assert "/kaggle/pass_b_uncertainty.json" not in full_code
    
    # (5) generated notebook does not call kaggle competitions submit
    assert "kaggle competitions submit" not in full_code
    
    # Assert starter.py compiles and no placeholder
    starter_cell = None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if source.startswith("%%writefile starter.py"):
                starter_cell = source
                break
    assert starter_cell is not None
    starter_code = "\n".join(starter_cell.splitlines()[1:])
    compile(starter_code, '<string>', 'exec')
    assert "{uncertainty_routing}" not in full_code


def test_train_mode_routes_arc_solver_workers_to_training_tasks(tmp_path: Path):
    output_path = tmp_path / "training_worker_route.ipynb"
    build(output_path=output_path, validation_mode="train", pass_b_policy="broad")
    nb = json.loads(output_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    solver_cell = next(source for source in code_cells if source.startswith("%%writefile arc_solver.py"))
    solver_source = "\n".join(solver_cell.splitlines()[1:])
    runner_source = next(source for source in code_cells if "env_a = dict(" in source)

    compile(solver_source, "arc_solver.py", "exec")
    assert 'validation_mode = os.getenv("ARC_VALIDATION_MODE", "submission")' in solver_source
    assert "arc-agi_training_challenges.json" in solver_source
    assert runner_source.count('ARC_VALIDATION_MODE="train"') == 2


def test_eval_mode_routes_arc_solver_workers_to_evaluation_tasks(tmp_path: Path):
    output_path = tmp_path / "evaluation_worker_route.ipynb"
    build(output_path=output_path, validation_mode="eval", pass_b_policy="broad")
    nb = json.loads(output_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    solver_cell = next(source for source in code_cells if source.startswith("%%writefile arc_solver.py"))
    solver_source = "\n".join(solver_cell.splitlines()[1:])
    runner_source = next(source for source in code_cells if "env_a = dict(" in source)

    compile(solver_source, "arc_solver.py", "exec")
    assert 'validation_mode = os.getenv("ARC_VALIDATION_MODE", "submission")' in solver_source
    assert "arc-agi_evaluation_challenges.json" in solver_source
    assert runner_source.count('ARC_VALIDATION_MODE="eval"') == 2


def test_uncertainty_map_failure_stops_before_pass_b(tmp_path: Path):
    output_path = tmp_path / "uncertainty_fail_closed.ipynb"
    build(output_path=output_path, validation_mode="train", pass_b_policy="uncertainty")
    nb = json.loads(output_path.read_text(encoding="utf-8"))
    runner_source = next(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code" and "Computing Pass-A Uncertainty Map" in "".join(cell.get("source", []))
    )

    assert 'raise RuntimeError("Pass-A uncertainty map generation failed")' in runner_source
    assert "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json" in runner_source
    assert "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json" in runner_source


def test_submission_default_preserves_existing_easy_task_order(tmp_path: Path):
    output_path = tmp_path / "default_submission.ipynb"
    build(output_path=output_path, validation_mode="submission", pass_b_policy="broad")
    nb = json.loads(output_path.read_text(encoding="utf-8"))
    starter_cell = next(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
        and "".join(cell.get("source", [])).startswith("%%writefile starter.py")
    )
    starter_source = "\n".join(starter_cell.splitlines()[1:])
    merge_source = next(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
        and "decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)" in "".join(cell.get("source", []))
    )

    assert "hard_tasks = sorted(hard_tasks, key=lambda k: estimated_work(data[k]))" in starter_source
    assert "easy_tasks = sorted(easy_tasks, key=lambda k: estimated_work(data[k]))" not in starter_source
    compile(merge_source, "merge.py", "exec")


def test_validation_routes_support_both_kaggle_input_mounts(tmp_path: Path):
    for mode, dataset in (("train", "training"), ("eval", "evaluation")):
        output_path = tmp_path / f"validation_{mode}.ipynb"
        build(output_path=output_path, validation_mode=mode, pass_b_policy="broad")
        nb = json.loads(output_path.read_text(encoding="utf-8"))
        code_cells = [
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        starter_cell = next(source for source in code_cells if source.startswith("%%writefile starter.py"))
        solver_cell = next(source for source in code_cells if source.startswith("%%writefile arc_solver.py"))
        merge_source = next(
            source
            for source in code_cells
            if "decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)" in source
        )
        compile(merge_source, "merge.py", "exec")
        starter_source = "\n".join(starter_cell.splitlines()[1:])
        solver_source = "\n".join(solver_cell.splitlines()[1:])
        challenge_name = f"arc-agi_{dataset}_challenges.json"
        solution_name = f"arc-agi_{dataset}_solutions.json"
        for source in (starter_source, solver_source, merge_source):
            assert f"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/{challenge_name}" in source
            assert f"/kaggle/input/arc-prize-2026-arc-agi-2/{challenge_name}" in source
        assert f"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/{solution_name}" in merge_source
        assert f"/kaggle/input/arc-prize-2026-arc-agi-2/{solution_name}" in merge_source


def test_paired_policy_freezes_pass_a_and_runs_two_equal_budget_arms(tmp_path: Path):
    output_path = tmp_path / "paired_validation.ipynb"
    build(output_path=output_path, validation_mode="train", pass_b_policy="paired")
    nb = json.loads(output_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    runner_source = next(source for source in code_cells if "env_a = dict(" in source)
    starter_cell = next(source for source in code_cells if source.startswith("%%writefile starter.py"))
    starter_source = "\n".join(starter_cell.splitlines()[1:])
    merge_sources = [source for source in code_cells if "submission = data.get_submission" in source]

    compile(runner_source, "paired_runner.py", "exec")
    for source in code_cells:
        if source.startswith("%%writefile "):
            script_name = source.splitlines()[0].split(maxsplit=1)[1]
            compile("\n".join(source.splitlines()[1:]), script_name, "exec")

    assert "paired_inference_deadline" in runner_source
    assert "pass_a_end_time" in runner_source
    assert "pass_b_arm_budget" in runner_source
    assert "broad_end_time = min(paired_inference_deadline, time.time() + pass_b_arm_budget)" in runner_source
    assert "if uncertainty_time_remaining < 900:" in runner_source
    assert "uncertainty_end_time = min(paired_inference_deadline, time.time() + pass_b_arm_budget)" in runner_source
    assert runner_source.count('ARC_SEED="137"') == 2
    assert 'ARC_OUT_DIR="/kaggle/inference_outputs_broad"' in runner_source
    assert 'ARC_OUT_DIR="/kaggle/inference_outputs_uncertainty"' in runner_source
    assert 'ARC_PASS_B_POLICY="broad"' in runner_source
    assert 'ARC_PASS_B_POLICY="uncertainty"' in runner_source
    assert 'os.getenv("ARC_PASS_B_POLICY", "broad") == "uncertainty"' in starter_source
    assert runner_source.index("Computing Pass-A Uncertainty Map") < runner_source.index("Paired Pass B 1/2")
    assert runner_source.index("Paired Pass B 1/2") < runner_source.index("Paired Pass B 2/2")
    assert 'if not any(os.scandir("/kaggle/inference_outputs_broad")):' in runner_source
    assert 'if not any(os.scandir("/kaggle/inference_outputs_uncertainty")):' in runner_source
    assert len(merge_sources) == 2
    assert any("validation_predictions_broad.json" in source for source in merge_sources)
    assert any("validation_predictions_uncertainty.json" in source for source in merge_sources)
    assert all('decoder.load_decoded_results("/kaggle/inference_outputs")' in source for source in merge_sources)
    assert all("benchmark_selection_algos" not in source for source in merge_sources)
    for source in merge_sources:
        compile(source, "paired_merge.py", "exec")


def test_paired_policy_rejects_non_training_validation_mode(tmp_path: Path):
    with pytest.raises(ValueError, match="only supported for train CV"):
        build(
            output_path=tmp_path / "invalid_paired_eval.ipynb",
            validation_mode="eval",
            pass_b_policy="paired",
        )


def test_barrier_isolation_and_cleanup_in_builder(tmp_path: Path):
    output_path = tmp_path / "barrier_test.ipynb"
    build(output_path=output_path, validation_mode="train", pass_b_policy="paired")
    nb = json.loads(output_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    starter_cell = next(source for source in code_cells if source.startswith("%%writefile starter.py"))
    runner_source = next(source for source in code_cells if "env_a = dict(" in source)
    starter_source = "\n".join(starter_cell.splitlines()[1:])

    # Verify starter.py has cleanup, tag-based naming, and barrier timeout
    assert "cleanup_barriers" in starter_source
    assert "ARC_BARRIER_DIR" in starter_source
    assert "ARC_BARRIER_TAG" in starter_source
    assert "Timed out waiting for barrier" in starter_source

    # Verify runner passes distinct barrier tags
    assert 'ARC_BARRIER_TAG="pass_a"' in runner_source
    assert 'ARC_BARRIER_TAG="pass_b_broad"' in runner_source
    assert 'ARC_BARRIER_TAG="pass_b_uncertainty"' in runner_source
    assert "cleanup_kaggle_barriers" in runner_source


def test_barrier_cleanup_semantics_unit_test(tmp_path: Path):
    output_path = tmp_path / "barrier_semantics.ipynb"
    build(output_path=output_path, validation_mode="train", pass_b_policy="paired")
    nb = json.loads(output_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    starter_cell = next(source for source in code_cells if source.startswith("%%writefile starter.py"))
    runner_source = next(source for source in code_cells if "env_a = dict(" in source)
    starter_source = "\n".join(starter_cell.splitlines()[1:])

    # Execute helper functions in an isolated namespace
    import glob
    import os
    import time

    starter_ns = {"os": os, "time": time}
    fn_lines_starter = []
    in_fn = False
    for line in starter_source.splitlines():
        if line.startswith("def cleanup_barriers"):
            in_fn = True
        elif in_fn and line and not line.startswith(" ") and not line.startswith("\t"):
            break
        if in_fn:
            fn_lines_starter.append(line)
    exec(compile("\n".join(fn_lines_starter), "starter_fn.py", "exec"), starter_ns)
    cleanup_barriers = starter_ns["cleanup_barriers"]

    fn_lines_runner = []
    in_fn = False
    for line in runner_source.splitlines():
        if line.startswith("def cleanup_kaggle_barriers"):
            in_fn = True
        elif in_fn and line and not line.startswith(" ") and not line.startswith("\t"):
            break
        if in_fn:
            fn_lines_runner.append(line)
    runner_ns = {"os": os, "glob": glob}
    exec(compile("\n".join(fn_lines_runner), "runner_fn.py", "exec"), runner_ns)
    cleanup_kaggle_barriers = runner_ns["cleanup_kaggle_barriers"]

    barrier_dir = tmp_path / "barriers"
    barrier_dir.mkdir()

    # 1. Test runner-level barrier sweep removes all worker files
    (barrier_dir / "worker0").write_text("ok")
    (barrier_dir / "worker_pass_a_1").write_text("ok")
    (barrier_dir / "unrelated.txt").write_text("keep")

    orig_dir = os.environ.get("ARC_BARRIER_DIR")
    try:
        os.environ["ARC_BARRIER_DIR"] = str(barrier_dir)
        cleanup_kaggle_barriers()
        assert not (barrier_dir / "worker0").exists()
        assert not (barrier_dir / "worker_pass_a_1").exists()
        assert (barrier_dir / "unrelated.txt").exists()

        # 2. Test starter cleanup_barriers deletes specified tag and legacy files
        (barrier_dir / "worker_pass_a_0").write_text("ok")
        (barrier_dir / "worker_pass_b_0").write_text("ok")
        cleanup_barriers(str(barrier_dir), barrier_tag="pass_a")
        assert not (barrier_dir / "worker_pass_a_0").exists()
        assert (barrier_dir / "worker_pass_b_0").exists()
    finally:
        if orig_dir is not None:
            os.environ["ARC_BARRIER_DIR"] = orig_dir
        else:
            os.environ.pop("ARC_BARRIER_DIR", None)


def test_submission_builder_injects_diversity_selection(tmp_path: Path):
    output_path = tmp_path / "submission_diversity.ipynb"
    build(output_path=output_path, validation_mode="submission")
    nb = json.loads(output_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    merge_source = next(source for source in code_cells if "decoder.load_decoded_results" in source)
    assert "select_diverse_pair" in merge_source



