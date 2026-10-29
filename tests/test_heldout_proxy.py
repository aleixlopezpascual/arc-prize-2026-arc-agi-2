from __future__ import annotations

from src.tools.heldout_proxy import (
    family_loo_precision,
    policy_attempts,
    score_policies,
)

A = [[1]]
B = [[2]]
C = [[3]]
D = [[4]]


def _sym(pred, family="fam"):
    return {"name": family, "family": family, "preds": [pred], "error": None}


def test_union_keeps_n1_and_replaces_n2():
    assert policy_attempts("union", _sym(A), 0, B, C) == (A, B)


def test_union_keeps_n2_when_symbolic_agrees_with_n1():
    assert policy_attempts("union", _sym(B), 0, B, C) == (B, C)


def test_no_symbolic_falls_back_to_neural():
    assert policy_attempts("union", None, 0, B, C) == (B, C)
    assert policy_attempts("union", {"preds": None}, 0, B, C) == (B, C)


def test_no_ca_policy_skips_cellular_automata():
    ca = _sym(A, family="cellular_automatax3")
    assert policy_attempts("union_no_ca", ca, 0, B, C) == (B, C)
    assert policy_attempts("union_no_ca", _sym(A), 0, B, C) == (A, B)


def test_loo_gate_uses_threshold():
    assert policy_attempts("union_loo", _sym(A), 0, B, C, loo_precision=0.9) == (A, B)
    assert policy_attempts("union_loo", _sym(A), 0, B, C, loo_precision=0.1) == (B, C)
    assert policy_attempts("union_loo", _sym(A), 0, B, C, loo_precision=None) == (B, C)


def test_loo_precision_excludes_own_task():
    symbolic = {"t1": _sym(A), "t2": _sym(B), "t3": _sym(A)}
    solutions = {"t1": [A], "t2": [C], "t3": [A]}
    loo = family_loo_precision(symbolic, solutions)
    # t1: others are t2 (wrong) and t3 (right) -> (1 + .5) / (2 + 1)
    assert loo[("t1", "fam")] == 1.5 / 3
    # t2: others are both right -> (2 + .5) / (2 + 1)
    assert loo[("t2", "fam")] == 2.5 / 3


def test_score_policies_counts_gains_and_losses():
    # t1: symbolic right, neural wrong -> union gains.
    # t2: only N2 right and symbolic wrong -> union loses.
    symbolic = {"t1": _sym(A), "t2": _sym(D)}
    neural = {
        "t1": [{"attempt_1": B, "attempt_2": C}],
        "t2": [{"attempt_1": B, "attempt_2": C}],
    }
    solutions = {"t1": [A], "t2": [C]}
    res = score_policies(symbolic, neural, solutions, policies=("neural", "union"))
    assert res["neural"]["correct"] == 1
    assert res["union"]["correct"] == 1
    assert (res["union"]["gains"], res["union"]["losses"]) == (1, 1)


def test_check_neural_reference_flags_hybrid_output():
    from src.tools.heldout_proxy import check_neural_reference

    symbolic = {"t1": _sym(A), "t2": _sym(B)}
    hybrid = {"t1": [{"attempt_1": A, "attempt_2": C}], "t2": [{"attempt_1": B, "attempt_2": C}]}
    neural_only = {"t1": [{"attempt_1": D, "attempt_2": C}], "t2": [{"attempt_1": B, "attempt_2": C}]}
    assert check_neural_reference(symbolic, hybrid) == (2, 2)
    assert check_neural_reference(symbolic, neural_only) == (1, 2)


def test_restrict_to_families_disables_other_families():
    from src.tools.heldout_proxy import restrict_to_families

    symbolic = {"t1": _sym(A, "generic"), "t2": _sym(B, "one_off")}
    out = restrict_to_families(symbolic, {"generic"})
    assert out["t1"]["preds"] == [A]
    assert out["t2"]["preds"] is None


def test_score_policies_handles_subset_neural_file():
    # Neural file covers only t1/t2; t3 (symbolic fires, in solutions) must not count anywhere.
    symbolic = {"t1": _sym(A), "t2": _sym(D), "t3": _sym(A)}
    neural = {
        "t1": [{"attempt_1": B, "attempt_2": C}],
        "t2": [{"attempt_1": B, "attempt_2": C}],
    }
    solutions = {"t1": [A], "t2": [C], "t3": [A]}
    res = score_policies(symbolic, neural, solutions, policies=("neural", "union", "union_loo"))
    assert res["neural"]["total"] == 2
    assert res["union"]["total"] == 2
    assert (res["union"]["gains"], res["union"]["losses"]) == (1, 1)
    # LOO precision ignores t3: for t1 the only other covered task is t2 (wrong) -> (0 + .5) / (1 + 1) = .25 < .5
    assert res["union_loo"]["gains"] == 0
