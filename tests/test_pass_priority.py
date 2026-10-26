from collections.abc import Mapping, Sequence

import pytest

from src.neural.pass_priority import (
    rank_tasks_by_uncertainty,
    uncertainty_from_candidate_evidence,
)


def test_rank_tasks_by_uncertainty_highest_first() -> None:
    uncertainty: Mapping[str, float] = {"task1": 0.2, "task2": 0.9, "task3": 0.5}
    work: Mapping[str, float] = {"task1": 10.0, "task2": 10.0, "task3": 10.0}
    assert rank_tasks_by_uncertainty(uncertainty, work) == ["task2", "task3", "task1"]

def test_rank_tasks_by_uncertainty_breaks_ties_by_cheaper_work() -> None:
    uncertainty: Mapping[str, float] = {"task1": 0.5, "task2": 0.5, "task3": 0.5}
    work: Mapping[str, float] = {"task1": 20.0, "task2": 5.0, "task3": 10.0}
    assert rank_tasks_by_uncertainty(uncertainty, work) == ["task2", "task3", "task1"]

def test_rank_tasks_by_uncertainty_breaks_ties_by_lexicographic_id() -> None:
    uncertainty: Mapping[str, float] = {"b_task": 0.5, "a_task": 0.5, "c_task": 0.5}
    work: Mapping[str, float] = {"b_task": 10.0, "a_task": 10.0, "c_task": 10.0}
    assert rank_tasks_by_uncertainty(uncertainty, work) == ["a_task", "b_task", "c_task"]

def test_rank_tasks_by_uncertainty_returns_every_task_exactly_once() -> None:
    uncertainty: Mapping[str, float] = {"task1": 0.8, "task2": 0.1, "task3": 0.5}
    work: Mapping[str, float] = {"task1": 10.0, "task2": 5.0, "task3": 10.0}
    result = rank_tasks_by_uncertainty(uncertainty, work)
    assert len(result) == 3
    assert set(result) == {"task1", "task2", "task3"}

def test_rank_tasks_by_uncertainty_mismatched_keys() -> None:
    uncertainty: Mapping[str, float] = {"task1": 0.5, "task2": 0.5}
    work: Mapping[str, float] = {"task1": 10.0}
    with pytest.raises(ValueError):
        rank_tasks_by_uncertainty(uncertainty, work)
        
    uncertainty2: Mapping[str, float] = {"task1": 0.5}
    work2: Mapping[str, float] = {"task1": 10.0, "task2": 10.0}
    with pytest.raises(ValueError):
        rank_tasks_by_uncertainty(uncertainty2, work2)

def test_rank_tasks_by_uncertainty_non_finite_uncertainty() -> None:
    work: Mapping[str, float] = {"task1": 10.0}
    with pytest.raises(ValueError):
        rank_tasks_by_uncertainty({"task1": float('inf')}, work)
    with pytest.raises(ValueError):
        rank_tasks_by_uncertainty({"task1": float('-inf')}, work)
    with pytest.raises(ValueError):
        rank_tasks_by_uncertainty({"task1": float('nan')}, work)

def test_rank_tasks_by_uncertainty_out_of_range_uncertainty() -> None:
    work: Mapping[str, float] = {"task1": 10.0}
    with pytest.raises(ValueError):
        rank_tasks_by_uncertainty({"task1": -0.1}, work)
    with pytest.raises(ValueError):
        rank_tasks_by_uncertainty({"task1": 1.1}, work)

def test_rank_tasks_by_uncertainty_negative_non_finite_work() -> None:
    uncertainty: Mapping[str, float] = {"task1": 0.5}
    with pytest.raises(ValueError):
        rank_tasks_by_uncertainty(uncertainty, {"task1": -1.0})
    with pytest.raises(ValueError):
        rank_tasks_by_uncertainty(uncertainty, {"task1": float('inf')})
    with pytest.raises(ValueError):
        rank_tasks_by_uncertainty(uncertainty, {"task1": float('-inf')})
    with pytest.raises(ValueError):
        rank_tasks_by_uncertainty(uncertainty, {"task1": float('nan')})

def test_uncertainty_from_candidate_evidence_same_winner_strong_margin() -> None:
    probability_ranked: Sequence[str] = ["A", "B"]
    vote_ranked: Sequence[str] = ["A", "B"]
    vote_counts: Mapping[str, int] = {"A": 4, "B": 1}
    assert uncertainty_from_candidate_evidence(probability_ranked, vote_ranked, vote_counts) == pytest.approx(0.2)

def test_uncertainty_from_candidate_evidence_single_candidate() -> None:
    probability_ranked: Sequence[str] = ["A"]
    vote_ranked: Sequence[str] = ["A"]
    vote_counts: Mapping[str, int] = {"A": 5}
    assert uncertainty_from_candidate_evidence(probability_ranked, vote_ranked, vote_counts) == pytest.approx(0.0)

def test_uncertainty_from_candidate_evidence_ranker_disagreement_strong_margin() -> None:
    probability_ranked: Sequence[str] = ["B", "A"]
    vote_ranked: Sequence[str] = ["A", "B"]
    vote_counts: Mapping[str, int] = {"A": 4, "B": 1}
    assert uncertainty_from_candidate_evidence(probability_ranked, vote_ranked, vote_counts) == pytest.approx(0.7)

def test_uncertainty_from_candidate_evidence_tied_votes() -> None:
    probability_ranked: Sequence[str] = ["A", "B"]
    vote_ranked: Sequence[str] = ["A", "B"]
    vote_counts: Mapping[str, int] = {"A": 2, "B": 2}
    assert uncertainty_from_candidate_evidence(probability_ranked, vote_ranked, vote_counts) == pytest.approx(0.5)

def test_uncertainty_from_candidate_evidence_empty_evidence() -> None:
    assert uncertainty_from_candidate_evidence([], ["A"], {"A": 1}) == 1.0
    assert uncertainty_from_candidate_evidence(["A"], [], {"A": 1}) == 1.0
    assert uncertainty_from_candidate_evidence(["A"], ["A"], {}) == 1.0

def test_uncertainty_from_candidate_evidence_invalid_counts() -> None:
    with pytest.raises(ValueError):
        uncertainty_from_candidate_evidence(["A"], ["A"], {"A": -1})
    with pytest.raises(TypeError):
        uncertainty_from_candidate_evidence(["A"], ["A"], {"A": 1.5})
    with pytest.raises(TypeError):
        uncertainty_from_candidate_evidence(["A"], ["A"], {"A": True})
