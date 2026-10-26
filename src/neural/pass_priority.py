import math
from collections.abc import Mapping, Sequence


def uncertainty_from_candidate_evidence(
    probability_ranked: Sequence[str],
    vote_ranked: Sequence[str],
    vote_counts: Mapping[str, int]
) -> float:
    """
    Calculate uncertainty based on candidate rankings and vote counts.
    """
    if not probability_ranked or not vote_ranked:
        return 1.0

    # Validate vote_counts and find positive counts
    positive_counts = []
    for count in vote_counts.values():
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("Vote counts must be integers.")
        if count < 0:
            raise ValueError("Vote counts cannot be negative.")
        if count > 0:
            positive_counts.append(count)

    if not positive_counts:
        return 1.0

    # disagreement
    disagreement = 1.0 if probability_ranked[0] != vote_ranked[0] else 0.0

    # normalized_margin
    positive_counts.sort(reverse=True)
    if len(positive_counts) == 1:
        margin = 1.0
    else:
        max1 = positive_counts[0]
        max2 = positive_counts[1]
        total_positive = sum(positive_counts)
        margin = (max1 - max2) / total_positive

    return 0.5 * disagreement + 0.5 * (1 - margin)


def rank_tasks_by_uncertainty(
    uncertainty_by_task: Mapping[str, float],
    estimated_work_by_task: Mapping[str, float]
) -> list[str]:
    """
    Rank tasks based on uncertainty (descending), estimated work (ascending), and lexicographic ID.

    Args:
        uncertainty_by_task: Mapping of task ID to uncertainty value (0.0 to 1.0 inclusive).
        estimated_work_by_task: Mapping of task ID to estimated work (>= 0).

    Returns:
        List of task IDs ordered by priority.

    Raises:
        ValueError: If key sets do not match, or if values are out of bounds or non-finite.
        TypeError: If values are not numeric.
    """
    if set(uncertainty_by_task.keys()) != set(estimated_work_by_task.keys()):
        raise ValueError("Key sets for uncertainty and estimated work must be identical.")

    tasks_to_rank = []

    for task_id in uncertainty_by_task:
        unc = uncertainty_by_task[task_id]
        work = estimated_work_by_task[task_id]

        # Validate uncertainty
        if type(unc) is bool or not isinstance(unc, (int, float)):
            raise TypeError(f"Uncertainty for task {task_id} must be a float or int.")
        if not math.isfinite(unc):
            raise ValueError(f"Uncertainty for task {task_id} must be finite.")
        if not (0.0 <= unc <= 1.0):
            raise ValueError(f"Uncertainty for task {task_id} must be between 0.0 and 1.0 (inclusive).")

        # Validate estimated work
        if type(work) is bool or not isinstance(work, (int, float)):
            raise TypeError(f"Estimated work for task {task_id} must be a float or int.")
        if not math.isfinite(work):
            raise ValueError(f"Estimated work for task {task_id} must be finite.")
        if work < 0:
            raise ValueError(f"Estimated work for task {task_id} cannot be negative.")

        tasks_to_rank.append((task_id, float(unc), float(work)))

    # Sort: descending uncertainty (-unc), ascending work (work), lexicographic ID (task_id)
    tasks_to_rank.sort(key=lambda x: (-x[1], x[2], x[0]))

    return [t[0] for t in tasks_to_rank]
