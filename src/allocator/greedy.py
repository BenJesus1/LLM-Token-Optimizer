"""Greedy value-per-token baseline. Comparison point, not the exact solver."""

from __future__ import annotations

from collections.abc import Sequence

from .allocation import Allocation
from .task import Task


def _value_per_token(task: Task) -> float:
    """Density used to rank tasks. Zero-cost, positive-value tasks rank first."""

    if task.token_cost == 0:
        return float("inf") if task.value > 0 else float("-inf")
    return task.value / task.token_cost


def knapsack_greedy(tasks: Sequence[Task], budget: int) -> Allocation:
    """Fill the budget by taking whole tasks in value-per-token order.

    This is the density heuristic from fractional knapsack, applied without
    splitting tasks: take the next highest-density task if it fits, otherwise
    skip it and keep going.

    It is *not* optimal for 0/1 knapsack. A high-density task can leave a
    remainder that two slightly lower-density tasks would fill for more total
    value. Fractional knapsack *would* be optimal if a task could be partially
    served; v1 cannot split a task, so leftover budget is wasted.

    Time is O(n log n) from the sort.
    """

    if budget < 0:
        raise ValueError("budget must be a non-negative integer")

    ranked = sorted(enumerate(tasks), key=lambda item: _value_per_token(item[1]), reverse=True)
    chosen_indexes: list[int] = []
    remaining = budget
    for index, task in ranked:
        if task.value == 0:
            continue
        if task.token_cost <= remaining:
            chosen_indexes.append(index)
            remaining -= task.token_cost

    chosen_indexes.sort()
    return Allocation(selected=tuple(tasks[i] for i in chosen_indexes))
