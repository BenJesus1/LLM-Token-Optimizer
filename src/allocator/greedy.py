"""Greedy value-per-token baseline. Comparison point, not the exact solver."""

from __future__ import annotations

from collections.abc import Sequence

from .allocation import Allocation, allocation_from_counts, allocation_from_tasks
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
    return allocation_from_tasks(tuple(tasks[i] for i in chosen_indexes))


def _greedy_step(task: Task, current: int, remaining: int) -> tuple[float, int]:
    """Density and token delta of the best greedy increment for ``task``.

    Diminishing curves take one token. Constant (0/1 step) curves jump to the
    cap in one shot, because unit steps in between have zero gain.
    """

    if remaining <= 0 or current >= task.token_cost:
        return (-1.0, 0)
    if task.curve == "constant":
        need = task.token_cost - current
        if need <= remaining and need > 0:
            gain = task.value_at(current + need) - task.value_at(current)
            if gain > 0:
                return (gain / need, need)
        return (-1.0, 0)
    gain = task.value_at(current + 1) - task.value_at(current)
    if gain > 0:
        return (gain, 1)
    return (-1.0, 0)


def allocate_greedy(tasks: Sequence[Task], budget: int) -> Allocation:
    """Spend tokens on the highest-density increment until the budget is gone.

    This is the token-level baseline for ``allocate_dp``. It is not the 0/1
    ``knapsack_greedy`` solver. Time is O(n * budget).
    """

    if budget < 0:
        raise ValueError("budget must be a non-negative integer")

    counts = [0] * len(tasks)
    remaining = budget
    while remaining > 0:
        best_i = -1
        best_density = 0.0
        best_dt = 0
        for i, task in enumerate(tasks):
            density, dt = _greedy_step(task, counts[i], remaining)
            if dt > 0 and density > best_density:
                best_density = density
                best_dt = dt
                best_i = i
        if best_i < 0:
            break
        counts[best_i] += best_dt
        remaining -= best_dt

    return allocation_from_counts(tasks, counts)
