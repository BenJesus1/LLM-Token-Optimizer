"""Exact 0/1 knapsack solver over a discrete task list."""

from __future__ import annotations

from collections.abc import Sequence

from .allocation import Allocation
from .task import Task


def knapsack_dp(tasks: Sequence[Task], budget: int) -> Allocation:
    """Return a maximum-value subset of ``tasks`` whose costs sum to at most ``budget``.

    Each task is included at most once (0/1 knapsack). Time and memory are
    O(n * budget).
    """

    if budget < 0:
        raise ValueError("budget must be a non-negative integer")

    n = len(tasks)
    # dp[i][w] = max value from a subset of tasks[:i] with total cost <= w.
    # tasks[:i] means the first i tasks (tasks[0] .. tasks[i-1]).
    dp: list[list[int]] = [[0] * (budget + 1) for _ in range(n + 1)]

    for i, task in enumerate(tasks, start=1):
        cost = task.token_cost
        value = task.value
        for w in range(budget + 1):
            # Recurrence: every subset of tasks[:i] either
            #   - excludes tasks[i-1], so the best value is dp[i-1][w], or
            #   - includes it, which is feasible only when cost <= w, and then
            #     the rest of the subset is an optimal solution of tasks[:i-1]
            #     under remaining budget w - cost.
            # Those two cases partition all subsets, so the max is optimal.
            skip = dp[i - 1][w]
            if cost <= w:
                take = dp[i - 1][w - cost] + value
                dp[i][w] = max(skip, take)
            else:
                dp[i][w] = skip

    # Walk backward from dp[n][budget]. A 2D table is kept (instead of the
    # usual 1D rolling array) so this reconstruction is possible.
    # On a tie between skip and take, skip: both are optimal, and skipping
    # avoids zero-value tasks that do not improve the objective.
    selected: list[Task] = []
    remaining = budget
    for i in range(n, 0, -1):
        task = tasks[i - 1]
        skip = dp[i - 1][remaining]
        can_take = task.token_cost <= remaining
        take = (
            dp[i - 1][remaining - task.token_cost] + task.value
            if can_take
            else skip
        )
        if can_take and dp[i][remaining] == take and take > skip:
            selected.append(task)
            remaining -= task.token_cost

    selected.reverse()
    return Allocation(selected=tuple(selected))
