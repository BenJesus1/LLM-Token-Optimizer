"""Exact resource-allocation DP: integer tokens per task, not include/exclude.

This is *not* GPU or KV-cache placement. Each task gets a token count in
0..token_cost; value comes from the task's simulated curve.
"""

from __future__ import annotations

from collections.abc import Sequence

from .allocation import Allocation, Assignment
from .task import Task


def allocate_dp(tasks: Sequence[Task], budget: int) -> Allocation:
    """Choose an integer token count for each task to maximize total value.

    ``task.token_cost`` is a cap, not an all-or-nothing size. Recurrence:

        dp[i][w] = max_{t=0..min(cap_i, w)}  dp[i-1][w-t] + value_i(t)

    Every allocation of the first i tasks with cost <= w is some t tokens to
    task i-1 plus an optimal allocation of the earlier tasks into w-t, so the
    max is exact. Ties keep the smaller t so we do not spend tokens that do
    not raise value.

    Time is O(n * budget * cap). This is multiple-choice knapsack / separable
    resource allocation, not a memory-placement solver.
    """

    if budget < 0:
        raise ValueError("budget must be a non-negative integer")

    n = len(tasks)
    dp: list[list[float]] = [[0.0] * (budget + 1) for _ in range(n + 1)]
    choice: list[list[int]] = [[0] * (budget + 1) for _ in range(n + 1)]

    for i, task in enumerate(tasks, start=1):
        cap = task.token_cost
        for w in range(budget + 1):
            best_value = dp[i - 1][w] + task.value_at(0)
            best_t = 0
            for t in range(1, min(cap, w) + 1):
                candidate = dp[i - 1][w - t] + task.value_at(t)
                if candidate > best_value:
                    best_value = candidate
                    best_t = t
            dp[i][w] = best_value
            choice[i][w] = best_t

    picked: list[Assignment] = []
    remaining = budget
    for i in range(n, 0, -1):
        tokens = choice[i][remaining]
        if tokens > 0:
            picked.append(Assignment(task=tasks[i - 1], tokens=tokens))
        remaining -= tokens

    picked.reverse()
    return Allocation(assignments=tuple(picked))
