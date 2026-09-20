"""Seeded random token allocation. Comparison baseline, not a solver to trust."""

from __future__ import annotations

import random
from collections.abc import Sequence

from .allocation import Allocation, allocation_from_counts
from .task import Task


def allocate_random(
    tasks: Sequence[Task],
    budget: int,
    *,
    seed: int,
) -> Allocation:
    """Give one token at a time to a uniformly chosen task that still has cap left.

    ``seed`` is required so benchmark rows are reproducible from a cold clone.
    """

    if budget < 0:
        raise ValueError("budget must be a non-negative integer")

    rng = random.Random(seed)
    counts = [0] * len(tasks)
    remaining = budget
    while remaining > 0:
        open_indexes = [
            i
            for i, task in enumerate(tasks)
            if counts[i] < task.token_cost
        ]
        if not open_indexes:
            break
        i = rng.choice(open_indexes)
        counts[i] += 1
        remaining -= 1

    return allocation_from_counts(tasks, counts)
