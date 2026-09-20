"""Seeded synthetic task sets for the benchmark harness.

Charts in later steps must be reproducible from a cold clone: always pass
the same ``seed``. Values are simulated; this module never calls an LLM API.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from allocator import CURVES, Task

SCALES: tuple[int, ...] = (10, 100, 1_000)
DEFAULT_SEED = 2026
CURVE_CHOICES: tuple[str, ...] = tuple(sorted(CURVES))


@dataclass(frozen=True)
class SyntheticSet:
    """One generated instance: tasks plus a budget derived from their caps."""

    tasks: tuple[Task, ...]
    budget: int
    seed: int

    def __post_init__(self) -> None:
        if self.budget < 0:
            raise ValueError("budget must be a non-negative integer")


def generate_tasks(
    n: int,
    *,
    seed: int = DEFAULT_SEED,
    min_cost: int = 1,
    max_cost: int = 20,
    min_weight: float = 0.5,
    max_weight: float = 5.0,
) -> tuple[Task, ...]:
    """Draw ``n`` tasks with random caps, weights, and named value curves."""

    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if min_cost < 1 or max_cost < min_cost:
        raise ValueError("cost range must satisfy 1 <= min_cost <= max_cost")
    if min_weight < 0 or max_weight < min_weight:
        raise ValueError("weight range must satisfy 0 <= min_weight <= max_weight")

    rng = random.Random(seed)
    tasks: list[Task] = []
    for i in range(n):
        tasks.append(
            Task(
                id=f"t{i:04d}",
                token_cost=rng.randint(min_cost, max_cost),
                weight=rng.uniform(min_weight, max_weight),
                curve=rng.choice(CURVE_CHOICES),
            )
        )
    return tuple(tasks)


def generate_set(n: int, *, seed: int = DEFAULT_SEED) -> SyntheticSet:
    """Task list of size ``n`` plus a budget of one quarter of total cap.

    The quarter-cap budget keeps the instance interesting: not every task
    can be fully served, and dumping the whole budget on one task is capped.
    """

    tasks = generate_tasks(n, seed=seed)
    total_cap = sum(task.token_cost for task in tasks)
    budget = max(1, total_cap // 4) if tasks else 0
    return SyntheticSet(tasks=tasks, budget=budget, seed=seed)


def generate_scale_suite(*, seed: int = DEFAULT_SEED) -> dict[int, SyntheticSet]:
    """Build the 10 / 100 / 1,000-task instances used by later benchmark steps.

    Each scale uses ``seed + n`` so the three sets are independent but still
    determined by a single published seed.
    """

    return {n: generate_set(n, seed=seed + n) for n in SCALES}
