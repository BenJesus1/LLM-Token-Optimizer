"""Measure DP vs token-level greedy vs random on the seeded synthetic sets.

Run from the repo root (after ``pip install -e .``)::

    python -m benchmarks.measure

pytest must not import this module's ``__main__`` path and must not pass
n=1000 into ``measure_suite``. The 1,000-task DP is only for this script.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from allocator import Allocation, Task, allocate_dp, allocate_greedy, allocate_random

from .generate import DEFAULT_SEED, SCALES, SyntheticSet, generate_scale_suite

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

SolverFn = Callable[[Sequence[Task], int], Allocation]


def _solvers(*, random_seed: int) -> dict[str, SolverFn]:
    return {
        "dp": allocate_dp,
        "greedy": allocate_greedy,
        "random": lambda tasks, budget: allocate_random(tasks, budget, seed=random_seed),
    }


def measure_instance(
    instance: SyntheticSet,
    *,
    random_seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Time each token-level solver on one synthetic set."""

    rows: list[dict[str, Any]] = []
    n = len(instance.tasks)
    for name, solve in _solvers(random_seed=random_seed).items():
        started = time.perf_counter()
        allocation = solve(instance.tasks, instance.budget)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "n": n,
                "budget": instance.budget,
                "seed": instance.seed,
                "solver": name,
                "runtime_s": elapsed,
                "total_value": allocation.total_value,
                "total_cost": allocation.total_cost,
            }
        )
    return rows


def measure_suite(
    *,
    seed: int = DEFAULT_SEED,
    scales: Sequence[int] = SCALES,
) -> list[dict[str, Any]]:
    """Measure the requested scales. Callers that run under pytest must pass a
    small ``scales`` tuple so the 1,000-task DP never hits CI.
    """

    suite = generate_scale_suite(seed=seed)
    rows: list[dict[str, Any]] = []
    for n in scales:
        rows.extend(measure_instance(suite[n], random_seed=seed))
    return rows


def save_results(rows: Sequence[dict[str, Any]], path: Path = RESULTS_PATH) -> Path:
    payload = {
        "seed": DEFAULT_SEED,
        "solvers": ["dp", "greedy", "random"],
        "rows": list(rows),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_results(path: Path = RESULTS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = measure_suite(seed=DEFAULT_SEED, scales=SCALES)
    saved = save_results(rows)
    print(f"wrote {saved}")
    for row in rows:
        print(
            f"n={row['n']:<5} {row['solver']:<7} "
            f"value={row['total_value']:.4f}  runtime={row['runtime_s']:.4f}s"
        )


if __name__ == "__main__":
    main()
