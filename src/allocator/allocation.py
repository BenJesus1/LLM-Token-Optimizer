"""Chosen subset of tasks plus derived totals."""

from __future__ import annotations

from dataclasses import dataclass

from .task import Task


@dataclass(frozen=True)
class Allocation:
    """Tasks selected by a solver, in the same order they appeared in the input."""

    selected: tuple[Task, ...]

    @property
    def total_cost(self) -> int:
        return sum(task.token_cost for task in self.selected)

    @property
    def total_value(self) -> int:
        return sum(task.value for task in self.selected)
