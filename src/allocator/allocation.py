"""Chosen token assignments plus derived totals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .task import Task


@dataclass(frozen=True)
class Assignment:
    """How many tokens one task received."""

    task: Task
    tokens: int

    def __post_init__(self) -> None:
        if self.tokens < 0:
            raise ValueError("tokens must be a non-negative integer")

    @property
    def value(self) -> float:
        return self.task.value_at(self.tokens)


@dataclass(frozen=True)
class Allocation:
    """Solver output. ``assignments`` are in input order and omit zero-token tasks."""

    assignments: tuple[Assignment, ...]

    @property
    def selected(self) -> tuple[Task, ...]:
        return tuple(item.task for item in self.assignments)

    @property
    def total_cost(self) -> int:
        return sum(item.tokens for item in self.assignments)

    @property
    def total_value(self) -> float:
        return sum(item.value for item in self.assignments)

    def tokens_for(self, task_id: str) -> int:
        for item in self.assignments:
            if item.task.id == task_id:
                return item.tokens
        return 0


def allocation_from_tasks(tasks: tuple[Task, ...]) -> Allocation:
    """0/1 result: each chosen task is fully allocated its ``token_cost``."""

    return Allocation(
        assignments=tuple(Assignment(task=task, tokens=task.token_cost) for task in tasks)
    )


def allocation_from_counts(tasks: Sequence[Task], counts: Sequence[int]) -> Allocation:
    """Variable-token result from a per-task token vector (input order)."""

    return Allocation(
        assignments=tuple(
            Assignment(task=task, tokens=n)
            for task, n in zip(tasks, counts, strict=True)
            if n > 0
        )
    )
