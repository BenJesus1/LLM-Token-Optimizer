"""Task record competing for a shared token budget."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    """One discrete job: identifier, integer token cost, and a fixed value.

    Value is a constant in this phase so the solver is textbook 0/1 knapsack.
    Phase 3 replaces it with a diminishing-returns function of tokens allocated.
    """

    id: str
    token_cost: int
    value: int

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Task id must be a non-empty string")
        if self.token_cost < 0:
            raise ValueError("token_cost must be a non-negative integer")
        if self.value < 0:
            raise ValueError("value must be a non-negative integer")
