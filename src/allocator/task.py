"""Task record competing for a shared token budget."""

from __future__ import annotations

from dataclasses import dataclass

from .value import CURVES, evaluate_curve


@dataclass(frozen=True)
class Task:
    """One job: identifier, a token size, and a simulated value curve.

    ``weight`` scales a named curve (default ``log``):
    value(tokens) = log(1 + tokens) * weight. Values are mocked — never
    fetched from a live LLM API. The ``constant`` curve ignores tokens and
    recovers the fixed-value 0/1 knapsack used in phase 2.
    """

    id: str
    token_cost: int
    weight: float
    curve: str = "log"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Task id must be a non-empty string")
        if self.token_cost < 0:
            raise ValueError("token_cost must be a non-negative integer")
        if self.weight < 0:
            raise ValueError("weight must be a non-negative number")
        if self.curve not in CURVES:
            known = ", ".join(sorted(CURVES))
            raise ValueError(f"unknown value curve {self.curve!r}; expected one of: {known}")

    @classmethod
    def constant(cls, id: str, token_cost: int, value: float) -> Task:
        """Build a fixed-value task for 0/1 knapsack examples."""

        return cls(id=id, token_cost=token_cost, weight=value, curve="constant")

    def value_at(self, tokens: int) -> float:
        """Simulated value of allocating ``tokens`` to this task."""

        return evaluate_curve(self.curve, tokens, self.weight)

    @property
    def value(self) -> float:
        """Value if the task is fully included at ``token_cost`` (0/1 include)."""

        return self.value_at(self.token_cost)
