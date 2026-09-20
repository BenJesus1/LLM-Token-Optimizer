"""Simulated per-task value curves. No live LLM API calls."""

from __future__ import annotations

import math
from collections.abc import Callable

ValueFn = Callable[[int, float], float]


def log_curve(tokens: int, weight: float) -> float:
    """Diminishing returns: log(1 + tokens) * weight."""

    return math.log(1 + tokens) * weight


def sqrt_curve(tokens: int, weight: float) -> float:
    """Diminishing returns: sqrt(tokens) * weight."""

    return math.sqrt(tokens) * weight


def constant_curve(tokens: int, weight: float) -> float:
    """Fixed value, independent of tokens. Used by the 0/1 knapsack phase."""

    return float(weight)


CURVES: dict[str, ValueFn] = {
    "log": log_curve,
    "sqrt": sqrt_curve,
    "constant": constant_curve,
}


def evaluate_curve(curve: str, tokens: int, weight: float) -> float:
    """Return simulated value for an allocation of ``tokens`` to one task."""

    if tokens < 0:
        raise ValueError("tokens must be a non-negative integer")
    try:
        fn = CURVES[curve]
    except KeyError:
        known = ", ".join(sorted(CURVES))
        raise ValueError(f"unknown value curve {curve!r}; expected one of: {known}") from None
    return fn(tokens, weight)
