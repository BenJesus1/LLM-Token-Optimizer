"""Value curves: simulated log/sqrt/constant plus cached live tables."""

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
    """Placeholder for the named ``constant`` curve.

    Task.value_at applies the 0/1 step using token_cost; this function is only
    used if evaluate_curve is called directly.
    """

    return float(weight) if tokens > 0 else 0.0


def table_curve(tokens: int, weight: float) -> float:
    """Placeholder. ``Task.value_at`` reads ``value_table`` for ``curve='table'``."""

    del tokens, weight
    return 0.0


CURVES: dict[str, ValueFn] = {
    "log": log_curve,
    "sqrt": sqrt_curve,
    "constant": constant_curve,
    "table": table_curve,
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
