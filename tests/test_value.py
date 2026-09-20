"""Tests for simulated diminishing-returns value curves.

Values are functions of allocated tokens and a per-task weight. They are
never fetched from a live LLM API.
"""

import math

import pytest

from allocator import Task, evaluate_curve


def test_log_curve_matches_closed_form() -> None:
    task = Task(id="a", token_cost=10, weight=2.0, curve="log")

    assert task.value_at(0) == 0.0
    assert task.value_at(1) == pytest.approx(2.0 * math.log(2))
    assert task.value_at(10) == pytest.approx(2.0 * math.log(11))


def test_log_curve_has_diminishing_marginal_value() -> None:
    # Marginal gain log(1+t+1) - log(1+t) = log((t+2)/(t+1)) falls as t grows.
    task = Task(id="a", token_cost=100, weight=1.0, curve="log")

    early = task.value_at(2) - task.value_at(1)
    mid = task.value_at(11) - task.value_at(10)
    late = task.value_at(101) - task.value_at(100)

    assert early > mid > late > 0


def test_sqrt_curve_is_available_per_task() -> None:
    task = Task(id="a", token_cost=4, weight=3.0, curve="sqrt")

    assert task.value_at(0) == 0.0
    assert task.value_at(4) == pytest.approx(6.0)


def test_tasks_can_use_different_curves() -> None:
    log_task = Task(id="log", token_cost=4, weight=1.0, curve="log")
    sqrt_task = Task(id="sqrt", token_cost=4, weight=1.0, curve="sqrt")

    assert log_task.value_at(4) != sqrt_task.value_at(4)


def test_zero_one_include_uses_value_at_token_cost() -> None:
    task = Task(id="a", token_cost=3, weight=2.0, curve="log")

    assert task.value == task.value_at(3)


def test_evaluate_curve_rejects_negative_tokens() -> None:
    with pytest.raises(ValueError, match="tokens"):
        evaluate_curve("log", -1, 1.0)


def test_evaluate_curve_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown value curve"):
        evaluate_curve("thompson", 4, 1.0)
