"""Tests for the Task data model."""

from dataclasses import FrozenInstanceError

import pytest

from allocator import Task


def test_task_stores_id_cost_weight_and_curve() -> None:
    task = Task(id="summarize", token_cost=12, weight=40, curve="log")

    assert task.id == "summarize"
    assert task.token_cost == 12
    assert task.weight == 40
    assert task.curve == "log"


def test_default_curve_is_log() -> None:
    task = Task(id="a", token_cost=4, weight=2)

    assert task.curve == "log"


def test_constant_constructor_uses_fixed_value() -> None:
    task = Task.constant(id="a", token_cost=5, value=9)

    assert task.curve == "constant"
    assert task.weight == 9
    assert task.value == 9
    assert task.value_at(1) == 0
    assert task.value_at(5) == 9
    assert task.value_at(100) == 9


def test_task_allows_zero_cost_and_zero_weight() -> None:
    task = Task.constant(id="noop", token_cost=0, value=0)

    assert task.token_cost == 0
    assert task.value == 0


def test_tasks_with_the_same_fields_are_equal() -> None:
    left = Task.constant(id="a", token_cost=5, value=9)
    right = Task.constant(id="a", token_cost=5, value=9)

    assert left == right


def test_task_is_immutable() -> None:
    task = Task.constant(id="a", token_cost=1, value=1)

    with pytest.raises(FrozenInstanceError):
        task.weight = 99  # type: ignore[misc]


def test_empty_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="id"):
        Task(id="", token_cost=1, weight=1)


def test_negative_token_cost_is_rejected() -> None:
    with pytest.raises(ValueError, match="token_cost"):
        Task(id="a", token_cost=-1, weight=1)


def test_negative_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="weight"):
        Task(id="a", token_cost=1, weight=-1)


def test_unknown_curve_is_rejected() -> None:
    with pytest.raises(ValueError, match="curve"):
        Task(id="a", token_cost=1, weight=1, curve="bandit")
