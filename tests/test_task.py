"""Tests for the Task data model."""

from dataclasses import FrozenInstanceError

import pytest

from allocator import Task


def test_task_stores_id_cost_and_value() -> None:
    task = Task(id="summarize", token_cost=12, value=40)

    assert task.id == "summarize"
    assert task.token_cost == 12
    assert task.value == 40


def test_task_allows_zero_cost_and_zero_value() -> None:
    task = Task(id="noop", token_cost=0, value=0)

    assert task.token_cost == 0
    assert task.value == 0


def test_tasks_with_the_same_fields_are_equal() -> None:
    left = Task(id="a", token_cost=5, value=9)
    right = Task(id="a", token_cost=5, value=9)

    assert left == right


def test_task_is_immutable() -> None:
    task = Task(id="a", token_cost=1, value=1)

    with pytest.raises(FrozenInstanceError):
        task.value = 99  # type: ignore[misc]


def test_empty_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="id"):
        Task(id="", token_cost=1, value=1)


def test_negative_token_cost_is_rejected() -> None:
    with pytest.raises(ValueError, match="token_cost"):
        Task(id="a", token_cost=-1, value=1)


def test_negative_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="value"):
        Task(id="a", token_cost=1, value=-1)
