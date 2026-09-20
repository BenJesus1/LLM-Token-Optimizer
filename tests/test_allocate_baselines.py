"""Tests for token-level greedy and random baselines (not 0/1 knapsack_greedy)."""

import math

import pytest

from allocator import Task, allocate_dp, allocate_greedy, allocate_random


def test_token_greedy_splits_two_identical_log_tasks() -> None:
    a = Task(id="a", token_cost=3, weight=1.0, curve="log")
    b = Task(id="b", token_cost=3, weight=1.0, curve="log")

    greedy = allocate_greedy([a, b], budget=3)
    exact = allocate_dp([a, b], budget=3)

    assert greedy.total_cost == 3
    assert greedy.total_value == pytest.approx(exact.total_value)
    assert greedy.total_value == pytest.approx(math.log(3) + math.log(2))


def test_token_greedy_jumps_constant_tasks_in_one_shot() -> None:
    # Unit steps of a constant task have zero gain; greedy must take the
    # whole cap or skip it. Here only one task fits.
    a = Task.constant(id="a", token_cost=4, value=10)
    b = Task.constant(id="b", token_cost=4, value=6)

    allocation = allocate_greedy([a, b], budget=4)

    assert allocation.tokens_for("a") == 4
    assert allocation.tokens_for("b") == 0
    assert allocation.total_value == 10


def test_random_is_reproducible_with_the_same_seed() -> None:
    tasks = tuple(
        Task(id=f"t{i}", token_cost=8, weight=1.0, curve="log") for i in range(5)
    )

    first = allocate_random(tasks, budget=12, seed=2026)
    second = allocate_random(tasks, budget=12, seed=2026)
    other = allocate_random(tasks, budget=12, seed=99)

    assert first.assignments == second.assignments
    assert first.assignments != other.assignments
    assert first.total_cost == 12


def test_random_respects_caps() -> None:
    task = Task(id="only", token_cost=3, weight=1.0, curve="log")

    allocation = allocate_random([task], budget=10, seed=0)

    assert allocation.tokens_for("only") == 3


def test_negative_budget_is_rejected() -> None:
    task = Task(id="a", token_cost=1, weight=1.0, curve="log")
    with pytest.raises(ValueError, match="budget"):
        allocate_greedy([task], budget=-1)
    with pytest.raises(ValueError, match="budget"):
        allocate_random([task], budget=-1, seed=0)
