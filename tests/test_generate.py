"""Tests for seeded synthetic task generation."""

import pytest

from allocator import CURVES
from benchmarks import DEFAULT_SEED, SCALES, generate_scale_suite, generate_set, generate_tasks


def test_generate_tasks_has_requested_size() -> None:
    tasks = generate_tasks(10, seed=DEFAULT_SEED)

    assert len(tasks) == 10
    assert [task.id for task in tasks] == [f"t{i:04d}" for i in range(10)]


def test_same_seed_reproduces_the_same_tasks() -> None:
    first = generate_tasks(20, seed=DEFAULT_SEED)
    second = generate_tasks(20, seed=DEFAULT_SEED)

    assert first == second


def test_different_seeds_change_the_draw() -> None:
    first = generate_tasks(20, seed=1)
    second = generate_tasks(20, seed=2)

    assert first != second


def test_costs_weights_and_curves_are_in_range() -> None:
    tasks = generate_tasks(50, seed=DEFAULT_SEED)

    for task in tasks:
        assert 1 <= task.token_cost <= 20
        assert 0.5 <= task.weight <= 5.0
        assert task.curve in CURVES


def test_scale_suite_covers_10_100_and_1000() -> None:
    suite = generate_scale_suite(seed=DEFAULT_SEED)

    assert tuple(suite) == SCALES
    for n, instance in suite.items():
        assert len(instance.tasks) == n
        assert instance.budget >= 1
        total_cap = sum(task.token_cost for task in instance.tasks)
        assert instance.budget == total_cap // 4


def test_scale_suite_is_reproducible() -> None:
    assert generate_scale_suite(seed=DEFAULT_SEED) == generate_scale_suite(seed=DEFAULT_SEED)


def test_empty_set_has_zero_budget() -> None:
    instance = generate_set(0, seed=DEFAULT_SEED)

    assert instance.tasks == ()
    assert instance.budget == 0


def test_negative_n_is_rejected() -> None:
    with pytest.raises(ValueError, match="n"):
        generate_tasks(-1)
