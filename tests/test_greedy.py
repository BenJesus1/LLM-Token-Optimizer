"""Tests for the greedy value-per-token baseline.

Hand-computed cases are worked out in comments *before* the assertions.
Greedy is a comparison point: some tests exist specifically to show it
can be suboptimal versus the DP solver.
"""

import pytest

from allocator import Task, knapsack_dp, knapsack_greedy


def test_greedy_takes_highest_density_tasks_that_fit() -> None:
    # Densities: cite 1.5, review ~1.33, draft 1.25, polish 1.2, translate ~1.17
    # Budget 10, consider in that order:
    #   cite      cost 2  remaining 8
    #   review    cost 3  remaining 5
    #   draft     cost 4  remaining 1
    #   polish / translate do not fit
    # Greedy set: {cite, review, draft}, value 12, cost 9.
    draft = Task(id="draft", token_cost=4, value=5)
    review = Task(id="review", token_cost=3, value=4)
    cite = Task(id="cite", token_cost=2, value=3)
    polish = Task(id="polish", token_cost=5, value=6)
    translate = Task(id="translate", token_cost=6, value=7)

    allocation = knapsack_greedy(
        [draft, review, cite, polish, translate],
        budget=10,
    )

    assert allocation.selected == (draft, review, cite)
    assert allocation.total_cost == 9
    assert allocation.total_value == 12


def test_greedy_is_suboptimal_on_the_five_task_instance() -> None:
    # Same instance as the DP hand example. DP selects {review, cite, polish}
    # for value 13; greedy stops at value 12 (see previous test).
    draft = Task(id="draft", token_cost=4, value=5)
    review = Task(id="review", token_cost=3, value=4)
    cite = Task(id="cite", token_cost=2, value=3)
    polish = Task(id="polish", token_cost=5, value=6)
    translate = Task(id="translate", token_cost=6, value=7)
    tasks = [draft, review, cite, polish, translate]

    greedy = knapsack_greedy(tasks, budget=10)
    exact = knapsack_dp(tasks, budget=10)

    assert greedy.total_value < exact.total_value
    assert exact.total_value == 13
    assert greedy.total_value == 12


def test_greedy_skips_an_over_budget_high_density_task() -> None:
    heavy = Task(id="heavy", token_cost=10, value=100)
    light = Task(id="light", token_cost=3, value=3)

    allocation = knapsack_greedy([heavy, light], budget=5)

    assert allocation.selected == (light,)
    assert allocation.total_value == 3


def test_zero_cost_positive_value_task_is_taken_first() -> None:
    free = Task(id="free", token_cost=0, value=4)
    paid = Task(id="paid", token_cost=3, value=10)

    allocation = knapsack_greedy([paid, free], budget=0)

    assert allocation.selected == (free,)
    assert allocation.total_value == 4


def test_zero_budget_selects_nothing_when_every_task_costs_tokens() -> None:
    tasks = [
        Task(id="a", token_cost=1, value=5),
        Task(id="b", token_cost=2, value=9),
    ]

    allocation = knapsack_greedy(tasks, budget=0)

    assert allocation.selected == ()


def test_empty_task_list_returns_empty_allocation() -> None:
    allocation = knapsack_greedy([], budget=10)

    assert allocation.selected == ()


def test_zero_value_task_is_not_selected() -> None:
    useful = Task(id="useful", token_cost=2, value=5)
    filler = Task(id="filler", token_cost=1, value=0)

    allocation = knapsack_greedy([useful, filler], budget=3)

    assert allocation.selected == (useful,)


def test_negative_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="budget"):
        knapsack_greedy([Task(id="a", token_cost=1, value=1)], budget=-1)
