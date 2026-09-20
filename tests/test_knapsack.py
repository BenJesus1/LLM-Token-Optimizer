"""Tests for the exact 0/1 knapsack DP solver.

Hand-computed cases are worked out in comments *before* the assertions so the
expected allocation is independent of the implementation.
"""

import pytest

from allocator import Task, knapsack_dp


def test_hand_computed_five_task_instance() -> None:
    # Tasks and budget:
    #   draft      cost 4  value 5
    #   review     cost 3  value 4
    #   cite       cost 2  value 3
    #   polish     cost 5  value 6
    #   translate  cost 6  value 7
    #   budget = 10
    #
    # Feasible subsets with the highest values:
    #   review + cite + polish = cost 10, value 13
    #   draft + review + cite  = cost  9, value 12
    #   draft + translate      = cost 10, value 12
    # No four-task subset fits (minimum four-task cost is 4+3+2+5 = 14).
    # Optimal value is 13, uniquely from {review, cite, polish}.
    draft = Task.constant(id="draft", token_cost=4, value=5)
    review = Task.constant(id="review", token_cost=3, value=4)
    cite = Task.constant(id="cite", token_cost=2, value=3)
    polish = Task.constant(id="polish", token_cost=5, value=6)
    translate = Task.constant(id="translate", token_cost=6, value=7)

    allocation = knapsack_dp(
        [draft, review, cite, polish, translate],
        budget=10,
    )

    assert allocation.selected == (review, cite, polish)
    assert allocation.total_cost == 10
    assert allocation.total_value == 13


def test_zero_budget_selects_nothing_when_every_task_costs_tokens() -> None:
    tasks = [
        Task.constant(id="a", token_cost=1, value=5),
        Task.constant(id="b", token_cost=2, value=9),
    ]

    allocation = knapsack_dp(tasks, budget=0)

    assert allocation.selected == ()
    assert allocation.total_value == 0


def test_zero_cost_positive_value_task_is_taken_even_at_zero_budget() -> None:
    free = Task.constant(id="free", token_cost=0, value=4)
    paid = Task.constant(id="paid", token_cost=3, value=10)

    allocation = knapsack_dp([free, paid], budget=0)

    assert allocation.selected == (free,)
    assert allocation.total_value == 4


def test_single_task_over_budget_is_skipped() -> None:
    heavy = Task.constant(id="heavy", token_cost=8, value=20)

    allocation = knapsack_dp([heavy], budget=5)

    assert allocation.selected == ()
    assert allocation.total_value == 0


def test_all_tasks_fit_when_budget_covers_their_cost() -> None:
    tasks = [
        Task.constant(id="a", token_cost=2, value=3),
        Task.constant(id="b", token_cost=3, value=4),
    ]

    allocation = knapsack_dp(tasks, budget=10)

    assert allocation.selected == (tasks[0], tasks[1])
    assert allocation.total_value == 7


def test_empty_task_list_returns_empty_allocation() -> None:
    allocation = knapsack_dp([], budget=10)

    assert allocation.selected == ()
    assert allocation.total_cost == 0
    assert allocation.total_value == 0


def test_zero_value_task_is_not_selected() -> None:
    useful = Task.constant(id="useful", token_cost=2, value=5)
    filler = Task.constant(id="filler", token_cost=1, value=0)

    allocation = knapsack_dp([useful, filler], budget=3)

    assert allocation.selected == (useful,)
    assert allocation.total_value == 5


def test_negative_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="budget"):
        knapsack_dp([Task.constant(id="a", token_cost=1, value=1)], budget=-1)
