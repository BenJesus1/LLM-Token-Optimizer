"""Tests for variable-token resource-allocation DP.

Hand-computed cases are worked out in comments *before* the assertions.
This solver assigns an integer token count per task; it does not place
KV-cache or GPU memory.
"""

import math

import pytest

from allocator import Task, allocate_dp, knapsack_dp


def test_hand_computed_split_beats_dumping_on_one_task() -> None:
    # Two identical log tasks, weight 1, cap 3, budget 3.
    # value(t) = log(1+t)
    #   (3, 0) or (0, 3): log(4)           ≈ 1.386
    #   (2, 1) or (1, 2): log(3)+log(2)    ≈ 1.792
    #   (1, 1) uses only 2 tokens          ≈ 1.386
    # Spreading is uniquely better. On a (2,1)/(1,2) tie the DP keeps the
    # smaller t for the later task, so a gets 2 and b gets 1.
    a = Task(id="a", token_cost=3, weight=1.0, curve="log")
    b = Task(id="b", token_cost=3, weight=1.0, curve="log")

    allocation = allocate_dp([a, b], budget=3)

    assert allocation.tokens_for("a") == 2
    assert allocation.tokens_for("b") == 1
    assert allocation.total_cost == 3
    assert allocation.total_value == pytest.approx(math.log(3) + math.log(2))
    assert allocation.total_value > math.log(4)


def test_caps_prevent_one_task_from_taking_the_whole_budget() -> None:
    a = Task(id="a", token_cost=1, weight=1.0, curve="log")
    b = Task(id="b", token_cost=1, weight=1.0, curve="log")

    allocation = allocate_dp([a, b], budget=5)

    assert allocation.tokens_for("a") == 1
    assert allocation.tokens_for("b") == 1
    assert allocation.total_cost == 2


def test_single_task_takes_min_of_cap_and_budget() -> None:
    task = Task(id="only", token_cost=10, weight=1.0, curve="log")

    allocation = allocate_dp([task], budget=4)

    assert allocation.tokens_for("only") == 4
    assert allocation.total_value == pytest.approx(math.log(5))


def test_zero_budget_allocates_nothing() -> None:
    task = Task(id="a", token_cost=5, weight=1.0, curve="log")

    allocation = allocate_dp([task], budget=0)

    assert allocation.assignments == ()
    assert allocation.total_value == 0


def test_empty_task_list_returns_empty_allocation() -> None:
    allocation = allocate_dp([], budget=10)

    assert allocation.assignments == ()


def test_constant_curve_embeds_zero_one_knapsack() -> None:
    # Constant tasks only score when they receive their full token_cost, so
    # allocate_dp must recover the 0/1 optimum on the five-task hand instance.
    draft = Task.constant(id="draft", token_cost=4, value=5)
    review = Task.constant(id="review", token_cost=3, value=4)
    cite = Task.constant(id="cite", token_cost=2, value=3)
    polish = Task.constant(id="polish", token_cost=5, value=6)
    translate = Task.constant(id="translate", token_cost=6, value=7)
    tasks = [draft, review, cite, polish, translate]

    variable = allocate_dp(tasks, budget=10)
    zero_one = knapsack_dp(tasks, budget=10)

    assert variable.selected == zero_one.selected
    assert variable.total_value == zero_one.total_value
    assert variable.total_cost == zero_one.total_cost
    for task in variable.selected:
        assert variable.tokens_for(task.id) == task.token_cost


def test_zero_weight_task_gets_no_tokens() -> None:
    useful = Task(id="useful", token_cost=4, weight=1.0, curve="log")
    filler = Task(id="filler", token_cost=4, weight=0.0, curve="log")

    allocation = allocate_dp([useful, filler], budget=4)

    assert allocation.tokens_for("useful") == 4
    assert allocation.tokens_for("filler") == 0


def test_negative_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="budget"):
        allocate_dp([Task(id="a", token_cost=1, weight=1.0)], budget=-1)
