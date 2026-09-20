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


def test_hand_computed_three_task_weighted_log_instance() -> None:
    # a: cap 2, weight 2, log  →  value_a(t) = 2 log(1+t)
    # b: cap 2, weight 1, log  →  value_b(t) = log(1+t)
    # c: cap 2, weight 1, log  →  value_c(t) = log(1+t)
    # budget = 3
    #
    # Closed forms used below:
    #   2 log 2 ≈ 1.3863    log 2 ≈ 0.6931
    #   2 log 3 ≈ 2.1972    log 3 ≈ 1.0986
    #
    # Feasible assignments with the highest totals (each coord 0..2, sum ≤ 3):
    #   (2, 1, 0) = 2 log 3 + log 2 ≈ 2.8903
    #   (2, 0, 1) = 2 log 3 + log 2 ≈ 2.8903
    #   (1, 1, 1) = 2 log 2 + log 2 + log 2 ≈ 2.7726
    #   (1, 2, 0) = 2 log 2 + log 3         ≈ 2.4849
    # Dumping on a, (2, 0, 0) = 2 log 3 ≈ 2.1972, is strictly worse.
    #
    # DP table (rows = after task i, columns = budget w = 0..3).
    # Ties keep the smaller t.
    #
    # after a:  dp = [0, 1.3863, 2.1972, 2.1972]   choice t = [0, 1, 2, 2]
    # after b, w=3:
    #   t=0 → 2.1972
    #   t=1 → 2.1972 + 0.6931 = 2.8903
    #   t=2 → 1.3863 + 1.0986 = 2.4849   → choose t=1
    # after c, w=3:
    #   t=0 → 2.8903
    #   t=1 → 2.1972 + 0.6931 = 2.8903 (tie, keep t=0)
    #   t=2 → 1.3863 + 1.0986 = 2.4849   → choose t=0
    #
    # Reconstruct: c=0, remaining 3; b=1, remaining 2; a=2.
    a = Task(id="a", token_cost=2, weight=2.0, curve="log")
    b = Task(id="b", token_cost=2, weight=1.0, curve="log")
    c = Task(id="c", token_cost=2, weight=1.0, curve="log")

    allocation = allocate_dp([a, b, c], budget=3)

    assert allocation.tokens_for("a") == 2
    assert allocation.tokens_for("b") == 1
    assert allocation.tokens_for("c") == 0
    assert allocation.total_cost == 3
    assert allocation.total_value == pytest.approx(2 * math.log(3) + math.log(2))


def test_hand_computed_three_task_value_is_optimal_among_all_assignments() -> None:
    # Same instance as test_hand_computed_three_task_weighted_log_instance.
    # Enumerate every cap-respecting assignment so a slip in the table above
    # cannot sneak through: the DP value must equal the exhaustive max.
    tasks = [
        Task(id="a", token_cost=2, weight=2.0, curve="log"),
        Task(id="b", token_cost=2, weight=1.0, curve="log"),
        Task(id="c", token_cost=2, weight=1.0, curve="log"),
    ]
    budget = 3
    best = 0.0
    for ta in range(3):
        for tb in range(3):
            for tc in range(3):
                if ta + tb + tc > budget:
                    continue
                value = (
                    tasks[0].value_at(ta)
                    + tasks[1].value_at(tb)
                    + tasks[2].value_at(tc)
                )
                if value > best:
                    best = value

    allocation = allocate_dp(tasks, budget)

    assert allocation.total_value == pytest.approx(best)
    assert best == pytest.approx(2 * math.log(3) + math.log(2))


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
