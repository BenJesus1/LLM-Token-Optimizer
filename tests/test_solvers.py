"""Hand-computed cases and edges that both solvers must satisfy.

Expected allocations were computed on paper *before* the assertions were
written. Greedy is allowed to lose on optimality; it is not allowed to
violate budget or to select a task that does not fit.
"""

from collections.abc import Callable, Sequence

import pytest

from allocator import Allocation, Task, knapsack_dp, knapsack_greedy

Solver = Callable[[Sequence[Task], int], Allocation]

SOLVERS = pytest.mark.parametrize(
    "solve",
    [knapsack_dp, knapsack_greedy],
    ids=["dp", "greedy"],
)


def _four_tasks() -> tuple[Task, Task, Task, Task]:
    return (
        Task(id="a", token_cost=2, value=3),
        Task(id="b", token_cost=3, value=4),
        Task(id="c", token_cost=4, value=5),
        Task(id="d", token_cost=5, value=6),
    )


def test_hand_computed_four_task_dp() -> None:
    # Tasks: a(2,3), b(3,4), c(4,5), d(5,6). Budget 7.
    # Feasible subsets:
    #   a+b = cost 5, value 7
    #   a+c = cost 6, value 8
    #   a+d = cost 7, value 9
    #   b+c = cost 7, value 9
    #   b+d / c+d / any three-task set exceed the budget.
    # Optimal value is 9. The DP skip-on-tie reconstruction keeps b+c
    # (d ties skip at the last row, then c and b are required).
    a, b, c, d = _four_tasks()

    allocation = knapsack_dp([a, b, c, d], budget=7)

    assert allocation.selected == (b, c)
    assert allocation.total_cost == 7
    assert allocation.total_value == 9


def test_hand_computed_four_task_greedy() -> None:
    # Same instance. Densities: a=1.5, b~1.33, c=1.25, d=1.2.
    # Take a (remaining 5), take b (remaining 2); c and d do not fit.
    # Greedy value 7 is strictly below the optimal 9.
    a, b, c, d = _four_tasks()

    allocation = knapsack_greedy([a, b, c, d], budget=7)

    assert allocation.selected == (a, b)
    assert allocation.total_cost == 5
    assert allocation.total_value == 7


def test_hand_computed_five_task_both_solvers() -> None:
    # draft(4,5), review(3,4), cite(2,3), polish(5,6), translate(6,7). Budget 10.
    # Best feasible subsets:
    #   review+cite+polish = cost 10, value 13   <- unique optimum
    #   draft+review+cite  = cost  9, value 12
    #   draft+translate    = cost 10, value 12
    # Greedy density order cite, review, draft, polish, translate fills
    # cite+review+draft (cost 9, value 12) and then cannot fit polish.
    draft = Task(id="draft", token_cost=4, value=5)
    review = Task(id="review", token_cost=3, value=4)
    cite = Task(id="cite", token_cost=2, value=3)
    polish = Task(id="polish", token_cost=5, value=6)
    translate = Task(id="translate", token_cost=6, value=7)
    tasks = [draft, review, cite, polish, translate]

    exact = knapsack_dp(tasks, budget=10)
    greedy = knapsack_greedy(tasks, budget=10)

    assert exact.selected == (review, cite, polish)
    assert exact.total_value == 13
    assert greedy.selected == (draft, review, cite)
    assert greedy.total_value == 12
    assert greedy.total_value < exact.total_value
    assert greedy.total_cost <= 10
    assert exact.total_cost <= 10


@SOLVERS
def test_zero_budget_selects_nothing_when_every_task_costs_tokens(solve: Solver) -> None:
    tasks = [
        Task(id="a", token_cost=1, value=5),
        Task(id="b", token_cost=2, value=9),
    ]

    allocation = solve(tasks, 0)

    assert allocation.selected == ()
    assert allocation.total_value == 0


@SOLVERS
def test_one_task_over_budget_is_skipped_and_the_rest_can_still_be_taken(solve: Solver) -> None:
    # heavy costs 10 > budget 5, so it cannot appear.
    # light+mid = cost 5, value 7 — the only feasible pair, so both solvers
    # must return that set (greedy skips heavy first, then takes light, mid).
    heavy = Task(id="heavy", token_cost=10, value=50)
    light = Task(id="light", token_cost=2, value=3)
    mid = Task(id="mid", token_cost=3, value=4)

    allocation = solve([heavy, light, mid], 5)

    assert allocation.selected == (light, mid)
    assert allocation.total_cost == 5
    assert allocation.total_value == 7


@SOLVERS
def test_tied_values_only_one_fits(solve: Solver) -> None:
    # a and b have the same value 8 and the same cost 4. Budget 4, so exactly
    # one can be chosen; either is optimal. DP skip-on-tie keeps a (already
    # in the table). Greedy ranks them equal and stable-sorts, so also a.
    a = Task(id="a", token_cost=4, value=8)
    b = Task(id="b", token_cost=4, value=8)

    allocation = solve([a, b], 4)

    assert allocation.selected == (a,)
    assert allocation.total_value == 8


@SOLVERS
def test_tied_values_both_fit(solve: Solver) -> None:
    # a and b are identical (cost 2, value 5). Budget 4 fits both; taking
    # both is uniquely optimal (value 10 vs 5).
    a = Task(id="a", token_cost=2, value=5)
    b = Task(id="b", token_cost=2, value=5)

    allocation = solve([a, b], 4)

    assert allocation.selected == (a, b)
    assert allocation.total_value == 10


@SOLVERS
def test_every_task_individually_over_budget(solve: Solver) -> None:
    tasks = [
        Task(id="a", token_cost=5, value=10),
        Task(id="b", token_cost=6, value=12),
        Task(id="c", token_cost=7, value=20),
    ]

    allocation = solve(tasks, 4)

    assert allocation.selected == ()
    assert allocation.total_cost == 0
    assert allocation.total_value == 0
