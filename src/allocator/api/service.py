"""Translate JSON into ``Task`` records and call the existing solvers."""

from __future__ import annotations

import time
from pathlib import Path

from allocator.allocate import allocate_dp
from allocator.greedy import allocate_greedy
from allocator.task import Task

from .models import (
    AllocatePredictedRequest,
    AllocateRequest,
    AllocateResponse,
    AssignmentOut,
    GreedyComparison,
    RouteRequest,
    RouteResponse,
)

MAX_TASKS = 100
MAX_BUDGET = 2000
MAX_TOKEN_COST = 2000
SOLVER_NAMES = ["allocate_dp", "allocate_greedy"]


def run_allocate(request: AllocateRequest) -> AllocateResponse:
    """Solve one instance. Raises ``ValueError`` for domain / size violations."""

    if not request.tasks:
        raise ValueError("task list is empty")
    if len(request.tasks) > MAX_TASKS:
        raise ValueError(f"at most {MAX_TASKS} tasks allowed")
    if request.budget > MAX_BUDGET:
        raise ValueError(f"budget must be at most {MAX_BUDGET}")
    for item in request.tasks:
        if item.token_cost > MAX_TOKEN_COST:
            raise ValueError(f"token_cost must be at most {MAX_TOKEN_COST}")

    tasks = tuple(
        Task(
            id=item.id,
            token_cost=item.token_cost,
            weight=item.weight,
            curve=item.curve,
        )
        for item in request.tasks
    )
    started = time.perf_counter()
    dp = allocate_dp(tasks, request.budget)
    dp_runtime_s = time.perf_counter() - started
    greedy = allocate_greedy(tasks, request.budget)

    delta = dp.total_value - greedy.total_value
    if greedy.total_value == 0:
        delta_pct: float | None = None
    else:
        delta_pct = 100.0 * delta / greedy.total_value

    return AllocateResponse(
        assignments=[
            AssignmentOut(id=item.task.id, tokens=item.tokens, value=item.value)
            for item in dp.assignments
        ],
        total_cost=dp.total_cost,
        total_value=dp.total_value,
        dp_runtime_s=dp_runtime_s,
        greedy=GreedyComparison(
            total_value=greedy.total_value,
            total_cost=greedy.total_cost,
            delta_vs_dp=delta,
            delta_vs_dp_pct=delta_pct,
        ),
        solvers=list(SOLVER_NAMES),
    )


def _v4_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_allocate_predicted(request: AllocatePredictedRequest) -> AllocateResponse:
    """DP/greedy on predicted envelopes. Never calls Anthropic."""

    import os

    from allocator.live.actions import default_actions
    from allocator.live.cache import load_cache
    from allocator.live.collect import load_questions
    from allocator.live.config import V4_CACHE_ENV, V4_QUESTIONS_ENV, V4_TRAIN_N
    from allocator.live.predict import fit_predictor, train_test_split
    from benchmarks.v4_replay import predicted_tasks

    if not request.questions:
        raise ValueError("question list is empty")
    if len(request.questions) > MAX_TASKS:
        raise ValueError(f"at most {MAX_TASKS} questions allowed")
    if request.budget > MAX_BUDGET:
        raise ValueError(f"budget must be at most {MAX_BUDGET}")

    root = _v4_repo_root()
    questions_path = Path(
        os.environ.get(
            V4_QUESTIONS_ENV,
            str(root / "benchmarks" / "live" / "questions_v4.json"),
        )
    )
    live_cache = root / "benchmarks" / "live" / "cache_v4.json"
    fixture_cache = root / "benchmarks" / "live" / "fixture_cache_v4.json"
    default_cache = live_cache if live_cache.is_file() else fixture_cache
    cache_path = Path(os.environ.get(V4_CACHE_ENV, str(default_cache)))
    train_pool = load_questions(questions_path)
    cache = load_cache(cache_path)
    train, _held = train_test_split(train_pool, n_train=min(V4_TRAIN_N, len(train_pool)))
    predictor = fit_predictor(train, cache)
    cap = max(action.cost_units for action in default_actions())
    items = [
        {"id": row.id, "question": row.question, "gold": row.gold}
        for row in request.questions
    ]
    tasks = predicted_tasks(items, predictor, cap=cap)
    started = time.perf_counter()
    dp = allocate_dp(tasks, request.budget)
    dp_runtime_s = time.perf_counter() - started
    greedy = allocate_greedy(tasks, request.budget)
    delta = dp.total_value - greedy.total_value
    if greedy.total_value == 0:
        delta_pct: float | None = None
    else:
        delta_pct = 100.0 * delta / greedy.total_value
    return AllocateResponse(
        assignments=[
            AssignmentOut(id=item.task.id, tokens=item.tokens, value=item.value)
            for item in dp.assignments
        ],
        total_cost=dp.total_cost,
        total_value=dp.total_value,
        dp_runtime_s=dp_runtime_s,
        greedy=GreedyComparison(
            total_value=greedy.total_value,
            total_cost=greedy.total_cost,
            delta_vs_dp=delta,
            delta_vs_dp_pct=delta_pct,
        ),
        solvers=list(SOLVER_NAMES),
    )


def run_route(request: RouteRequest) -> RouteResponse:
    """Prompt-only model choice. Never calls Anthropic. Does not run DP."""

    import os

    from allocator.live.cache import load_cache
    from allocator.live.collect import load_questions
    from allocator.live.config import V5_CACHE_ENV, V5_QUESTIONS_ENV, V5_TRAIN_N
    from allocator.live.router import choose_model, fit_router

    question = request.question.strip()
    if not question:
        raise ValueError("question is empty")

    root = _v4_repo_root()
    questions_path = Path(
        os.environ.get(
            V5_QUESTIONS_ENV,
            str(root / "benchmarks" / "live" / "questions_v5.json"),
        )
    )
    live_cache = root / "benchmarks" / "live" / "cache_v5.json"
    fixture_cache = root / "benchmarks" / "live" / "fixture_cache_v5.json"
    default_cache = live_cache if live_cache.is_file() else fixture_cache
    cache_path = Path(os.environ.get(V5_CACHE_ENV, str(default_cache)))
    train_pool = load_questions(questions_path)
    cache = load_cache(cache_path)
    predictor, tau, _train, _test = fit_router(
        train_pool, cache, n_train=min(V5_TRAIN_N, len(train_pool))
    )
    decision = choose_model(question, predictor, tau=tau)
    return RouteResponse(
        model=decision.model,
        p_haiku=decision.p_haiku,
        tau=decision.tau,
        reason=decision.reason,
    )
