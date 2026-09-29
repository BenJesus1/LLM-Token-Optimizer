"""v4 discrete actions: cost-unit envelopes for existing table-valued Tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from allocator.task import Task

from .cache import lookup
from .config import (
    COST_HAIKU_256,
    COST_HAIKU_64,
    MODEL,
    V4_K,
    V4_TOKEN_LEVELS,
)


@dataclass(frozen=True)
class Action:
    """One (model, max_tokens) option and its integer cost-unit price."""

    model: str
    max_tokens: int
    cost_units: int
    name: str


def default_actions(*, model: str = MODEL) -> tuple[Action, ...]:
    """Haiku skip is implicit (cost 0). Cheap = 64 / 1 unit; longer = 256 / 4 units."""

    return (
        Action(model=model, max_tokens=64, cost_units=COST_HAIKU_64, name="haiku_64"),
        Action(model=model, max_tokens=256, cost_units=COST_HAIKU_256, name="haiku_256"),
    )


def envelope_table(points: Sequence[tuple[int, float]]) -> tuple[tuple[int, float], ...]:
    """Strictly increasing costs with cumulative-max scores; drop dominated points.

    Existing ``Task.value_at`` reads the largest level ``<= t``. After this
    envelope, that equals the best affordable action (skip is t = 0).
    """

    ranked = sorted(((int(cost), float(score)) for cost, score in points), key=lambda p: p[0])
    out: list[tuple[int, float]] = []
    best = 0.0
    for cost, score in ranked:
        if cost <= 0:
            continue
        if score > best:
            best = score
            out.append((cost, best))
    if not out:
        cap = max((p[0] for p in ranked), default=1)
        out.append((max(1, cap), 0.0))
    return tuple(out)


def mean_score(
    cache: Mapping[str, Any],
    *,
    question_id: str,
    question: str,
    model: str,
    token_level: int,
    k: int = V4_K,
) -> float:
    """Mean of k binary scores. Missing samples count as 0.0."""

    total = 0.0
    for sample in range(k):
        entry = lookup(
            dict(cache),
            question_id=question_id,
            token_level=token_level,
            model=model,
            question=question,
            sample=sample,
        )
        if entry is not None:
            total += float(entry["score"])
    return total / float(k) if k else 0.0


def action_means(
    item: Mapping[str, Any],
    cache: Mapping[str, Any],
    actions: Sequence[Action],
    *,
    k: int = V4_K,
) -> list[tuple[int, float]]:
    """``(cost_units, mean_score)`` for each action on one question."""

    question_id = str(item["id"])
    question = str(item["question"])
    rows: list[tuple[int, float]] = []
    for action in actions:
        rows.append(
            (
                action.cost_units,
                mean_score(
                    cache,
                    question_id=question_id,
                    question=question,
                    model=action.model,
                    token_level=action.max_tokens,
                    k=k,
                ),
            )
        )
    return rows


def task_from_points(
    question_id: str,
    points: Sequence[tuple[int, float]],
    *,
    cap: int | None = None,
) -> Task:
    table = envelope_table(points)
    token_cost = cap if cap is not None else max(level for level, _score in table)
    return Task(
        id=question_id,
        token_cost=token_cost,
        weight=0.0,
        curve="table",
        value_table=table,
    )


def tasks_from_v4_cache(
    questions: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    *,
    actions: Sequence[Action] | None = None,
    k: int = V4_K,
    levels: Sequence[int] = V4_TOKEN_LEVELS,
) -> tuple[Task, ...]:
    """One envelope Task per question. ``token_cost`` is cost units, not tokens."""

    del levels  # actions carry max_tokens; kept for call-site symmetry with v3
    chosen = tuple(actions) if actions is not None else default_actions()
    cap = max(action.cost_units for action in chosen)
    return tuple(
        task_from_points(str(item["id"]), action_means(item, cache, chosen, k=k), cap=cap)
        for item in questions
    )
