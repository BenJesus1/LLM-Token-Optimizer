"""Build table-valued Tasks from a live cache. Offline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from allocator.task import Task

from .cache import lookup
from .config import MODEL, TOKEN_LEVELS


def tasks_from_cache(
    questions: Sequence[Mapping[str, Any]],
    cache: dict[str, Any],
    *,
    model: str = MODEL,
    levels: Sequence[int] = TOKEN_LEVELS,
) -> tuple[Task, ...]:
    """One Task per question. Missing cache rows score 0.0 at that level."""

    cap = max(levels)
    tasks: list[Task] = []
    for item in questions:
        question_id = str(item["id"])
        question = str(item["question"])
        rows: list[tuple[int, float]] = []
        for level in levels:
            entry = lookup(
                cache,
                question_id=question_id,
                token_level=level,
                model=model,
                question=question,
            )
            score = float(entry["score"]) if entry is not None else 0.0
            rows.append((int(level), score))
        tasks.append(
            Task(
                id=question_id,
                token_cost=cap,
                weight=0.0,
                curve="table",
                value_table=tuple(rows),
            )
        )
    return tuple(tasks)
