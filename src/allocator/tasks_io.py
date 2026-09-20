"""Load Task records from JSON or CSV files."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

from .task import Task

_REQUIRED = ("id", "token_cost", "weight")


def _task_from_row(row: Mapping[str, object]) -> Task:
    """Build a Task from one JSON object or CSV row."""
    missing = [key for key in _REQUIRED if key not in row]
    if missing:
        raise ValueError(f"task is missing fields: {', '.join(missing)}")
    curve = str(row["curve"]) if "curve" in row and row["curve"] not in (None, "") else "log"
    return Task(
        id=str(row["id"]),
        token_cost=int(row["token_cost"]),
        weight=float(row["weight"]),
        curve=curve,
    )


def load_tasks(path: Path) -> tuple[Task, ...]:
    """Read tasks from a ``.json`` list or a ``.csv`` table."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("JSON task file must be a list of objects")
        return tuple(_task_from_row(item) for item in raw)
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return tuple(_task_from_row(row) for row in rows)
    raise ValueError(f"unsupported task file type: {path.suffix}")
