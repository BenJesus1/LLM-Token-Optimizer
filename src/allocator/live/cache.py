"""JSON cache of live completions. Hits skip the network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import MODEL, PROMPT_TEMPLATE, render_prompt


def prompt_hash(question: str, template: str = PROMPT_TEMPLATE) -> str:
    filled = template.format(question=question)
    return hashlib.sha256(filled.encode("utf-8")).hexdigest()


def entry_key(entry: dict[str, Any]) -> tuple[str, int, str, str, int]:
    """Cache identity. ``sample`` defaults to 0 so v3 rows still hit."""

    return (
        str(entry["question_id"]),
        int(entry["token_level"]),
        str(entry["model"]),
        str(entry["prompt_hash"]),
        int(entry.get("sample", 0)),
    )


def empty_cache(*, model: str = MODEL) -> dict[str, Any]:
    return {"model": model, "prompt_template": PROMPT_TEMPLATE, "entries": []}


def load_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return empty_cache()
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def index_entries(
    payload: dict[str, Any],
) -> dict[tuple[str, int, str, str, int], dict[str, Any]]:
    return {entry_key(entry): entry for entry in payload.get("entries", [])}


def lookup(
    payload: dict[str, Any],
    *,
    question_id: str,
    token_level: int,
    model: str,
    question: str,
    sample: int = 0,
) -> dict[str, Any] | None:
    key = (
        question_id,
        token_level,
        model,
        prompt_hash(question, payload.get("prompt_template", PROMPT_TEMPLATE)),
        int(sample),
    )
    return index_entries(payload).get(key)
