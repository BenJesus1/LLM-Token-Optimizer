"""Score a completion against gold. No network."""

from __future__ import annotations

import re

_NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")
_LAST_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    """Strip, lowercase, collapse whitespace."""

    return " ".join(text.strip().lower().split())


def completion_has_number(completion: str) -> bool:
    """True if any integer or decimal appears. Used by v5 cascade (no gold)."""

    return bool(_LAST_NUMBER.search(completion or ""))


def score_answer(completion: str, gold: str) -> float:
    """Return 1.0 or 0.0 using the v3 scoring rule."""

    gold_n = normalize(gold)
    comp_n = normalize(completion)
    if not comp_n:
        return 0.0
    if _NUMERIC.fullmatch(gold_n):
        found = _LAST_NUMBER.findall(comp_n)
        if not found:
            return 0.0
        try:
            return 1.0 if float(found[-1]) == float(gold_n) else 0.0
        except ValueError:
            return 0.0
    return 1.0 if comp_n == gold_n else 0.0
