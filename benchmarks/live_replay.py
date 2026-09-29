"""Replay a live cache through DP, greedy, and random. No API key."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from allocator import allocate_dp, allocate_greedy, allocate_random
from allocator.live.cache import load_cache
from allocator.live.collect import load_questions
from allocator.live.config import MODEL, RANDOM_SEED, TOKEN_LEVELS, global_budget
from allocator.live.tasks import tasks_from_cache

LIVE_DIR = Path(__file__).resolve().parent / "live"
DEFAULT_QUESTIONS = LIVE_DIR / "questions.json"
DEFAULT_CACHE = LIVE_DIR / "fixture_cache.json"
RESULTS_PATH = LIVE_DIR / "results.json"


def replay(
    questions: Sequence[dict[str, Any]],
    cache: dict[str, Any],
    *,
    budget: int | None = None,
    model: str = MODEL,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run the three v1 token-level solvers on table tasks from ``cache``."""

    tasks = tasks_from_cache(questions, cache, model=model, levels=TOKEN_LEVELS)
    b = budget if budget is not None else global_budget(len(tasks))
    dp = allocate_dp(tasks, b)
    greedy = allocate_greedy(tasks, b)
    random_alloc = allocate_random(tasks, b, seed=seed)
    usage_total = sum(
        int((entry.get("usage") or {}).get("total_tokens") or 0)
        for entry in cache.get("entries", [])
    )
    return {
        "source": cache.get("source", "cache"),
        "model": model,
        "n": len(tasks),
        "budget": b,
        "seed": seed,
        "cached_usage_total_tokens": usage_total,
        "note": "one completion per (question, level); scores are cached 0/1, not P(correct)",
        "rows": [
            {
                "solver": "dp",
                "total_value": dp.total_value,
                "total_cost": dp.total_cost,
            },
            {
                "solver": "greedy",
                "total_value": greedy.total_value,
                "total_cost": greedy.total_cost,
            },
            {
                "solver": "random",
                "total_value": random_alloc.total_value,
                "total_cost": random_alloc.total_cost,
            },
        ],
    }


def save_results(payload: dict[str, Any], path: Path = RESULTS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay cached live scores")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    questions = load_questions(args.questions)
    cache = load_cache(args.cache)
    payload = replay(questions, cache, budget=args.budget)
    saved = save_results(payload, args.out)
    print(f"wrote {saved}")
    for row in payload["rows"]:
        print(
            f"{row['solver']:<7} value={row['total_value']:.4f}  "
            f"cost={row['total_cost']}"
        )


if __name__ == "__main__":
    main()
