"""v4 collect: k=3, temperature 0.5, questions_v4.json → cache_v4.json. Opt-in --live."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .actions import default_actions
from .cache import load_cache, save_cache
from .client import anthropic_completer
from .collect import collect, load_dotenv, load_questions
from .config import MAX_NEW_CALLS_V4, MODEL, V4_K, V4_TEMPERATURE, V4_TOKEN_LEVELS

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QUESTIONS = REPO_ROOT / "benchmarks" / "live" / "questions_v4.json"
DEFAULT_CACHE = REPO_ROOT / "benchmarks" / "live" / "cache_v4.json"


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Collect v4 k=3 live scores (new cache file)")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call Anthropic for cache misses (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--show-work",
        action="store_true",
        help="Require step-by-step work so 64 tokens can miss the final answer",
    )
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        questions = load_questions(args.questions)
        cache = load_cache(args.cache)
        cache.setdefault("temperature", V4_TEMPERATURE)
        cache.setdefault("k", V4_K)
        if args.show_work:
            from .config import V4_WORK_PROMPT_TEMPLATE

            cache["prompt_template"] = V4_WORK_PROMPT_TEMPLATE
        completer = None
        if args.live:
            key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not key:
                raise ValueError("ANTHROPIC_API_KEY is missing")
            completer = anthropic_completer(
                api_key=key,
                model=MODEL,
                temperature=V4_TEMPERATURE,
            )
        collect(
            questions,
            cache,
            live=args.live,
            completer=completer,
            model=MODEL,
            levels=V4_TOKEN_LEVELS,
            k=V4_K,
            max_calls=MAX_NEW_CALLS_V4,
        )
        save_cache(cache, args.cache)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.cache} ({len(cache['entries'])} entries)")
    print(f"actions: {[a.name for a in default_actions()]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
