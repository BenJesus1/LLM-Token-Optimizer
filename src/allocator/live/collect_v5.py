"""v5 two-model collect: Haiku + Sonnet at 256 tokens. Opt-in --live."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .cache import load_cache, save_cache
from .client import Completer, anthropic_completer
from .collect import collect, load_dotenv, load_questions, planned_misses
from .config import (
    MAX_NEW_CALLS_V5,
    MODEL_CHEAP,
    MODEL_QUALITY,
    V4_WORK_PROMPT_TEMPLATE,
    V5_K,
    V5_MAX_TOKENS,
    V5_MODELS,
    V5_TEMPERATURE,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QUESTIONS = REPO_ROOT / "benchmarks" / "live" / "questions_v5.json"
DEFAULT_CACHE = REPO_ROOT / "benchmarks" / "live" / "cache_v5.json"


def collect_two_models(
    questions: Sequence[Mapping[str, Any]],
    cache: dict[str, Any],
    *,
    live: bool,
    completers: Mapping[str, Completer] | None = None,
    max_calls: int = MAX_NEW_CALLS_V5,
) -> dict[str, Any]:
    """Fill Haiku and quality misses. Tests inject stub completers."""

    cache["prompt_template"] = V4_WORK_PROMPT_TEMPLATE
    cache["k"] = V5_K
    cache["temperature"] = V5_TEMPERATURE
    cache["models"] = list(V5_MODELS)
    total_misses = 0
    for model in V5_MODELS:
        total_misses += len(
            planned_misses(
                questions,
                cache,
                model=model,
                levels=(V5_MAX_TOKENS,),
                k=V5_K,
            )
        )
    if total_misses > max_calls:
        raise ValueError(
            f"collect would make {total_misses} new HTTP calls; cap is {max_calls}"
        )
    for model in V5_MODELS:
        completer = None if completers is None else completers.get(model)
        collect(
            questions,
            cache,
            live=live,
            completer=completer,
            model=model,
            levels=(V5_MAX_TOKENS,),
            k=V5_K,
            max_calls=max_calls,
        )
    return cache


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Collect v5 Haiku+Sonnet scores")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        questions = load_questions(args.questions)
        cache = load_cache(args.cache)
        completers: dict[str, Completer] | None = None
        if args.live:
            key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not key:
                raise ValueError("ANTHROPIC_API_KEY is missing")
            completers = {
                model: anthropic_completer(
                    api_key=key,
                    model=model,
                    temperature=V5_TEMPERATURE,
                )
                for model in V5_MODELS
            }
        collect_two_models(
            questions,
            cache,
            live=args.live,
            completers=completers,
        )
        save_cache(cache, args.cache)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.cache} ({len(cache['entries'])} entries)")
    print(f"models: {MODEL_CHEAP} / {MODEL_QUALITY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
