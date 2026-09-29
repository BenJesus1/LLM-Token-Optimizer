"""CLI: print the v5 model choice for one prompt. No Anthropic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cache import load_cache
from .collect import load_questions
from .config import V5_TRAIN_N
from .router import choose_model, fit_router

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QUESTIONS = REPO_ROOT / "benchmarks" / "live" / "questions_v5.json"
DEFAULT_CACHE = REPO_ROOT / "benchmarks" / "live" / "fixture_cache_v5.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v5 prompt-only model route")
    parser.add_argument("--question", required=True)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args(argv)
    questions = load_questions(args.questions)
    live = REPO_ROOT / "benchmarks" / "live" / "cache_v5.json"
    cache_path = args.cache
    if args.cache == DEFAULT_CACHE and live.is_file():
        cache_path = live
    cache = load_cache(cache_path)
    predictor, tau, _train, _test = fit_router(questions, cache, n_train=min(V5_TRAIN_N, len(questions)))
    decision = choose_model(args.question, predictor, tau=tau)
    print(
        json.dumps(
            {
                "model": decision.model,
                "p_haiku": decision.p_haiku,
                "tau": decision.tau,
                "reason": decision.reason,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
