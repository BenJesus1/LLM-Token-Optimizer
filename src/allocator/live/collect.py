"""Collect cached live completions. pytest must inject a stub completer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .cache import load_cache, lookup, prompt_hash, save_cache
from .client import Completer, anthropic_completer
from .config import MAX_NEW_CALLS, MODEL, PROMPT_TEMPLATE, TOKEN_LEVELS
from .score import score_answer

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QUESTIONS = REPO_ROOT / "benchmarks" / "live" / "questions.json"
DEFAULT_CACHE = REPO_ROOT / "benchmarks" / "live" / "cache.json"


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines into os.environ if the key is not already set."""

    env_path = path if path is not None else REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_questions(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("questions file must be a non-empty JSON list")
    return list(raw)


def planned_misses(
    questions: Sequence[Mapping[str, Any]],
    cache: dict[str, Any],
    *,
    model: str,
    levels: Sequence[int],
    k: int = 1,
) -> list[tuple[dict[str, Any], int, int]]:
    """Return ``(question, token_level, sample)`` triples not yet in ``cache``."""

    misses: list[tuple[dict[str, Any], int, int]] = []
    for item in questions:
        for level in levels:
            for sample in range(k):
                hit = lookup(
                    cache,
                    question_id=str(item["id"]),
                    token_level=int(level),
                    model=model,
                    question=str(item["question"]),
                    sample=sample,
                )
                if hit is None:
                    misses.append((dict(item), int(level), sample))
    return misses


def collect(
    questions: Sequence[Mapping[str, Any]],
    cache: dict[str, Any],
    *,
    live: bool,
    completer: Completer | None,
    model: str = MODEL,
    levels: Sequence[int] = TOKEN_LEVELS,
    k: int = 1,
    max_calls: int = MAX_NEW_CALLS,
) -> dict[str, Any]:
    """Fill cache misses. ``live`` plus a completer are required when misses exist."""

    cache.setdefault("model", model)
    cache.setdefault("prompt_template", PROMPT_TEMPLATE)
    cache.setdefault("entries", [])
    template = str(cache.get("prompt_template") or PROMPT_TEMPLATE)
    misses = planned_misses(questions, cache, model=model, levels=levels, k=k)
    if len(misses) > max_calls:
        raise ValueError(
            f"collect would make {len(misses)} new HTTP calls; cap is {max_calls}"
        )
    if misses and not live:
        raise ValueError("cache is incomplete; pass --live to call the API")
    if misses and completer is None:
        raise ValueError("live collect requires a completer")
    for item, level, sample in misses:
        assert completer is not None
        prompt = template.format(question=str(item["question"]))
        text, usage = completer(prompt=prompt, max_tokens=level)
        row: dict[str, Any] = {
            "question_id": str(item["id"]),
            "token_level": level,
            "model": model,
            "prompt_hash": prompt_hash(str(item["question"]), template),
            "completion": text,
            "score": score_answer(text, str(item["gold"])),
            "usage": usage,
        }
        if k > 1 or sample != 0:
            row["sample"] = sample
        cache["entries"].append(row)
    return cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect live LLM scores into a cache")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call Anthropic for cache misses (requires ANTHROPIC_API_KEY)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    try:
        questions = load_questions(args.questions)
        cache = load_cache(args.cache)
        completer = None
        if args.live:
            key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if not key:
                raise ValueError("ANTHROPIC_API_KEY is missing")
            completer = anthropic_completer(api_key=key, model=MODEL)
        collect(questions, cache, live=args.live, completer=completer)
        save_cache(cache, args.cache)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.cache} ({len(cache['entries'])} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
