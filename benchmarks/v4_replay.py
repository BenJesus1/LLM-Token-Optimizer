"""Replay v4: oracle vs predicted vs uniform policies. No API key."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from allocator import allocate_dp, allocate_greedy, allocate_random
from allocator.allocation import Allocation, allocation_from_counts
from allocator.live.actions import (
    default_actions,
    task_from_points,
    tasks_from_v4_cache,
)
from allocator.live.cache import load_cache
from allocator.live.collect import load_questions
from allocator.live.config import (
    COST_HAIKU_256,
    COST_HAIKU_64,
    MODEL,
    RANDOM_SEED,
    V4_K,
    V4_TOKEN_LEVELS,
    V4_TRAIN_N,
    global_budget,
)
from allocator.live.predict import (
    assert_no_label_leak,
    fit_predictor,
    predicted_points,
    train_test_split,
)
from allocator.task import Task

LIVE_DIR = Path(__file__).resolve().parent / "live"
DEFAULT_QUESTIONS = LIVE_DIR / "questions_v4.json"
DEFAULT_CACHE = LIVE_DIR / "fixture_cache_v4.json"
RESULTS_PATH = LIVE_DIR / "v4_results.json"
ROUNDTRIP_PATH = LIVE_DIR / "v4_roundtrip.json"
CHART_PATH = Path(__file__).resolve().parent / "charts" / "v4_realized_score_vs_policy.png"


def v4_budget(n: int, *, cap: int | None = None) -> int:
    action_cap = cap if cap is not None else max(a.cost_units for a in default_actions())
    return global_budget(n, cap=action_cap)


def uniform_counts(tasks: Sequence[Task], cost_per: int, budget: int) -> list[int]:
    remaining = budget
    counts = [0] * len(tasks)
    for i, task in enumerate(tasks):
        if cost_per <= remaining and cost_per <= task.token_cost:
            counts[i] = cost_per
            remaining -= cost_per
    return counts


def realized_of(allocation: Allocation, realized_tasks: Sequence[Task]) -> tuple[float, int]:
    """Score a (possibly predicted) assignment on realized envelope tables."""

    by_id = {task.id: task for task in realized_tasks}
    value = 0.0
    cost = 0
    for item in allocation.assignments:
        real = by_id[item.task.id]
        value += real.value_at(item.tokens)
        cost += item.tokens
    return value, cost


def predicted_tasks(
    questions: Sequence[Mapping[str, Any]],
    predictor: Any,
    *,
    cap: int,
) -> tuple[Task, ...]:
    return tuple(
        task_from_points(
            str(item["id"]),
            predicted_points(item, predictor),
            cap=cap,
        )
        for item in questions
    )


def policy_row(name: str, value: float, cost: int) -> dict[str, Any]:
    return {"policy": name, "realized_value": value, "cost_units": cost}


def replay_v4(
    questions: Sequence[Mapping[str, Any]],
    cache: dict[str, Any],
    *,
    budget: int | None = None,
    seed: int = RANDOM_SEED,
    n_train: int = V4_TRAIN_N,
) -> dict[str, Any]:
    """Oracle / predicted / uniform comparison on the held-out split."""

    train, test = train_test_split(questions, seed=seed, n_train=n_train)
    cap = max(a.cost_units for a in default_actions())
    b = budget if budget is not None else v4_budget(len(test), cap=cap)
    realized = tasks_from_v4_cache(test, cache, k=V4_K, levels=V4_TOKEN_LEVELS)
    predictor = fit_predictor(train, cache, model=MODEL, levels=V4_TOKEN_LEVELS, k=V4_K)
    assert_no_label_leak(predictor, test)
    predicted = predicted_tasks(test, predictor, cap=cap)

    oracle_dp = allocate_dp(realized, b)
    oracle_greedy = allocate_greedy(realized, b)
    oracle_random = allocate_random(realized, b, seed=seed)
    pred_dp = allocate_dp(predicted, b)
    pred_greedy = allocate_greedy(predicted, b)
    pred_dp_real, pred_dp_cost = realized_of(pred_dp, realized)
    pred_greedy_real, pred_greedy_cost = realized_of(pred_greedy, realized)

    cheap = allocation_from_counts(realized, uniform_counts(realized, COST_HAIKU_64, b))
    long = allocation_from_counts(realized, uniform_counts(realized, COST_HAIKU_256, b))

    roundtrip = []
    for item, pred_task, real_task in zip(test, predicted, realized, strict=True):
        tokens = pred_dp.tokens_for(str(item["id"]))
        roundtrip.append(
            {
                "id": str(item["id"]),
                "predicted_value_at_assignment": pred_task.value_at(tokens),
                "realized_value_at_assignment": real_task.value_at(tokens),
                "cost_units": tokens,
            }
        )

    beat_uniform = pred_dp_real > cheap.total_value + 1e-12
    return {
        "source": cache.get("source", "cache"),
        "model": MODEL,
        "n": len(questions),
        "n_train": len(train),
        "n_test": len(test),
        "test_ids": [str(item["id"]) for item in test],
        "train_ids": [str(item["id"]) for item in train],
        "budget": b,
        "seed": seed,
        "k": V4_K,
        "token_levels": list(V4_TOKEN_LEVELS),
        "note": (
            "predicted policies are scored on realized k=3 envelopes; "
            "DP is exact on predictions, not on the next API call"
        ),
        "predicted_dp_beats_uniform_64": beat_uniform,
        "rows": [
            policy_row("oracle_dp", oracle_dp.total_value, oracle_dp.total_cost),
            policy_row("oracle_greedy", oracle_greedy.total_value, oracle_greedy.total_cost),
            policy_row("oracle_random", oracle_random.total_value, oracle_random.total_cost),
            policy_row("predicted_dp", pred_dp_real, pred_dp_cost),
            policy_row("predicted_greedy", pred_greedy_real, pred_greedy_cost),
            policy_row("uniform_64", cheap.total_value, cheap.total_cost),
            policy_row("uniform_256", long.total_value, long.total_cost),
        ],
        "roundtrip": roundtrip,
    }


def save_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def plot_v4(payload: dict[str, Any], path: Path = CHART_PATH) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["policy"] for row in payload["rows"]]
    values = [row["realized_value"] for row in payload["rows"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.bar(names, values)
    ax.set_title("v4 realized test score by policy")
    ax.set_ylabel("Realized mean accuracy (sum)")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    fig.text(
        0.01,
        0.01,
        f"n_test={payload.get('n_test')} · budget {payload.get('budget')} · "
        f"k={payload.get('k')} · predicted-DP vs uniform-64: "
        f"{'beats' if payload.get('predicted_dp_beats_uniform_64') else 'does not beat'}",
        fontsize=8,
        color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v4 predict-then-allocate replay (offline)")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument("--roundtrip", type=Path, default=ROUNDTRIP_PATH)
    parser.add_argument("--chart", type=Path, default=CHART_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    questions = load_questions(args.questions)
    cache = load_cache(args.cache)
    payload = replay_v4(questions, cache, budget=args.budget)
    saved = save_json(payload, args.out)
    save_json({"test_ids": payload["test_ids"], "rows": payload["roundtrip"]}, args.roundtrip)
    chart = plot_v4(payload, args.chart)
    print(f"wrote {saved}")
    print(f"wrote {args.roundtrip}")
    print(f"wrote {chart}")
    for row in payload["rows"]:
        print(
            f"{row['policy']:<18} realized={row['realized_value']:.4f}  "
            f"cost={row['cost_units']}"
        )
    beat = "yes" if payload["predicted_dp_beats_uniform_64"] else "no"
    print(f"predicted_dp beats uniform_64: {beat}")


if __name__ == "__main__":
    main()
