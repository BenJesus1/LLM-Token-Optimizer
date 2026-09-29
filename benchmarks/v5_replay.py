"""Offline v5 policy table: always-Haiku / always-quality / router / cascades."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from allocator.live.cache import load_cache
from allocator.live.collect import load_questions
from allocator.live.config import (
    COST_HAIKU_CALL,
    COST_QUALITY_CALL,
    MODEL_CHEAP,
    MODEL_QUALITY,
    RANDOM_SEED,
    V5_TRAIN_N,
)
from allocator.live.predict import assert_no_label_leak
from allocator.live.router import (
    cache_score,
    choose_model,
    fit_router,
    heuristic_uses_quality,
)

LIVE_DIR = Path(__file__).resolve().parent / "live"
DEFAULT_QUESTIONS = LIVE_DIR / "questions_v5.json"
DEFAULT_CACHE = LIVE_DIR / "fixture_cache_v5.json"
RESULTS_PATH = LIVE_DIR / "v5_results.json"
CHART_PATH = Path(__file__).resolve().parent / "charts" / "v5_router_vs_baselines.png"


def _row(name: str, score: float, cost: int, n: int) -> dict[str, Any]:
    efficiency = None if cost == 0 else score / float(cost)
    return {
        "policy": name,
        "realized_score": score,
        "cost_units": cost,
        "n_test": n,
        "score_per_cost": efficiency,
    }


def eval_always(
    test: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    *,
    model: str,
    cost_each: int,
) -> tuple[float, int]:
    score = sum(cache_score(cache, item, model=model) for item in test)
    return score, cost_each * len(test)


def eval_router(
    test: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    predictor: Any,
    tau: float,
) -> tuple[float, int]:
    score = 0.0
    cost = 0
    for item in test:
        decision = choose_model(str(item["question"]), predictor, tau=tau)
        score += cache_score(cache, item, model=decision.model)
        cost += COST_HAIKU_CALL if decision.model == MODEL_CHEAP else COST_QUALITY_CALL
    return score, cost


def eval_heuristic_cascade(
    test: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
) -> tuple[float, int]:
    score = 0.0
    cost = 0
    for item in test:
        if heuristic_uses_quality(item, cache):
            score += cache_score(cache, item, model=MODEL_QUALITY)
            cost += COST_HAIKU_CALL + COST_QUALITY_CALL
        else:
            score += cache_score(cache, item, model=MODEL_CHEAP)
            cost += COST_HAIKU_CALL
    return score, cost


def eval_oracle_cascade(
    test: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
) -> tuple[float, int]:
    score = 0.0
    cost = 0
    for item in test:
        haiku = cache_score(cache, item, model=MODEL_CHEAP)
        if haiku >= 1.0:
            score += haiku
            cost += COST_HAIKU_CALL
        else:
            score += cache_score(cache, item, model=MODEL_QUALITY)
            cost += COST_HAIKU_CALL + COST_QUALITY_CALL
    return score, cost


def replay_v5(
    questions: Sequence[Mapping[str, Any]],
    cache: dict[str, Any],
    *,
    seed: int = RANDOM_SEED,
    n_train: int = V5_TRAIN_N,
) -> dict[str, Any]:
    predictor, tau, train, test = fit_router(questions, cache, seed=seed, n_train=n_train)
    assert_no_label_leak(predictor, test)
    n = len(test)
    always_h = eval_always(test, cache, model=MODEL_CHEAP, cost_each=COST_HAIKU_CALL)
    always_q = eval_always(test, cache, model=MODEL_QUALITY, cost_each=COST_QUALITY_CALL)
    routed = eval_router(test, cache, predictor, tau)
    heur = eval_heuristic_cascade(test, cache)
    oracle = eval_oracle_cascade(test, cache)
    beat_score = routed[0] > always_h[0] + 1e-12
    beat_eff = False
    if routed[1] and always_h[1]:
        beat_eff = routed[0] / routed[1] > always_h[0] / always_h[1] + 1e-12
    return {
        "source": cache.get("source", "cache"),
        "cheap_model": MODEL_CHEAP,
        "quality_model": MODEL_QUALITY,
        "n": len(questions),
        "n_train": len(train),
        "n_test": n,
        "tau": tau,
        "train_ids": [str(item["id"]) for item in train],
        "test_ids": [str(item["id"]) for item in test],
        "seed": seed,
        "note": (
            "router uses prompt-only features; oracle_cascade uses Haiku gold scores "
            "and is not a deployable policy"
        ),
        "router_beats_always_haiku_score": beat_score,
        "router_beats_always_haiku_efficiency": beat_eff,
        "rows": [
            _row("always_haiku", always_h[0], always_h[1], n),
            _row("always_quality", always_q[0], always_q[1], n),
            _row("router", routed[0], routed[1], n),
            _row("heuristic_cascade", heur[0], heur[1], n),
            _row("oracle_cascade", oracle[0], oracle[1], n),
        ],
    }


def save_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def plot_v5(payload: dict[str, Any], path: Path = CHART_PATH) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["policy"] for row in payload["rows"]]
    scores = [row["realized_score"] for row in payload["rows"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.bar(names, scores)
    ax.set_title("v5 realized test score by routing policy")
    ax.set_ylabel("Exact-match sum")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    fig.text(
        0.01,
        0.01,
        f"n_test={payload.get('n_test')} · tau={payload.get('tau')} · "
        f"router beats always-Haiku (score): {payload.get('router_beats_always_haiku_score')}",
        fontsize=8,
        color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="v5 router replay (offline)")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument("--chart", type=Path, default=CHART_PATH)
    args = parser.parse_args(argv)
    questions = load_questions(args.questions)
    cache = load_cache(args.cache)
    payload = replay_v5(questions, cache)
    save_json(payload, args.out)
    plot_v5(payload, args.chart)
    print(f"wrote {args.out}")
    print(f"wrote {args.chart}")
    print(f"tau={payload['tau']}")
    for row in payload["rows"]:
        eff = row["score_per_cost"]
        eff_s = "na" if eff is None else f"{eff:.4f}"
        print(
            f"{row['policy']:<20} score={row['realized_score']:.4f}  "
            f"cost={row['cost_units']}  eff={eff_s}"
        )
    print(
        "router beats always_haiku (score): "
        f"{payload['router_beats_always_haiku_score']}"
    )
    print(
        "router beats always_haiku (efficiency): "
        f"{payload['router_beats_always_haiku_efficiency']}"
    )


if __name__ == "__main__":
    main()
