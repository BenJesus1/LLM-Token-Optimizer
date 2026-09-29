"""Prompt-only Haiku vs quality router. No gold at decide time. No network."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .cache import lookup
from .config import (
    COST_HAIKU_CALL,
    COST_QUALITY_CALL,
    MODEL_CHEAP,
    MODEL_QUALITY,
    RANDOM_SEED,
    V5_LAMBDA,
    V5_MAX_TOKENS,
    V5_TRAIN_N,
)
from .predict import (
    FeatureVec,
    Predictor,
    predict_clip,
    ridge_fit,
    train_test_split,
)
from .score import completion_has_number

_WORD_CUES = (
    "how many",
    "how much",
    "yesterday",
    "altogether",
    "times as much",
    "half as many",
    "percent",
    "sale",
    "workers",
    "each person",
)


def route_features(question: str) -> FeatureVec:
    """Prompt-only features. Gold must not change this vector."""

    text = question.strip()
    lower = text.lower()
    words = text.split()
    cues = sum(1.0 for cue in _WORD_CUES if cue in lower)
    return (
        1.0,
        len(text) / 200.0,
        len(words) / 40.0,
        min(cues, 3.0) / 3.0,
        1.0 if len(words) > 20 else 0.0,
    )


def cache_score(
    cache: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    model: str,
    token_level: int = V5_MAX_TOKENS,
) -> float:
    entry = lookup(
        dict(cache),
        question_id=str(item["id"]),
        token_level=token_level,
        model=model,
        question=str(item["question"]),
        sample=0,
    )
    if entry is None:
        return 0.0
    return float(entry["score"])


def cache_completion(
    cache: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    model: str,
    token_level: int = V5_MAX_TOKENS,
) -> str:
    entry = lookup(
        dict(cache),
        question_id=str(item["id"]),
        token_level=token_level,
        model=model,
        question=str(item["question"]),
        sample=0,
    )
    if entry is None:
        return ""
    return str(entry.get("completion") or "")


def fit_haiku_predictor(
    train_questions: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    *,
    model: str = MODEL_CHEAP,
) -> Predictor:
    rows = [route_features(str(item["question"])) for item in train_questions]
    targets = [cache_score(cache, item, model=model) for item in train_questions]
    fitted = Predictor({(model, V5_MAX_TOKENS): ridge_fit(rows, targets, lam=1.0)})
    fitted.train_ids = frozenset(str(item["id"]) for item in train_questions)
    return fitted


def p_haiku_ok(question: str, predictor: Predictor, *, model: str = MODEL_CHEAP) -> float:
    weights = predictor.weights_by_key.get((model, V5_MAX_TOKENS))
    if weights is None:
        return 0.0
    return predict_clip(weights, route_features(question))


@dataclass(frozen=True)
class RouteDecision:
    model: str
    p_haiku: float
    tau: float
    reason: str


def choose_model(
    question: str,
    predictor: Predictor,
    *,
    tau: float,
    cheap: str = MODEL_CHEAP,
    quality: str = MODEL_QUALITY,
) -> RouteDecision:
    p_hat = p_haiku_ok(question, predictor, model=cheap)
    if p_hat >= tau:
        return RouteDecision(
            model=cheap,
            p_haiku=p_hat,
            tau=tau,
            reason=f"p_haiku={p_hat:.3f} >= tau={tau:.3f}",
        )
    return RouteDecision(
        model=quality,
        p_haiku=p_hat,
        tau=tau,
        reason=f"p_haiku={p_hat:.3f} < tau={tau:.3f}",
    )


def _train_utility(
    train: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    predictor: Predictor,
    tau: float,
    *,
    lam: float,
) -> float:
    total_score = 0.0
    total_cost = 0.0
    for item in train:
        decision = choose_model(str(item["question"]), predictor, tau=tau)
        total_score += cache_score(cache, item, model=decision.model)
        total_cost += (
            COST_HAIKU_CALL if decision.model == MODEL_CHEAP else COST_QUALITY_CALL
        )
    return total_score - lam * total_cost


def tune_tau(
    train: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    predictor: Predictor,
    *,
    lam: float = V5_LAMBDA,
) -> float:
    """Grid search on train only."""

    best_tau = 0.5
    best_util = float("-inf")
    step = 0
    while step <= 10:
        tau = step / 10.0
        util = _train_utility(train, cache, predictor, tau, lam=lam)
        if util > best_util:
            best_util = util
            best_tau = tau
        step += 1
    return best_tau


def fit_router(
    questions: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    *,
    seed: int = RANDOM_SEED,
    n_train: int = V5_TRAIN_N,
) -> tuple[Predictor, float, list[dict[str, Any]], list[dict[str, Any]]]:
    train, test = train_test_split(questions, seed=seed, n_train=n_train)
    predictor = fit_haiku_predictor(train, cache)
    tau = tune_tau(train, cache, predictor)
    return predictor, tau, train, test


def heuristic_uses_quality(item: Mapping[str, Any], cache: Mapping[str, Any]) -> bool:
    """Escalate without gold: Haiku text has no number."""

    text = cache_completion(cache, item, model=MODEL_CHEAP)
    return not completion_has_number(text)
