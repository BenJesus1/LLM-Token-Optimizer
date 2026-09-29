"""Tiny ridge predictor from question features. No sklearn. Train labels only."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from .config import MODEL, RANDOM_SEED, V4_TOKEN_LEVELS, V4_TRAIN_N
from .score import normalize

FeatureVec = tuple[float, ...]
Weights = list[float]


def is_numeric_gold(gold: str) -> bool:
    text = normalize(gold)
    if text.startswith("-"):
        text = text[1:]
    return bool(text) and text.replace(".", "", 1).isdigit()


def features(item: Mapping[str, Any]) -> FeatureVec:
    """Intercept plus scaled length, word count, numeric gold, leetcode flag."""

    question = str(item.get("question") or "")
    gold = str(item.get("gold") or "")
    words = question.split()
    return (
        1.0,
        len(question) / 100.0,
        len(words) / 20.0,
        1.0 if is_numeric_gold(gold) else 0.0,
        1.0 if "leetcode" in question.lower() else 0.0,
    )


def train_test_split(
    questions: Sequence[Mapping[str, Any]],
    *,
    seed: int = RANDOM_SEED,
    n_train: int = V4_TRAIN_N,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shuffle with ``seed`` then cut. Remainder is the test split (expect 4 when n=10)."""

    shuffled = [dict(item) for item in questions]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if n_train < 0 or n_train > len(shuffled):
        raise ValueError("n_train must be between 0 and n inclusive")
    return shuffled[:n_train], shuffled[n_train:]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivot. ``matrix`` is mutated."""

    n = len(rhs)
    for i in range(n):
        pivot = i
        for r in range(i + 1, n):
            if abs(matrix[r][i]) > abs(matrix[pivot][i]):
                pivot = r
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        rhs[i], rhs[pivot] = rhs[pivot], rhs[i]
        diag = matrix[i][i]
        if abs(diag) < 1e-12:
            matrix[i][i] = 1e-12
            diag = matrix[i][i]
        scale = 1.0 / diag
        for c in range(i, n):
            matrix[i][c] *= scale
        rhs[i] *= scale
        for r in range(n):
            if r == i:
                continue
            factor = matrix[r][i]
            for c in range(i, n):
                matrix[r][c] -= factor * matrix[i][c]
            rhs[r] -= factor * rhs[i]
    return rhs


def ridge_fit(
    rows: Sequence[FeatureVec],
    targets: Sequence[float],
    *,
    lam: float = 1.0,
) -> Weights:
    """(X'X + λI)^{-1} X'y. ``lam`` keeps the 5-parameter fit stable on n=6."""

    if len(rows) != len(targets):
        raise ValueError("rows and targets must have the same length")
    if not rows:
        raise ValueError("ridge_fit needs at least one row")
    dim = len(rows[0])
    gram = [[0.0] * dim for _ in range(dim)]
    xt_y = [0.0] * dim
    for vec, target in zip(rows, targets, strict=True):
        for i in range(dim):
            xt_y[i] += vec[i] * target
            for j in range(dim):
                gram[i][j] += vec[i] * vec[j]
    for i in range(dim):
        gram[i][i] += lam
    return _solve(gram, xt_y)


def predict_clip(weights: Weights, vec: FeatureVec) -> float:
    raw = _dot(weights, vec)
    if raw < 0.0:
        return 0.0
    if raw > 1.0:
        return 1.0
    return raw


class Predictor:
    """One weight vector per (model, token_level). Fitted on train ids only."""

    def __init__(self, weights_by_key: dict[tuple[str, int], Weights]) -> None:
        self.weights_by_key = weights_by_key
        self.train_ids: frozenset[str] = frozenset()

    def predict_mean(self, item: Mapping[str, Any], *, model: str, token_level: int) -> float:
        key = (model, int(token_level))
        weights = self.weights_by_key.get(key)
        if weights is None:
            return 0.0
        return predict_clip(weights, features(item))


def fit_predictor(
    train_questions: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    *,
    model: str = MODEL,
    levels: Sequence[int] = V4_TOKEN_LEVELS,
    k: int = 3,
    lam: float = 1.0,
) -> Predictor:
    """Fit using train questions' realized means. Test ids must not appear here."""

    from .actions import mean_score

    train_ids = {str(item["id"]) for item in train_questions}
    weights_by_key: dict[tuple[str, int], Weights] = {}
    rows = [features(item) for item in train_questions]
    for level in levels:
        targets = [
            mean_score(
                cache,
                question_id=str(item["id"]),
                question=str(item["question"]),
                model=model,
                token_level=int(level),
                k=k,
            )
            for item in train_questions
        ]
        weights_by_key[(model, int(level))] = ridge_fit(rows, targets, lam=lam)
    fitted = Predictor(weights_by_key)
    fitted.train_ids = frozenset(train_ids)
    return fitted


def predicted_points(
    item: Mapping[str, Any],
    predictor: Predictor,
    *,
    model: str = MODEL,
    levels: Sequence[int] = V4_TOKEN_LEVELS,
    cost_of_level: Mapping[int, int] | None = None,
) -> list[tuple[int, float]]:
    """``(cost_units, predicted_mean)`` for each v4 level."""

    from .config import COST_HAIKU_256, COST_HAIKU_64

    costs = dict(cost_of_level) if cost_of_level is not None else {
        64: COST_HAIKU_64,
        256: COST_HAIKU_256,
    }
    return [
        (int(costs[int(level)]), predictor.predict_mean(item, model=model, token_level=int(level)))
        for level in levels
    ]


def assert_no_label_leak(predictor: Predictor, test_questions: Sequence[Mapping[str, Any]]) -> None:
    """Raise if any test id was in the train set used to fit ``predictor``."""

    leaked = [str(item["id"]) for item in test_questions if str(item["id"]) in predictor.train_ids]
    if leaked:
        raise ValueError(f"predictor trained on test ids: {leaked}")


def logistic(z: float) -> float:
    """Unused helper kept for a possible alternate head; ridge is the default."""

    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)
