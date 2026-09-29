"""v4 tests. Never read ANTHROPIC_API_KEY or open the network."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from allocator import allocate_dp, allocate_greedy
from allocator.api.app import app, create_app
from allocator.live.actions import envelope_table, mean_score, tasks_from_v4_cache
from allocator.live.cache import empty_cache, prompt_hash
from allocator.live.collect import collect
from allocator.live.config import MODEL, V4_CACHE_ENV, V4_LOCAL_ENV, V4_QUESTIONS_ENV
from allocator.live.predict import (
    assert_no_label_leak,
    fit_predictor,
    train_test_split,
)
from benchmarks.v4_replay import replay_v4

REPO = Path(__file__).resolve().parents[1]
QUESTIONS_V4 = REPO / "benchmarks" / "live" / "questions_v4.json"
FIXTURE_V4 = REPO / "benchmarks" / "live" / "fixture_cache_v4.json"
RESULTS_V4 = REPO / "benchmarks" / "live" / "v4_results.json"


def _load_json(path: Path) -> object:
    return __import__("json").loads(path.read_text(encoding="utf-8"))


def test_envelope_cumulative_max_drops_dominated() -> None:
    assert envelope_table(((1, 0.4), (4, 0.2))) == ((1, 0.4),)
    assert envelope_table(((1, 0.2), (4, 0.9))) == ((1, 0.2), (4, 0.9))
    assert envelope_table(((1, 0.5), (4, 0.5))) == ((1, 0.5),)


def test_k3_mean_score_averages_samples() -> None:
    question = "What is 1+1?"
    cache = empty_cache()
    cache["entries"] = [
        {
            "question_id": "q01",
            "token_level": 64,
            "model": MODEL,
            "prompt_hash": prompt_hash(question),
            "sample": sample,
            "completion": "2" if sample != 1 else "nope",
            "score": 1.0 if sample != 1 else 0.0,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        for sample in range(3)
    ]
    mean = mean_score(
        cache,
        question_id="q01",
        question=question,
        model=MODEL,
        token_level=64,
        k=3,
    )
    assert mean == pytest.approx(2.0 / 3.0)


def test_collect_k3_stub_then_cache_hit() -> None:
    questions = [{"id": "q01", "question": "What is 1+1?", "gold": "2"}]
    calls = {"n": 0}

    def stub(*, prompt: str, max_tokens: int) -> tuple[str, dict[str, int]]:
        del prompt, max_tokens
        calls["n"] += 1
        return "2", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    cache = empty_cache()
    collect(questions, cache, live=True, completer=stub, levels=(64,), k=3, max_calls=150)
    assert calls["n"] == 3
    collect(questions, cache, live=True, completer=stub, levels=(64,), k=3, max_calls=150)
    assert calls["n"] == 3


def test_predictor_does_not_fit_on_test_ids() -> None:
    questions = _load_json(QUESTIONS_V4)
    cache = _load_json(FIXTURE_V4)
    train, test = train_test_split(questions)
    assert len(train) == 6
    assert len(test) == 4
    predictor = fit_predictor(train, cache)
    assert_no_label_leak(predictor, test)
    leaked = predictor.train_ids | {str(test[0]["id"])}
    predictor.train_ids = frozenset(leaked)
    with pytest.raises(ValueError, match="test ids"):
        assert_no_label_leak(predictor, test)


def test_v4_fixture_oracle_dp_at_least_greedy() -> None:
    questions = _load_json(QUESTIONS_V4)
    cache = _load_json(FIXTURE_V4)
    payload = replay_v4(questions, cache)
    by_name = {row["policy"]: row for row in payload["rows"]}
    assert by_name["oracle_dp"]["realized_value"] >= by_name["oracle_greedy"]["realized_value"]
    test_tasks = tasks_from_v4_cache(questions[:2], cache)
    budget = 4
    exact = allocate_dp(test_tasks, budget)
    greedy = allocate_greedy(test_tasks, budget)
    assert exact.total_value >= greedy.total_value


def test_allocate_predicted_absent_by_default() -> None:
    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/allocate-predicted" not in paths
    client = TestClient(app)
    response = client.post(
        "/allocate-predicted",
        json={"budget": 4, "questions": [{"id": "v01", "question": "What is 7 + 5?", "gold": "12"}]},
    )
    assert response.status_code == 404


def test_allocate_predicted_local_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(V4_LOCAL_ENV, "1")
    monkeypatch.setenv(V4_CACHE_ENV, str(FIXTURE_V4))
    monkeypatch.setenv(V4_QUESTIONS_ENV, str(QUESTIONS_V4))
    local_app = create_app()
    client = TestClient(local_app)
    response = client.post(
        "/allocate-predicted",
        json={
            "budget": 4,
            "questions": [
                {"id": "v01", "question": "What is 7 + 5?", "gold": "12"},
                {"id": "v06", "question": "LeetCode 70 Climbing Stairs. n = 8.", "gold": "34"},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["solvers"] == ["allocate_dp", "allocate_greedy"]
    assert "greedy" in body
    assert body["total_cost"] <= 4


def test_hard_questions_are_ten_numeric() -> None:
    path = REPO / "benchmarks" / "live" / "questions_v4_hard.json"
    questions = _load_json(path)
    assert len(questions) == 10
    ids = [item["id"] for item in questions]
    assert len(set(ids)) == 10
    for item in questions:
        gold = str(item["gold"]).strip()
        assert gold.lstrip("-").replace(".", "", 1).isdigit()


def test_committed_v4_artifacts_exist() -> None:
    assert QUESTIONS_V4.is_file() and QUESTIONS_V4.stat().st_size > 0
    assert FIXTURE_V4.is_file() and FIXTURE_V4.stat().st_size > 0
    assert RESULTS_V4.is_file() and RESULTS_V4.stat().st_size > 0
    chart = REPO / "benchmarks" / "charts" / "v4_realized_score_vs_policy.png"
    assert chart.is_file() and chart.stat().st_size > 0
