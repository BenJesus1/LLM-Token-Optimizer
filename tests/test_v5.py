"""v5 tests. Never read ANTHROPIC_API_KEY or open the network."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from allocator.api.app import app, create_app
from allocator.live.cache import empty_cache
from allocator.live.collect_v5 import collect_two_models
from allocator.live.config import (
    MODEL_CHEAP,
    MODEL_QUALITY,
    V5_CACHE_ENV,
    V5_LOCAL_ENV,
    V5_QUESTIONS_ENV,
    V5_TEST_N,
    V5_TRAIN_N,
)
from allocator.live.predict import assert_no_label_leak, features as v4_features, train_test_split
from allocator.live.router import fit_haiku_predictor, fit_router, p_haiku_ok, route_features
from benchmarks.v5_replay import replay_v5

REPO = Path(__file__).resolve().parents[1]
QUESTIONS_V5 = REPO / "benchmarks" / "live" / "questions_v5.json"
FIXTURE_V5 = REPO / "benchmarks" / "live" / "fixture_cache_v5.json"
RESULTS_V5 = REPO / "benchmarks" / "live" / "v5_results.json"


def _load_json(path: Path) -> object:
    return __import__("json").loads(path.read_text(encoding="utf-8"))


def test_questions_v5_are_twenty_numeric() -> None:
    questions = _load_json(QUESTIONS_V5)
    assert isinstance(questions, list)
    assert len(questions) == 20
    ids = [item["id"] for item in questions]
    assert len(set(ids)) == 20
    for item in questions:
        gold = str(item["gold"]).strip()
        assert gold.lstrip("-").replace(".", "", 1).isdigit()


def test_gold_is_not_in_route_features() -> None:
    question = "What is 7 + 5?"
    numeric = {"id": "x", "question": question, "gold": "12"}
    non_numeric = {"id": "x", "question": question, "gold": "not-a-number"}
    assert route_features(question) == route_features(str(non_numeric["question"]))
    assert v4_features(numeric) != v4_features(non_numeric)
    questions = _load_json(QUESTIONS_V5)
    cache = _load_json(FIXTURE_V5)
    predictor, _tau, _train, _test = fit_router(questions, cache)
    assert p_haiku_ok(question, predictor) == p_haiku_ok(
        str(non_numeric["question"]), predictor
    )


def test_router_does_not_fit_on_test_ids() -> None:
    questions = _load_json(QUESTIONS_V5)
    cache = _load_json(FIXTURE_V5)
    train, test = train_test_split(questions, n_train=V5_TRAIN_N)
    assert len(train) == V5_TRAIN_N
    assert len(test) == V5_TEST_N
    predictor = fit_haiku_predictor(train, cache)
    assert_no_label_leak(predictor, test)
    leaked = predictor.train_ids | {str(test[0]["id"])}
    predictor.train_ids = frozenset(leaked)
    with pytest.raises(ValueError, match="test ids"):
        assert_no_label_leak(predictor, test)


def test_collect_two_models_stub_then_cache_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    questions = [
        {"id": "q01", "question": "What is 1+1?", "gold": "2"},
        {"id": "q02", "question": "What is 2+2?", "gold": "4"},
    ]
    calls = {"n": 0}

    def stub_for(_model: str):
        def stub(*, prompt: str, max_tokens: int) -> tuple[str, dict[str, int]]:
            del prompt, max_tokens
            calls["n"] += 1
            return "Answer: 2", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

        return stub

    cache = empty_cache()
    completers = {
        MODEL_CHEAP: stub_for(MODEL_CHEAP),
        MODEL_QUALITY: stub_for(MODEL_QUALITY),
    }
    collect_two_models(questions, cache, live=True, completers=completers)
    assert calls["n"] == 4
    models = {entry["model"] for entry in cache["entries"]}
    assert models == {MODEL_CHEAP, MODEL_QUALITY}
    collect_two_models(questions, cache, live=True, completers=completers)
    assert calls["n"] == 4


def test_fixture_router_runs_offline() -> None:
    questions = _load_json(QUESTIONS_V5)
    cache = _load_json(FIXTURE_V5)
    payload = replay_v5(questions, cache)
    by_name = {row["policy"]: row for row in payload["rows"]}
    assert set(by_name) == {
        "always_haiku",
        "always_quality",
        "router",
        "heuristic_cascade",
        "oracle_cascade",
    }
    assert payload["n_test"] == V5_TEST_N
    assert by_name["always_quality"]["realized_score"] >= by_name["always_haiku"]["realized_score"]


def test_route_absent_by_default() -> None:
    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/route" not in paths
    client = TestClient(app)
    response = client.post("/route", json={"question": "What is 7 + 5?"})
    assert response.status_code == 404


def test_route_local_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(V5_LOCAL_ENV, "1")
    monkeypatch.setenv(V5_CACHE_ENV, str(FIXTURE_V5))
    monkeypatch.setenv(V5_QUESTIONS_ENV, str(QUESTIONS_V5))
    local_app = create_app()
    client = TestClient(local_app)
    response = client.post("/route", json={"question": "What is 7 + 5?"})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] in {MODEL_CHEAP, MODEL_QUALITY}
    assert "p_haiku" in body and "tau" in body and "reason" in body


def test_committed_v5_artifacts_exist() -> None:
    assert QUESTIONS_V5.is_file() and QUESTIONS_V5.stat().st_size > 0
    assert FIXTURE_V5.is_file() and FIXTURE_V5.stat().st_size > 0
    assert RESULTS_V5.is_file() and RESULTS_V5.stat().st_size > 0
    chart = REPO / "benchmarks" / "charts" / "v5_router_vs_baselines.png"
    assert chart.is_file() and chart.stat().st_size > 0
    fixture = _load_json(FIXTURE_V5)
    assert len(fixture["entries"]) == 40
    models = {entry["model"] for entry in fixture["entries"]}
    assert models == {MODEL_CHEAP, MODEL_QUALITY}
