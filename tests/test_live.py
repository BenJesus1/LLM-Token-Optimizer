"""Live-eval tests. Never read ANTHROPIC_API_KEY or open the network."""

from __future__ import annotations

from pathlib import Path

import pytest

from allocator import allocate_dp, allocate_greedy, allocate_random
from allocator.live.cache import empty_cache, prompt_hash
from allocator.live.collect import collect
from allocator.live.config import MODEL, TOKEN_LEVELS
from allocator.live.score import score_answer
from allocator.live.tasks import tasks_from_cache
from allocator.task import Task
from benchmarks.live_plot import plot_live_scores
from benchmarks.live_replay import replay

FIXTURE_CACHE = Path(__file__).resolve().parents[1] / "benchmarks" / "live" / "fixture_cache.json"
FIXTURE_CHART = Path(__file__).resolve().parents[1] / "benchmarks" / "charts" / "live_score_vs_solver.png"
FIXTURE_RESULTS = Path(__file__).resolve().parents[1] / "benchmarks" / "live" / "results.json"


def test_score_numeric_uses_last_number() -> None:
    assert score_answer("The answer is 12.", "12") == 1.0
    assert score_answer("12 apples", "13") == 0.0


def test_score_string_is_normalized() -> None:
    assert score_answer("  Paris  ", "paris") == 1.0
    assert score_answer("london", "paris") == 0.0


def test_table_value_steps_at_ladder_levels() -> None:
    task = Task(
        id="a",
        token_cost=4,
        weight=0.0,
        curve="table",
        value_table=((2, 0.0), (4, 1.0)),
    )
    assert task.value_at(1) == 0.0
    assert task.value_at(2) == 0.0
    assert task.value_at(3) == 0.0
    assert task.value_at(4) == 1.0


def test_hand_table_greedy_jumps_and_dp_matches_value() -> None:
    # a scores only at 4; b scores at 2. Budget 4.
    a = Task(
        id="a",
        token_cost=4,
        weight=0.0,
        curve="table",
        value_table=((2, 0.0), (4, 1.0)),
    )
    b = Task(
        id="b",
        token_cost=4,
        weight=0.0,
        curve="table",
        value_table=((2, 1.0), (4, 1.0)),
    )
    tasks = [a, b]
    greedy = allocate_greedy(tasks, 4)
    exact = allocate_dp(tasks, 4)
    random_alloc = allocate_random(tasks, 4, seed=2026)

    assert greedy.total_cost <= 4
    assert exact.total_cost <= 4
    assert random_alloc.total_cost <= 4
    assert exact.total_value >= greedy.total_value
    assert greedy.tokens_for("b") >= 2
    assert greedy.tokens_for("a") == 0 or greedy.tokens_for("a") >= 4


def test_cache_hit_does_not_call_completer() -> None:
    questions = [{"id": "q01", "question": "What is 1+1?", "gold": "2"}]
    calls = {"n": 0}

    def stub(*, prompt: str, max_tokens: int) -> tuple[str, dict[str, int]]:
        calls["n"] += 1
        return "2", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    cache = empty_cache()
    collect(questions, cache, live=True, completer=stub, levels=(64, 128))
    first = calls["n"]
    assert first == 2
    collect(questions, cache, live=True, completer=stub, levels=(64, 128))
    assert calls["n"] == first


def test_incomplete_cache_without_live_is_rejected() -> None:
    questions = [{"id": "q01", "question": "What is 1+1?", "gold": "2"}]
    with pytest.raises(ValueError, match="--live"):
        collect(questions, empty_cache(), live=False, completer=None, levels=(64,))


def test_tasks_from_cache_use_table_curve() -> None:
    questions = [{"id": "q01", "question": "What is 1+1?", "gold": "2"}]
    prompt = "Answer with only the final answer. No explanation.\n\nWhat is 1+1?"
    cache = empty_cache()
    cache["entries"] = [
        {
            "question_id": "q01",
            "token_level": level,
            "model": MODEL,
            "prompt_hash": prompt_hash("What is 1+1?"),
            "completion": "2",
            "score": 1.0,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        for level in TOKEN_LEVELS
    ]
    tasks = tasks_from_cache(questions, cache)
    assert len(tasks) == 1
    assert tasks[0].curve == "table"
    assert tasks[0].value_at(64) == 1.0
    del prompt


def test_replay_fixture_cache_offline() -> None:
    questions = __import__("json").loads(
        (Path(__file__).resolve().parents[1] / "benchmarks" / "live" / "questions.json").read_text(
            encoding="utf-8"
        )
    )
    cache = __import__("json").loads(FIXTURE_CACHE.read_text(encoding="utf-8"))
    payload = replay(questions, cache)
    names = [row["solver"] for row in payload["rows"]]
    assert names == ["dp", "greedy", "random"]
    dp = next(row for row in payload["rows"] if row["solver"] == "dp")
    greedy = next(row for row in payload["rows"] if row["solver"] == "greedy")
    assert dp["total_value"] >= greedy["total_value"]


def test_committed_live_artifacts_exist() -> None:
    assert FIXTURE_CACHE.is_file() and FIXTURE_CACHE.stat().st_size > 0
    assert FIXTURE_RESULTS.is_file() and FIXTURE_RESULTS.stat().st_size > 0
    assert FIXTURE_CHART.is_file() and FIXTURE_CHART.stat().st_size > 0


def test_leetcode_questions_are_twenty_numeric() -> None:
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "live" / "questions_leetcode.json"
    questions = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert len(questions) == 20
    ids = [item["id"] for item in questions]
    assert len(set(ids)) == 20
    for item in questions:
        assert item["question"]
        gold = str(item["gold"]).strip()
        assert gold.replace("-", "", 1).replace(".", "", 1).isdigit()


def test_live_plot_writes_png(tmp_path: Path) -> None:
    payload = {
        "source": "test",
        "n": 2,
        "budget": 10,
        "rows": [
            {"solver": "dp", "total_value": 2},
            {"solver": "greedy", "total_value": 1},
            {"solver": "random", "total_value": 0},
        ],
    }
    path = tmp_path / "live.png"
    written = plot_live_scores(payload, path=path)
    assert written == path
    assert path.stat().st_size > 0
