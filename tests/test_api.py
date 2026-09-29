"""API-layer tests. Does not modify v1 solver tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport

from allocator import allocate_dp, allocate_greedy
from allocator.api.app import app
from allocator.api.models import AllocateRequest, TaskIn
from allocator.api.service import MAX_BUDGET, MAX_TASKS, MAX_TOKEN_COST, run_allocate
from allocator.task import Task

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_JSON = REPO_ROOT / "tasks.json"
client = TestClient(app)

SAMPLE_TWO_LOG = {
    "budget": 3,
    "tasks": [
        {"id": "a", "token_cost": 3, "weight": 1.0, "curve": "log"},
        {"id": "b", "token_cost": 3, "weight": 1.0, "curve": "log"},
    ],
}


def _sample_tasks_payload() -> dict:
    return {"budget": 20, "tasks": json.loads(TASKS_JSON.read_text(encoding="utf-8"))}


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_sample_tasks_json_matches_library() -> None:
    payload = _sample_tasks_payload()
    tasks = tuple(Task(**item) for item in payload["tasks"])
    library_dp = allocate_dp(tasks, payload["budget"])
    library_greedy = allocate_greedy(tasks, payload["budget"])

    response = client.post("/allocate", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert [row["id"] for row in body["assignments"]] == [
        item.task.id for item in library_dp.assignments
    ]
    assert [row["tokens"] for row in body["assignments"]] == [
        item.tokens for item in library_dp.assignments
    ]
    assert body["total_value"] == pytest.approx(library_dp.total_value)
    assert body["total_cost"] == library_dp.total_cost
    assert body["greedy"]["total_value"] == pytest.approx(library_greedy.total_value)
    assert body["greedy"]["total_cost"] == library_greedy.total_cost
    assert body["solvers"] == ["allocate_dp", "allocate_greedy"]
    assert "knapsack_greedy" not in json.dumps(body)


def test_hand_computed_spread_beats_dump_through_http() -> None:
    response = client.post("/allocate", json=SAMPLE_TWO_LOG)

    assert response.status_code == 200
    by_id = {row["id"]: row["tokens"] for row in response.json()["assignments"]}
    assert by_id["a"] == 2
    assert by_id["b"] == 1


def test_missing_budget_is_422() -> None:
    response = client.post(
        "/allocate",
        json={"tasks": [{"id": "a", "token_cost": 1, "weight": 1.0}]},
    )

    assert response.status_code == 422
    assert "detail" in response.json()


def test_negative_budget_is_400() -> None:
    response = client.post(
        "/allocate",
        json={
            "budget": -1,
            "tasks": [{"id": "a", "token_cost": 1, "weight": 1.0, "curve": "log"}],
        },
    )

    assert response.status_code == 400
    assert "budget" in response.json()["detail"]
    assert "Traceback" not in response.text


def test_empty_task_list_is_400() -> None:
    response = client.post("/allocate", json={"budget": 4, "tasks": []})

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_unknown_curve_is_400() -> None:
    response = client.post(
        "/allocate",
        json={
            "budget": 4,
            "tasks": [{"id": "a", "token_cost": 1, "weight": 1.0, "curve": "bandit"}],
        },
    )

    assert response.status_code == 400
    assert "curve" in response.json()["detail"]


def test_oversized_task_list_is_400() -> None:
    tasks = [
        {"id": f"t{i}", "token_cost": 1, "weight": 1.0, "curve": "log"}
        for i in range(MAX_TASKS + 1)
    ]
    response = client.post("/allocate", json={"budget": 1, "tasks": tasks})

    assert response.status_code == 400
    assert str(MAX_TASKS) in response.json()["detail"]


def test_oversized_budget_is_400() -> None:
    response = client.post(
        "/allocate",
        json={
            "budget": MAX_BUDGET + 1,
            "tasks": [{"id": "a", "token_cost": 1, "weight": 1.0, "curve": "log"}],
        },
    )

    assert response.status_code == 400
    assert str(MAX_BUDGET) in response.json()["detail"]


def test_oversized_token_cost_is_400() -> None:
    response = client.post(
        "/allocate",
        json={
            "budget": 1,
            "tasks": [
                {
                    "id": "a",
                    "token_cost": MAX_TOKEN_COST + 1,
                    "weight": 1.0,
                    "curve": "log",
                }
            ],
        },
    )

    assert response.status_code == 400
    assert str(MAX_TOKEN_COST) in response.json()["detail"]


def test_run_allocate_matches_library_on_sample() -> None:
    payload = _sample_tasks_payload()
    request = AllocateRequest(
        budget=payload["budget"],
        tasks=[TaskIn(**item) for item in payload["tasks"]],
    )
    tasks = tuple(Task(**item) for item in payload["tasks"])

    result = run_allocate(request)
    library = allocate_dp(tasks, payload["budget"])

    assert result.total_value == pytest.approx(library.total_value)
    assert result.total_cost == library.total_cost


def test_concurrent_requests_return_identical_allocations() -> None:
    """Solvers are pure: ten overlapping calls must agree with a serial call."""

    payload = _sample_tasks_payload()

    async def hammer() -> list[dict]:
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            async def once() -> dict:
                response = await http.post("/allocate", json=payload)
                assert response.status_code == 200
                body = response.json()
                body.pop("dp_runtime_s")
                return body

            return list(await asyncio.gather(*[once() for _ in range(10)]))

    rows = asyncio.run(hammer())
    serial = client.post("/allocate", json=payload).json()
    serial.pop("dp_runtime_s")
    assert all(row == serial for row in rows)
