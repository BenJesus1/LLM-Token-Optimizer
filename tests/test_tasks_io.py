"""Tests for loading Task lists from JSON and CSV."""

from pathlib import Path

from allocator.tasks_io import load_tasks


def test_load_json_tasks(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    path.write_text(
        '[{"id": "a", "token_cost": 4, "weight": 1.5, "curve": "log"}]',
        encoding="utf-8",
    )

    tasks = load_tasks(path)

    assert len(tasks) == 1
    assert tasks[0].id == "a"
    assert tasks[0].token_cost == 4
    assert tasks[0].weight == 1.5
    assert tasks[0].curve == "log"


def test_load_csv_tasks(tmp_path: Path) -> None:
    path = tmp_path / "tasks.csv"
    path.write_text(
        "id,token_cost,weight,curve\nb,3,2.0,sqrt\n",
        encoding="utf-8",
    )

    tasks = load_tasks(path)

    assert len(tasks) == 1
    assert tasks[0].id == "b"
    assert tasks[0].curve == "sqrt"


def test_sample_tasks_json_loads() -> None:
    tasks = load_tasks(Path("tasks.json"))

    assert [task.id for task in tasks] == [
        "summarize",
        "review",
        "translate",
        "classify",
    ]
