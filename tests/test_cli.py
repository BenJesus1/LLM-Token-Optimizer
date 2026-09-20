"""Happy-path tests for ``allocator run``."""

from pathlib import Path

import pytest

from allocator.cli import main


def test_run_prints_dp_allocation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "tasks.json"
    path.write_text(
        '[{"id": "a", "token_cost": 4, "weight": 1.0, "curve": "log"}]',
        encoding="utf-8",
    )

    code = main(["run", "--budget", "4", "--tasks", str(path)])
    captured = capsys.readouterr()

    assert code == 0
    assert "Allocation (DP):" in captured.out
    assert "a" in captured.out
    assert "Total value (DP):" in captured.out
    assert "Runtime (DP):" in captured.out
    assert "Greedy:" in captured.out
    assert "token-level greedy" in captured.out
    assert "knapsack_greedy" not in captured.out
    assert captured.err == ""


def _assert_cli_error(capsys: pytest.CaptureFixture[str], code: int) -> str:
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
    return captured.err


def test_missing_task_file(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["run", "--budget", "4", "--tasks", "does-not-exist.json"])
    err = _assert_cli_error(capsys, code)
    assert "not found" in err


def test_bad_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "tasks.json"
    path.write_text("{not json", encoding="utf-8")

    code = main(["run", "--budget", "4", "--tasks", str(path)])
    err = _assert_cli_error(capsys, code)
    assert "JSON" in err


def test_negative_budget(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "tasks.json"
    path.write_text(
        '[{"id": "a", "token_cost": 4, "weight": 1.0, "curve": "log"}]',
        encoding="utf-8",
    )

    code = main(["run", "--budget", "-1", "--tasks", str(path)])
    err = _assert_cli_error(capsys, code)
    assert "budget" in err


def test_empty_task_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "tasks.json"
    path.write_text("[]", encoding="utf-8")

    code = main(["run", "--budget", "4", "--tasks", str(path)])
    err = _assert_cli_error(capsys, code)
    assert "empty" in err


def test_unknown_curve(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "tasks.json"
    path.write_text(
        '[{"id": "a", "token_cost": 4, "weight": 1.0, "curve": "bandit"}]',
        encoding="utf-8",
    )

    code = main(["run", "--budget", "4", "--tasks", str(path)])
    err = _assert_cli_error(capsys, code)
    assert "curve" in err
