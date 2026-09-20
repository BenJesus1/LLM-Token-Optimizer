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
