"""Tests for benchmark plots. Reads saved results; does not run allocate_dp at n=1000."""

from pathlib import Path

from benchmarks.measure import load_results
from benchmarks.plot import plot_runtime, plot_value_gap


def test_runtime_plot_writes_a_png(tmp_path: Path) -> None:
    payload = load_results()
    path = tmp_path / "runtime.png"

    written = plot_runtime(payload, path=path)

    assert written == path
    assert path.is_file()
    assert path.stat().st_size > 0


def test_value_gap_plot_writes_a_png(tmp_path: Path) -> None:
    payload = load_results()
    path = tmp_path / "gap.png"

    written = plot_value_gap(payload, path=path)

    assert written == path
    assert path.is_file()
    assert path.stat().st_size > 0
