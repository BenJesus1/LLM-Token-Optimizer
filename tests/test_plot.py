"""Tests for benchmark plots. Reads saved results; does not run allocate_dp at n=1000."""

from pathlib import Path

from benchmarks.measure import load_results
from benchmarks.plot import (
    RUNTIME_CHART,
    VALUE_GAP_CHART,
    plot_runtime,
    plot_value_gap,
    save_charts,
)


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


def test_save_charts_writes_both_canonical_names(tmp_path: Path) -> None:
    runtime, gap = save_charts(load_results(), directory=tmp_path)

    assert runtime.name == RUNTIME_CHART.name
    assert gap.name == VALUE_GAP_CHART.name
    assert runtime.is_file() and gap.is_file()


def test_committed_charts_are_in_the_repo() -> None:
    assert RUNTIME_CHART.is_file() and RUNTIME_CHART.stat().st_size > 0
    assert VALUE_GAP_CHART.is_file() and VALUE_GAP_CHART.stat().st_size > 0
