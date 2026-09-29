"""Tests for the API load harness. Never start 50 live clients in pytest."""

from pathlib import Path

import pytest

from benchmarks.api_load import CONCURRENCIES, percentile, sample_payload
from benchmarks.api_plot import P95_CHART, plot_p95


def test_percentile_is_interpolated() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0
    assert percentile([1.0, 3.0], 50) == pytest.approx(2.0)


def test_sample_payload_is_the_committed_four_tasks() -> None:
    payload = sample_payload()

    assert payload["budget"] == 20
    assert len(payload["tasks"]) == 4
    assert 1000 not in CONCURRENCIES
    assert CONCURRENCIES == (1, 10, 50)


def test_p95_plot_writes_a_png(tmp_path: Path) -> None:
    payload = {
        "n_requests_per_cell": 4,
        "payload": "test",
        "rows": [
            {"concurrency": 1, "p50_s": 0.001, "p95_s": 0.002},
            {"concurrency": 10, "p50_s": 0.003, "p95_s": 0.004},
        ],
    }
    path = tmp_path / "api_p95.png"

    written = plot_p95(payload, path=path)

    assert written == path
    assert path.is_file()
    assert path.stat().st_size > 0


def test_committed_api_load_artifacts_are_in_the_repo() -> None:
    results = Path(__file__).resolve().parents[1] / "benchmarks" / "api_load_results.json"
    assert results.is_file() and results.stat().st_size > 0
    assert P95_CHART.is_file() and P95_CHART.stat().st_size > 0
