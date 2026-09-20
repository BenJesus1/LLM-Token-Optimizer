"""Tests for the benchmark harness. Never run the 1,000-task DP here."""

from benchmarks.generate import generate_set
from benchmarks.measure import measure_instance, measure_suite


def test_measure_instance_records_dp_greedy_and_random() -> None:
    instance = generate_set(10, seed=2026)
    rows = measure_instance(instance, random_seed=2026)
    names = [row["solver"] for row in rows]

    assert names == ["dp", "greedy", "random"]
    for row in rows:
        assert row["n"] == 10
        assert row["runtime_s"] >= 0
        assert row["total_value"] >= 0
        assert row["total_cost"] <= instance.budget


def test_measure_suite_stays_on_tiny_scales_in_pytest() -> None:
    # Explicit small scales: CI must never time allocate_dp on n=1000.
    rows = measure_suite(seed=2026, scales=(10,))

    assert {row["n"] for row in rows} == {10}
    assert 1000 not in {row["n"] for row in rows}
    dp = next(row for row in rows if row["solver"] == "dp")
    greedy = next(row for row in rows if row["solver"] == "greedy")
    random_row = next(row for row in rows if row["solver"] == "random")
    assert dp["total_value"] >= greedy["total_value"]
    assert dp["total_value"] >= random_row["total_value"]
