"""Charts from ``benchmarks/results.json``. Does not re-run the 1,000-task DP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .measure import RESULTS_PATH, load_results

CHARTS_DIR = Path(__file__).resolve().parent / "charts"
RUNTIME_CHART = CHARTS_DIR / "runtime_vs_task_count.png"
VALUE_GAP_CHART = CHARTS_DIR / "value_gap_vs_scale.png"

SOLVER_LABELS = {"dp": "DP", "greedy": "Greedy", "random": "Random"}
SOLVER_ORDER = ("dp", "greedy", "random")


def _rows_by_solver(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in SOLVER_ORDER}
    for row in payload["rows"]:
        grouped[row["solver"]].append(row)
    for name in grouped:
        grouped[name].sort(key=lambda row: row["n"])
    return grouped


def plot_runtime(payload: dict[str, Any], path: Path = RUNTIME_CHART) -> Path:
    """Runtime vs task count, log x-axis, one series per solver."""

    grouped = _rows_by_solver(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in SOLVER_ORDER:
        rows = grouped[name]
        ax.plot(
            [row["n"] for row in rows],
            [row["runtime_s"] for row in rows],
            marker="o",
            label=SOLVER_LABELS[name],
        )
    ax.set_xscale("log")
    ax.set_title("Solver runtime vs task count")
    ax.set_xlabel("Task count (log scale)")
    ax.set_ylabel("Wall-clock runtime (s)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig.text(
        0.01,
        0.01,
        f"Source: benchmarks/results.json · seed {payload['seed']} · mean of one run per cell",
        fontsize=8,
        color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    payload = load_results(RESULTS_PATH)
    saved = plot_runtime(payload)
    print(f"wrote {saved}")


if __name__ == "__main__":
    main()
