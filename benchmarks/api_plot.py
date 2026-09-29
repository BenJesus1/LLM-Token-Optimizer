"""Chart API p95 latency vs concurrency. Does not start a server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .api_load import RESULTS_PATH, load_results

CHARTS_DIR = Path(__file__).resolve().parent / "charts"
P95_CHART = CHARTS_DIR / "api_p95_vs_concurrency.png"


def plot_p95(payload: dict[str, Any], path: Path = P95_CHART) -> Path:
    """p50 and p95 HTTP latency vs concurrent clients."""

    rows = sorted(payload["rows"], key=lambda row: row["concurrency"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        [row["concurrency"] for row in rows],
        [row["p50_s"] * 1000.0 for row in rows],
        marker="o",
        label="p50",
    )
    ax.plot(
        [row["concurrency"] for row in rows],
        [row["p95_s"] * 1000.0 for row in rows],
        marker="o",
        label="p95",
    )
    ax.set_title("API latency vs concurrent POST /allocate")
    ax.set_xlabel("Concurrent clients")
    ax.set_ylabel("Latency (ms)")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)
    fig.text(
        0.01,
        0.01,
        f"Source: benchmarks/api_load_results.json · {payload.get('payload', 'tasks.json')} · "
        f"{payload.get('n_requests_per_cell', '?')} requests per cell",
        fontsize=8,
        color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_charts(
    payload: dict[str, Any] | None = None,
    *,
    directory: Path = CHARTS_DIR,
) -> Path:
    data = payload if payload is not None else load_results(RESULTS_PATH)
    return plot_p95(data, path=directory / P95_CHART.name)


def main() -> None:
    path = save_charts()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
