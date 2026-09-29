"""Bar chart of cached live scores by solver. Does not call an API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .live_replay import RESULTS_PATH

CHARTS_DIR = Path(__file__).resolve().parent / "charts"
LIVE_CHART = CHARTS_DIR / "live_score_vs_solver.png"
SOLVER_ORDER = ("dp", "greedy", "random")
SOLVER_LABELS = {"dp": "DP", "greedy": "Greedy", "random": "Random"}


def plot_live_scores(payload: dict[str, Any], path: Path = LIVE_CHART) -> Path:
    by_name = {row["solver"]: row["total_value"] for row in payload["rows"]}
    names = [SOLVER_LABELS[s] for s in SOLVER_ORDER]
    values = [by_name[s] for s in SOLVER_ORDER]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(names, values)
    ax.set_title("Cached live-eval score by solver")
    ax.set_xlabel("Solver")
    ax.set_ylabel("Total table score (sum of 0/1)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    fig.text(
        0.01,
        0.01,
        f"Source: {payload.get('source', 'cache')} · n={payload.get('n')} · "
        f"budget {payload.get('budget')} · one completion per cell · no CIs",
        fontsize=8,
        color="dimgray",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    payload = json_load_results()
    path = plot_live_scores(payload)
    print(f"wrote {path}")


def json_load_results() -> dict[str, Any]:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
