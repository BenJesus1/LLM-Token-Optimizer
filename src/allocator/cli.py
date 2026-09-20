"""Command-line interface: ``allocator run --budget N --tasks FILE``."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from .allocate import allocate_dp
from .allocation import Allocation
from .greedy import allocate_greedy
from .tasks_io import load_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="allocator",
        description="Allocate a token budget across competing LLM tasks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="solve one instance")
    run.add_argument("--budget", type=int, required=True, help="token budget")
    run.add_argument("--tasks", type=Path, required=True, help="JSON or CSV task list")
    return parser


def _print_allocation(allocation: Allocation) -> None:
    print("Allocation (DP):")
    if not allocation.assignments:
        print("  (none)")
        return
    for item in allocation.assignments:
        print(f"  {item.task.id}\t{item.tokens}")


def _print_summary(dp: Allocation, greedy: Allocation, runtime_s: float) -> None:
    print(f"Total value (DP): {dp.total_value:.4f}")
    print(f"Runtime (DP): {runtime_s:.4f}s")
    delta = dp.total_value - greedy.total_value
    if greedy.total_value == 0:
        rel = "n/a"
    else:
        rel = f"{100.0 * delta / greedy.total_value:+.1f}%"
    print(
        f"Greedy: value {greedy.total_value:.4f}  "
        f"(DP {delta:+.4f} / {rel} vs token-level greedy)"
    )


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.error(f"unknown command {args.command}")
    tasks = load_tasks(args.tasks)
    started = time.perf_counter()
    dp = allocate_dp(tasks, args.budget)
    runtime_s = time.perf_counter() - started
    greedy = allocate_greedy(tasks, args.budget)
    _print_allocation(dp)
    _print_summary(dp, greedy, runtime_s)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
