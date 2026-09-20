"""Command-line interface: ``allocator run --budget N --tasks FILE``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .allocate import allocate_dp
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


def _print_allocation(allocation) -> None:
    print("Allocation (DP):")
    if not allocation.assignments:
        print("  (none)")
        return
    for item in allocation.assignments:
        print(f"  {item.task.id}\t{item.tokens}")


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.error(f"unknown command {args.command}")
    tasks = load_tasks(args.tasks)
    allocation = allocate_dp(tasks, args.budget)
    _print_allocation(allocation)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
