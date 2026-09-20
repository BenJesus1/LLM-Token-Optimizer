"""Benchmark harness and chart generation for the token-budget allocator."""

from .generate import (
    DEFAULT_SEED,
    SCALES,
    SyntheticSet,
    generate_scale_suite,
    generate_set,
    generate_tasks,
)

__all__ = [
    "DEFAULT_SEED",
    "SCALES",
    "SyntheticSet",
    "generate_scale_suite",
    "generate_set",
    "generate_tasks",
]
