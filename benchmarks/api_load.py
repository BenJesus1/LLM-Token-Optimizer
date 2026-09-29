"""Load-test POST /allocate at concurrency 1 / 10 / 50.

Run a local server first::

    uvicorn allocator.api.app:app --host 127.0.0.1 --port 8000

then from the repo root::

    python -m benchmarks.api_load

pytest must not import this module's ``__main__`` path and must not hit
concurrency 50 against a live server. Percentile helpers are tested separately.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_JSON = REPO_ROOT / "tasks.json"
RESULTS_PATH = Path(__file__).resolve().parent / "api_load_results.json"
DEFAULT_URL = "http://127.0.0.1:8000/allocate"
CONCURRENCIES: tuple[int, ...] = (1, 10, 50)
DEFAULT_REQUESTS = 50
DEFAULT_BUDGET = 20


def sample_payload(*, budget: int = DEFAULT_BUDGET) -> dict[str, Any]:
    """The committed ``tasks.json`` body (n=4). Safe for CI-sized work."""

    return {
        "budget": budget,
        "tasks": json.loads(TASKS_JSON.read_text(encoding="utf-8")),
    }


def percentile(samples: Sequence[float], p: float) -> float:
    """Nearest-rank percentile. ``p`` is 0..100."""

    if not samples:
        raise ValueError("samples must be non-empty")
    if not 0 <= p <= 100:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def load_results(path: Path = RESULTS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_results(payload: dict[str, Any], path: Path = RESULTS_PATH) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


async def _timed_post(client: httpx.AsyncClient, url: str, payload: dict[str, Any]) -> float:
    started = time.perf_counter()
    response = await client.post(url, json=payload)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return elapsed


async def measure_concurrency(
    *,
    url: str,
    payload: dict[str, Any],
    concurrency: int,
    n_requests: int,
) -> dict[str, Any]:
    """Issue ``n_requests`` POSTs, at most ``concurrency`` in flight."""

    if concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    limits = httpx.Limits(max_connections=max(concurrency, 8), max_keepalive_connections=concurrency)
    latencies: list[float] = []
    async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:
        remaining = n_requests
        while remaining > 0:
            wave = min(concurrency, remaining)
            batch = await asyncio.gather(
                *[_timed_post(client, url, payload) for _ in range(wave)]
            )
            latencies.extend(batch)
            remaining -= wave
    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "p50_s": percentile(latencies, 50),
        "p95_s": percentile(latencies, 95),
        "mean_s": sum(latencies) / len(latencies),
        "max_s": max(latencies),
    }


def measure_suite(
    *,
    url: str = DEFAULT_URL,
    concurrencies: Sequence[int] = CONCURRENCIES,
    n_requests: int = DEFAULT_REQUESTS,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure the requested concurrencies. Pytest must pass a tiny tuple."""

    body = payload if payload is not None else sample_payload()
    rows = [
        asyncio.run(
            measure_concurrency(
                url=url,
                payload=body,
                concurrency=n,
                n_requests=n_requests,
            )
        )
        for n in concurrencies
    ]
    return {
        "url": url,
        "n_tasks": len(body["tasks"]),
        "budget": body["budget"],
        "n_requests_per_cell": n_requests,
        "payload": "tasks.json",
        "note": "one run per concurrency; wall time at the HTTP client",
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load-test POST /allocate")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = measure_suite(url=args.url, n_requests=args.requests)
    saved = save_results(payload)
    print(f"wrote {saved}")
    for row in payload["rows"]:
        print(
            f"c={row['concurrency']:<3}  "
            f"p50={row['p50_s']*1000:.2f}ms  p95={row['p95_s']*1000:.2f}ms"
        )


if __name__ == "__main__":
    main()
