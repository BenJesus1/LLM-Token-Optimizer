# API requirements — `POST /allocate`

Written **before** the FastAPI implementation. The HTTP service wraps the existing v1 solvers (`allocate_dp`, `allocate_greedy`). It does not reimplement them.

Local run (after the API is added): `uvicorn allocator.api.app:app --reload`

---

## 1. Functional requirements

### 1.1 Endpoint

| Item | Value |
| --- | --- |
| Method / path | `POST /allocate` |
| Content-Type | `application/json` |
| Auth | None (public demo) |
| Persistence | None (request-scoped task lists only) |

Optional probe for hosts that expect a GET: `GET /health` returns `200` and `{"status": "ok"}`. Not a solver endpoint.

### 1.2 Request body

Same fields as `tasks.json` / `load_tasks`:

```json
{
  "budget": 20,
  "tasks": [
    {"id": "summarize", "token_cost": 12, "weight": 2.5, "curve": "log"}
  ]
}
```

| Field | Type | Required | Rule |
| --- | --- | --- | --- |
| `budget` | integer | yes | Domain: non-negative. Negative values are rejected **after** JSON parse (same as the CLI / `allocate_dp`). |
| `tasks` | array of objects | yes | Must be non-empty. |
| `tasks[].id` | string | yes | Non-empty (enforced by `Task`). |
| `tasks[].token_cost` | integer | yes | Non-negative cap (enforced by `Task`). |
| `tasks[].weight` | number | yes | Non-negative (enforced by `Task`). |
| `tasks[].curve` | string | no | Default `"log"`. One of `log`, `sqrt`, `constant`. |

Domain checks reuse `Task.__post_init__` and the solvers. The API layer must not copy those rules into a second validator.

### 1.3 Success response (`200`)

Mirrors the CLI summary: DP assignment plus token-level greedy comparison. Zero-token tasks are omitted from `assignments`, matching `Allocation`.

```json
{
  "assignments": [{"id": "summarize", "tokens": 6, "value": 4.887}],
  "total_cost": 20,
  "total_value": 16.1086,
  "dp_runtime_s": 0.0004,
  "greedy": {
    "total_value": 16.1086,
    "total_cost": 20,
    "delta_vs_dp": 0.0,
    "delta_vs_dp_pct": 0.0
  },
  "solvers": ["allocate_dp", "allocate_greedy"]
}
```

| Field | Meaning |
| --- | --- |
| `assignments` | DP result, input order, positive tokens only. |
| `total_cost` / `total_value` | DP totals. |
| `dp_runtime_s` | Wall time of `allocate_dp` only (`time.perf_counter`), same as the CLI. |
| `greedy.total_value` | Token-level `allocate_greedy` (not `knapsack_greedy`). |
| `greedy.delta_vs_dp` | `dp.total_value - greedy.total_value`. |
| `greedy.delta_vs_dp_pct` | Percent of greedy value, or `null` when greedy value is 0 (CLI prints `n/a`). |
| `solvers` | Names actually called, for logs and clients. |

On the sample `tasks.json` with `budget` 20, the JSON assignments must match calling `allocate_dp` / `allocate_greedy` in-process (summarize 6, review 5, translate 4, classify 5).

### 1.4 Error cases

| Condition | HTTP | Body |
| --- | --- | --- |
| Malformed JSON / missing required fields / wrong JSON types | **422** | FastAPI / Pydantic validation error. |
| Empty `tasks` list | **400** | `{"detail": "..."}` — same domain as CLI `"task list is empty"`. |
| Negative `budget` | **400** | Domain `ValueError` from `allocate_dp`. |
| Unknown `curve`, empty `id`, negative `token_cost` / `weight` | **400** | `ValueError` from `Task`. |
| `len(tasks) > 100` | **400** | Oversized task list (see NFRs). |
| `budget > 2000` | **400** | Oversized budget. |
| Any `token_cost > 2000` | **400** | Per-task cap too large (DoS / `O(n · B · cap)`). |

No stack traces on the wire. Solver files are not modified to implement these caps; the API service layer rejects before calling DP.

---

## 2. Non-functional requirements

v1's 1,000-task DP is ~4 s on one machine. A public URL must not accept that size.

| Limit | Value | Why |
| --- | --- | --- |
| Max tasks `n` | **100** | v1 n=100 DP is ~31 ms; n=1000 is in-process only via `python -m benchmarks.measure`. |
| Max `budget` `B` | **2000** | Bounds the DP table width. |
| Max per-task `token_cost` | **2000** | Bounds the inner `t` loop. |
| Latency (load-test payload) | **p95 < 500 ms** at concurrency 10 on the load-test machine | Sample body: `tasks.json` / budget 20. One run per cell, same honesty as v1 benches. |
| Concurrency to measure | **1 / 10 / 50** simultaneous `POST /allocate` | Record p50 and p95 in `benchmarks/api_load_results.json`. |
| Logging | One structured JSON line per request on stdout | Fields: `task_count`, `budget`, `solvers`, `latency_s`, `status_code`. |
| Concurrency safety | Solvers have no shared mutable state | Prove with concurrent requests, not a comment only. |

CPU on `allocate_dp` is expected to saturate first under load; then p95 latency rises. Greedy is cheaper (`O(n · B)`).

---

## 3. Out of scope

No auth, no multi-tenant isolation, no database, no live LLM APIs, no Go sidecar, no frontend.
