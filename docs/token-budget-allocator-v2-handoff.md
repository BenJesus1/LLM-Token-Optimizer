# Project Handoff: LLM Token-Budget Allocator — v2 (Deployed Service)

**For:** Cursor (AI pair-programming context)
**Owner:** Ben
**Builds on:** `token-budget-allocator-handoff.md` (v1) — the DP/greedy/random allocator, already built, tested, and confirmed to satisfy the DS&A fundamental plus the OpenAI and Anthropic rows on the tracker.
**Purpose of this doc:** Scope a v2 phase that extends the existing repo rather than starting over, targeted specifically at the next open item on the internship build-order roadmap.

---

## 1. Why This Phase Exists

v1 proved the algorithm. v2's job is narrower: turn the CLI tool into a small networked, deployed service, because that's what checks off two roadmap items that are still open.

| Goal | How v2 delivers it |
|---|---|
| Check off **"SpaceX-style networked/systems project"** (build order #4 — currently unclaimed) | A FastAPI HTTP service wrapping the existing solvers, with a written requirements doc and a system diagram produced *before* the code — the exact "documentation, diagrams, debugging, testing" skills that posting calls out |
| Check off **"End-to-end deployed project"** (build order #5 / universal fundamental — currently unclaimed) | The service is actually deployed to a live URL, not just runnable on your machine |
| Add real interview material | Concurrency and latency under load is a harder, more interesting failure mode than anything in v1 — good raw material for the "ambiguous debugging" fundamental (build order #11) |
| Protect v1's credibility | The DP/greedy/random solvers are not touched. v2 wraps them; it does not rewrite, "optimize," or simplify them. |

**Explicitly not the target of this phase:** the Google row's "Partial" status (it wants 2-3+ languages; this repo is Python-only). Fixing that would mean porting the core solver to a second language, which is a different kind of work — it's listed below as an optional Phase 5 stretch, not core v2 scope. Trying to fix both the SpaceX/deployed gaps *and* the Google gap in one handoff is how a well-scoped v1 turns into a sprawling v2 — resist that.

**Target effort:** ~8-12 hours (Phases 0-4). Phase 5 is optional and not counted in that estimate.

---

## 2. One-Line Description

Wrap the existing, already-tested DP/greedy/random allocator behind a small, documented, load-tested, and deployed HTTP API — same algorithms, now reachable over the network with real request handling.

---

## 3. Scope

### In scope (v2 — what "done" means)

- **FastAPI service** exposing `POST /allocate`, accepting `{budget, tasks: [...]}` and returning the DP allocation plus a greedy comparison — built by calling the existing `allocate_dp` / `allocate_greedy`, not reimplementing them.
- **Pydantic request/response models** that mirror the existing `Task` / `Allocation` dataclasses (translate at the boundary; don't duplicate solver logic in the API layer).
- **A written requirements doc** (`docs/api-requirements.md`): functional requirements (accepted input, returned output, error cases) and non-functional requirements (latency budget, max task-list size, concurrency target) — written *before* Phase 1's code, not after.
- **A system architecture diagram** (Mermaid in the README is fine) showing the request flow: client → FastAPI → `{allocate_dp, allocate_greedy}` → JSON response.
- **Structured request logging**: task count, budget, solver used, latency per request.
- **Verified concurrency safety**: the solvers are already pure functions with no shared mutable state — this phase *proves* that under a real concurrent load test rather than just assuming it.
- **A load test** (e.g., `locust`, or a small `asyncio`/`httpx` script) at increasing concurrency (1 / 10 / 50 simultaneous requests), with p50/p95 latency recorded — same benchmarking discipline as v1's `benchmarks/`.
- **A real deployment**: live on a free-tier host (Render, Fly.io, or Railway — whichever has the least setup friction for FastAPI), with redeploy steps documented in the README.
- **New tests for the API layer**, additive to v1's suite — don't touch `tests/test_knapsack.py` or `tests/test_allocate.py`.

### Explicitly out of scope for v2

- **No changes to the solvers.** `knapsack.py`, `allocate.py`, `greedy.py`, `random_alloc.py`, `value.py`, `task.py`, `allocation.py` stay as-is. If Cursor finds something worth fixing in them while working on this, flag it — don't fix it silently inside a v2 PR.
- **No auth or multi-tenancy.** A public, read-only-ish demo endpoint is fine for a portfolio piece.
- **No second language this round.** Porting the solver to Go/Rust is Phase 5, optional, and separate from this phase's goal.
- **No database.** Task lists stay request-scoped; nothing here needs persistence.
- **No live LLM API calls.** Same reasoning as v1 — this stays a DS&A/systems artifact, not a billed client.

---

## 4. Phases

### Phase 0 — Requirements & design (~1-2 hrs)
- Write `docs/api-requirements.md` first: what `/allocate` accepts, what it returns, every error case (malformed JSON, negative budget, empty task list, oversized task list).
- Draw the system diagram (Mermaid in the README: client → FastAPI → solvers → response).
- This phase *is* the deliverable that satisfies the "documentation, diagrams, requirements" signal — don't skip to code before it exists.

### Phase 1 — API skeleton (~2-3 hrs)
- FastAPI app with `POST /allocate`; Pydantic models mirroring `Task`/`Allocation`.
- Wire the endpoint straight to `allocate_dp` and `allocate_greedy` — no algorithm changes.
- Error handling: 400 for malformed input, 422 for validation errors, clear messages.

### Phase 2 — API-layer tests (~2 hrs)
- Valid-request tests, including re-running a couple of v1's hand-computed cases through the API to prove the API's answer matches the library's answer exactly.
- Validation-error tests (missing fields, negative budget, empty list) return the right status codes.
- Keep these in a new test file — don't modify v1's solver tests.

### Phase 3 — Concurrency & load testing (~2-3 hrs)
- Confirm (don't just assume) the solvers have no shared mutable state under concurrent calls.
- Load test at 1 / 10 / 50 concurrent requests; record p50/p95 latency.
- Save results in the same style as v1's `benchmarks/results.json` (e.g., `benchmarks/api_load_results.json`) plus a chart.

### Phase 4 — Deploy (~1-2 hrs)
- Deploy to a free-tier host; document the exact redeploy steps in the README.
- Smoke-test the live URL with a real request before calling this phase done.

### Phase 5 — Stretch (optional)
- Port the core DP solver to Go as a callable sidecar, benchmarked against the Python version — this is what would also resolve the Google row's "Partial" status, but it's optional here so v2 doesn't overrun its own scope.
- A minimal single-page frontend that hits the API and visualizes the allocation.

---

## 5. Definition of Done

- [ ] `docs/api-requirements.md` exists and was written before Phase 1's code.
- [ ] System diagram is in the README.
- [ ] `/allocate` works locally and returns results matching the v1 CLI on the same input.
- [ ] New API-layer tests pass, and every v1 solver test still passes unmodified.
- [ ] Load-test results (1/10/50 concurrency) are saved in the repo.
- [ ] The service is live at a real URL, and the README documents how to redeploy it.
- [ ] Ben can explain the request lifecycle end-to-end, and name what would break first under heavy load, without re-reading the code.

---

## 6. Notes for Cursor

- Don't touch the existing solver files. v2 is additive — new code goes in a new `src/allocator/api/` module, not mixed into `cli.py`.
- Phase 0's requirements doc and diagram are the actual point of this phase — get those right before writing the endpoint.
- Match v1's engineering bar: type hints, docstrings, tests alongside code, honest incremental commits.
- If solver logic starts leaking into the API layer (validation re-implementing what `Task.__post_init__` already checks, etc.), that's a sign to reuse the existing dataclass instead of duplicating rules — flag it rather than quietly working around it.
