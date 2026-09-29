# Project Handoff: LLM Token-Budget Allocator — v3 (Live scores, cached)

**For:** Cursor (AI pair-programming context)  
**Owner:** Ben  
**Builds on:** v1 (DP / greedy / random solvers) and v2 (FastAPI wrapper).  
**Purpose of this doc:** Scope a v3 phase that answers a question v1 could not: **on real model outputs, does exact DP beat token-level greedy and random?** Values stop being `log` / `sqrt` / `constant` formulas and become **cached live-API scores**. The solvers stay the same functions.

> v1/v2 explicitly forbade live LLM calls. v3 lifts that ban **only** behind a collect-once cache. pytest, CI, and the public `/allocate` demo must still run **offline**.

---

## 1. Why This Phase Exists

v1 proved the math on simulated curves. v2 put that math on the network. Interviewers can still ask: “Would this matter if value came from a real model?” v3 is the smallest honest experiment that answers that, without turning the repo into a billed chatbot.

| Goal | How v3 delivers it |
|---|---|
| Compare DP vs greedy vs baseline on **live data** | Run a frozen question set through a real LLM at a few token budgets, save every completion, score it, then feed those tables into `allocate_dp` / `allocate_greedy` / `allocate_random` |
| Keep v1’s credibility | Do **not** rewrite the DP recurrence. Adapt *inputs* (`value_at(t)` from a table). If a solver file must change, flag it — prefer an additive `Task` field or a thin wrapper. |
| Control cost | Collect **once**, write JSON to disk, never hit the API from pytest. Tiny n (suggest **20 questions × 4 token levels**). |
| Honest write-up | Charts + a short Results/Limitations addendum: sample size, one-run noise, what the API did not prove |

**Target effort:** ~10–14 hours (Phases 0–4). Phase 5 is optional.

---

## 2. One-Line Description

Call a live LLM on a small, frozen question set at a few max-token budgets, cache the scored results, then reuse v1’s DP / greedy / random to split a **global** token budget — and report whether DP actually wins.

---

## 3. Scope

### In scope (v3 — what “done” means)

- **Frozen eval set:** a small public QA slice committed or downloaded with a pinned filename (e.g. 20 GSM8K-style or TriviaQA items, ids stable). Not a moving scrape.
- **Live provider:** one chat API (OpenAI **or** Anthropic — pick one in Phase 0 and stick to it). Key from env (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`), never committed.
- **Discrete token levels:** not every integer `t`. Default ladder: **64 / 128 / 256 / 512** max output tokens (adjust in Phase 0 if the model is expensive). `token_cost` cap = the largest level.
- **Collect-once cache:** `benchmarks/live/cache.json` (or similar) keyed by `(question_id, token_level, model, prompt_hash)`. Re-runs must **skip** the API when the key exists.
- **Scoring:** exact-match or simple numeric/string normalize against the gold answer (document the rule). Value at a level = **0/1 accuracy** for that one run, or mean accuracy if you allow `k` samples (default **k=1** to save money).
- **Table-valued tasks:** each question becomes a `Task` whose `value_at(t)` is the cached score at the **largest level ≤ t** (step interpolation). That matches “pick one budget level per question” without a new DP.
- **Comparison:** on a held **global** budget `B` (e.g. quarter of `n × max_level`, same spirit as v1), run:
  - `allocate_dp`
  - `allocate_greedy` (token-level; constant-style jumps between levels if the table is a step)
  - `allocate_random` (seed **2026**)
- **Offline replay:** `python -m benchmarks.live_replay --cache FILE --budget B` uses **only** the cache. This is what CI runs.
- **README addendum:** real totals (sum of scored values), spend (tokens billed if the API returns usage), and an honest “this is n=20, k=1, not a leaderboard.”
- **Tests:** new files only. Fixture cache with **fake** completions (no network). Assert replay DP ≥ greedy ≥? (greedy may win or tie — **do not force DP to win**). Assert cache hits do not call a stub client.

### Explicitly out of scope for v3

- **No live calls from the deployed FastAPI.** v2’s public `/allocate` stays simulated (or replay-from-cache if you add an optional local-only flag that is **off** in production). A public URL with your API key is how you get a surprise bill.
- **No seven-strategy zoo.** Only DP, token-level greedy, and random (v1’s three). 0/1 `knapsack_greedy` is optional extra, not required.
- **No difficulty predictor / train-test leak theater.** Do not estimate P(correct) from features in v3 core. Stretch only.
- **No bandits, no online learning, no multi-model router.**
- **No rewriting `allocate_dp`.** Discrete levels are encoded in `value_at`, not a new recurrence.
- **No CI that needs secrets.** GitHub Actions stays cache/fixture-only.
- **No huge datasets** (MMLU-full, 1k GSM8K). Cost and time will explode.

---

## 4. How live data maps onto v1

v1 DP already maximizes `Σ value_i(t_i)` with `Σ t_i ≤ B` and `t_i ≤ cap_i`.

Live experiment:

1. For each question `i` and each level `L` in `{64,128,256,512}`, call the model with `max_tokens=L`, score the answer → `s_i(L) ∈ {0,1}` (or `[0,1]`).
2. Define `value_i(t) = s_i(L*)` where `L*` is the greatest level `≤ t` (0 if `t` is below the smallest level).
3. Hand that into the **existing** `allocate_dp`. It will only “want” to sit on those steps because in-between tokens do not raise value (same idea as `constant` curves).

Greedy should jump to the next level when the marginal gain is positive (already true for `constant` in `_greedy_step`). If greedy stays unit-step on empirical tables, **fix greedy in an additive way** (treat table/step curves like `constant`) or document that unit-step greedy is a weak baseline on stepwise live scores. Flag this in Phase 1; do not silently change log/sqrt behavior.

Random still throws unit tokens; it remains the dumb baseline.

---

## 5. Phases

Work in order. Do **not** implement Phases 0–4 in one Agent turn.

### Phase 0 — Design & cost cap (~1–2 hrs)

Write `docs/live-eval-requirements.md` **before** any API client code:

- Provider + model id (e.g. a cheap small model, not the frontier flagship).
- Question source, n=20, gold-answer field.
- Token ladder, global `B` rule, scoring rule.
- Estimated worst-case calls: `20 × 4 × k` (with k=1 → 80 calls). Hard cap in code: abort if a run would exceed **N** new API calls (suggest **100**).
- Cache schema.
- Confirm: pytest never imports a client that reads the real key unless a marker is set (default off).

Optional: one Mermaid diagram — `questions → live collect → cache.json → replay → DP/greedy/random → chart`.

### Phase 1 — Cache + client (~3–4 hrs)

New package, e.g. `src/allocator/live/` (not mixed into `allocate.py`):

- HTTP client with retries/backoff, timeout, recorded `usage` tokens if present.
- `collect.py`: loop questions × levels, write/merge cache.
- Table adapter: build `Task` list from cache for a given model id.

**Allowed solver-adjacent change (if needed):** `Task.value_at` understands `curve="table"` plus an immutable lookup. Keep log/sqrt/constant tests green. If you can avoid touching `task.py` by a wrapper type that still satisfies what `allocate_dp` calls at runtime, prefer that and add a Protocol later.

`.env` is gitignored. Document `cp .env.example .env`.

### Phase 2 — Replay comparison (~2 hrs)

- `python -m benchmarks.live_replay` prints DP vs greedy vs random totals (sum of table values) and token spend.
- Save `benchmarks/live/results.json` in the same spirit as v1 `results.json`.
- Chart: three bars (or grouped bars) of **realized score** under the same `B`. If you only have one collect run, **do not** draw fake confidence intervals — say “one completion per cell.” Stretch: bootstrap CIs only if k>1 or you re-collect.

### Phase 3 — Tests (~2 hrs)

- Fixture cache: two questions, two levels, hand-computed DP vs greedy vs random.
- Client stub: collecting twice does not call the stub the second time.
- Missing gold / API error → skip or record a failure in cache, no traceback to the user.
- Do not modify `tests/test_allocate.py` / `tests/test_knapsack.py` unless `Task` gained a field and those tests need a default (prefer default `None` so they stay untouched).

### Phase 4 — Write-up (~2 hrs)

README section **Live eval (v3)**:

- How to collect (real key) vs how to replay (no key).
- Table of DP / greedy / random scores on the cached run.
- Limitations: k=1 noise, stepwise `max_tokens` ≠ “thinking longer,” judge/exact-match errors, n=20, DP is exact **on the cached table**, not on the unknown true P(correct).
- Cost: approximate USD or token usage from the cache.

### Phase 5 — Stretch (optional)

- k=3 samples per cell and simple 95% intervals.
- A held-out split + a tiny difficulty model (this is the earlier “predict before run” idea — **after** v3 core).
- Extra baselines (uniform levels, 0/1 density greedy).
- Optional `POST /allocate` replay-from-cache in **local** mode only.

---

## 6. Definition of Done

- [ ] `docs/live-eval-requirements.md` exists and was written before the client.
- [ ] A cache file can be replayed with **no** API key; pytest is green on fixtures.
- [ ] One documented live collect run (Ben’s machine) produced a committed cache **or** a gitignored cache plus committed summary JSON if raw completions are too bulky/sensitive.
- [ ] Replay shows numeric DP vs greedy vs random on that cache (honest: greedy may match or beat DP on a tiny noisy table).
- [ ] README live section + limitations.
- [ ] Public v2 deploy still does not spend API credits.
- [ ] Ben can explain: we optimized over **cached scores**, not over the unknown live future.

---

## 7. Notes for Cursor

- Default path is **offline replay**. `collect` is opt-in (`--live` flag).
- Never log API keys. Redact request headers in fixtures.
- Prefer `httpx` (already a v2 dep) over extra SDKs if a raw REST call stays readable; an official SDK is OK if it cuts auth bugs — declare it in `pyproject.toml`.
- If live greedy behaves badly on step tables, treat it as a **baseline bug worth a comment**, not a silent rewrite of log-curve greedy.
- Resist scope creep into “predict difficulty” and “seven strategies” until Phase 5.
- Incremental commits: requirements → client/cache → replay/chart → tests → README.

---

## 8. How Ben should prompt (one phase per turn)

> Do Phase N from `docs/token-budget-allocator-v3-handoff.md`. Constraints: no live calls in pytest; do not rewrite `allocate_dp`; public FastAPI stays offline; type hints, docstrings, new tests in new files.

Do not ask for Phases 0–4 in one shot.
