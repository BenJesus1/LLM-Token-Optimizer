# Project Handoff: LLM Token-Budget Allocator — v4 (Predict-then-allocate)

**For:** Cursor (AI pair-programming context)
**Owner:** Ben
**Builds on:** v1 (DP / greedy / random), v2 (FastAPI), v3 (cached live scores).
**Purpose of this doc:** Scope a v4 phase that answers the question v3 could not: **if values are unknown at allocate time, can we still beat a dumb policy?** Plans stay on the same `allocate_dp` / `allocate_greedy` functions.

> Public `/allocate`, pytest, and CI stay **offline**. Live k=3 collect is opt-in (`collect_v4 --live`) into a **new** cache file.

---

## 1. Why This Phase Exists

v3 showed DP = greedy on known 0/1 Haiku tables (easy QA and Easy LeetCode). That is not a bug: every win was worth 1.0 at 64 tokens. It is also **not useful as an allocator**, because a real service does not have the test cache yet.

| Goal | How v4 delivers it |
|---|---|
| Decide before seeing test scores | Fit a tiny predictor on a 6-question train split; allocate the 4-question test split from **predictions** |
| Unequal values so greedy can differ | k=3 mean accuracy in `[0, 1]`, temperature 0.5 |
| A knob that can change quality | Discrete actions: skip / Haiku-64 / Haiku-256 (cost units 0 / 1 / 4), not the unused 128/512 rungs |
| Keep v1 credibility | Do **not** rewrite the DP recurrence. Envelope tables so existing `value_at` stays valid |
| No surprise bills | Public `/allocate` unchanged. `/allocate-predicted` only if `ALLOCATOR_V4_LOCAL=1` |

**Target effort:** ~16 hours (Phases 0–4).

---

## 2. One-Line Description

Collect k=3 Haiku scores at 64 and 256 tokens, predict those means from question features on a held-out split, run v1 DP/greedy on the predictions, and judge the assignments on **realized** test tables versus uniform-64 and oracle DP.

---

## 3. Scope

### In scope

- Frozen `questions_v4.json` (n=10), `cache_v4.json`, cost-unit envelope adapter.
- k=3, T=0.5, levels 64 and 256, Haiku required; second model optional if pinned.
- Pure-Python ridge predictor; seed-2026 6/4 split; leak rule.
- Offline `python -m benchmarks.v4_replay` plus fixture cache for CI.
- Local `POST /allocate-predicted` behind `ALLOCATOR_V4_LOCAL=1`.
- README v4: predicted-DP realized score vs uniform-64 vs oracle DP (honest if the predictor loses).
- Thin roundtrip JSON (predicted vs realized). No bandits.

### Explicitly out of scope

- MMLU / Medium LeetCode dumps; rewriting `allocate_dp`; public live `/allocate`; DAGs/SLO/queues; forcing a DP vs greedy gap on v3 caches; overwriting v3 cache files.

---

## 4. How v4 maps onto v1

1. For each question and each action `(model, L)`, take the mean of k binary scores.
2. Map actions to cost units (64 → 1, 256 → 4). Build a **monotone envelope** (cumulative max, drop dominated).
3. Predict those means on test ids from train-only labels.
4. `allocate_dp(predicted_tasks, B)` then `realized_task.value_at(tokens)` for the score that matters.

Call counts: Haiku-only **60**; two models **120**; cap **150**.

---

## 5. Phases

Work in order. Requirements (`docs/v4-requirements.md`) before the v4 collect loop.

### Phase 0 — Docs

This file plus `docs/v4-requirements.md`. Pins: n=10, k=3, T=0.5, levels 64/256, cost 1/4, split 6/4, `MAX_NEW_CALLS_V4=150`.

### Phase 1 — Collect + adapter

Cache key grows `sample` (default 0). Completer accepts temperature (v3 default 0). `collect_v4` writes `cache_v4.json`. `actions.py` builds envelope `Task`s.

### Phase 2 — Predict and replay

`predict.py`, `benchmarks/v4_replay.py`, `fixture_cache_v4.json`, `v4_results.json`, new chart PNG (do not replace the v3 fixture chart).

### Phase 3 — Tests

`tests/test_v4.py` only. Envelope, leak rule, stub k=3 collect, local route off by default.

### Phase 4 — Local route + README

Env-gated endpoint, README v4 + limitations, package version **0.4.0**.

---

## 6. Definition of Done

- [x] `docs/v4-requirements.md` exists and was written before the v4 client loop.
- [x] Fixture replay is green in pytest with no secrets.
- [x] Live collect into `cache_v4.json` **or** fixture-only plus committed `v4_results.json`.
- [x] README reports predicted-DP **realized** vs uniform-64 vs oracle DP on the test split.
- [x] Public `/allocate` still does not call Anthropic.
- [x] Ben can explain: v3 optimized over a cache; v4 plans on a **predictor** and is judged on **held-out realized** means.

---

## 7. Notes for Cursor

- Never log API keys.
- Do not import `collect` from `allocator.live` package `__init__` (runpy warning).
- Do not force predicted-DP to beat greedy or uniform-64.
- v3 `MAX_NEW_CALLS` stays 100; v4 uses 150.
