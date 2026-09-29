# v4 requirements (predict-then-allocate)

Written **before** the v4 client loop. pytest, CI, and public `POST /allocate` stay **offline**. Live calls happen only from `python -m allocator.live.collect_v4 --live`.

---

## 1. Question v4 answers

Given a small **cost-unit** budget, does DP (or greedy) on **predicted** mean accuracy beat “everyone gets cheap Haiku-64,” and how close is that to oracle DP on **held-out realized** k=3 tables?

v3 optimized over a known cache. v4 plans on a predictor and is judged on realized test means.

---

## 2. Provider and sampling

| Item | Value |
| --- | --- |
| Provider | Anthropic Messages REST (`https://api.anthropic.com/v1/messages`) |
| Client | `httpx` (no extra SDK) |
| Required model | `claude-haiku-4-5-20251001` |
| Optional second model | Pin only if `GET /v1/models` lists it; default v4 core is Haiku-only |
| Auth | `ANTHROPIC_API_KEY` from the environment or `.env` (never committed) |
| Temperature | **0.5** (v4 collect only; v3 collect stays 0) |
| k | **3** completions per `(question_id, model, token_level)` |

v3 `temperature: 0` would make k=3 identical. Do not mix v4 samples into `cache.json` / `cache_leetcode.json`.

---

## 3. Questions

File: `benchmarks/live/questions_v4.json` — **n = 10** objects `{ "id", "question", "gold" }`. Mix of grade-school arithmetic and Easy LeetCode **instances**, numeric golds where possible. Frozen in git.

Harder frozen set: `benchmarks/live/questions_v4_hard.json` (n=10, mix of short arithmetic and multi-step word problems). Collect with `--show-work` into `benchmarks/live/cache_v4_hard.json`. Do **not** overwrite `questions_v4.json` / `cache_v4.json`.

Do **not** overwrite `questions.json` or `questions_leetcode.json`.

---

## 4. Actions and cost units

Per question the menu is **skip** (cost 0, value 0) plus:

| Action | `max_tokens` | Cost units |
| --- | --- | --- |
| Haiku cheap | 64 | **1** |
| Haiku longer | 256 | **4** |

128 and 512 are dropped (v3 extra `max_tokens` did not raise exact-match on easy items).

Cost units are a coarse dollar proxy: 64 Haiku output tokens = 1 unit, so 256 tokens = 4. Optional second-model actions, if pinned, use a larger integer cost from the public $/MTok ratio.

`Task.token_cost` is the cap in **cost units** (max action cost = 4 for Haiku-only), not raw tokens.

Global budget: `B = max(1, n * cap // 4)` with `cap = 4` → **10** when n = 10 (full set), **4** when n = 4 (test split). This is the v4 replay budget, not the v2 API cap (2000).

---

## 5. Prompt and scoring

Same prompt template and exact-match / last-number rule as v3 (`docs/live-eval-requirements.md` §§4–5). Value at an action is the **mean** of k=3 binary scores, in `[0, 1]`.

---

## 6. Cache schema

`benchmarks/live/cache_v4.json` (never the v3 cache files):

```json
{
  "model": "claude-haiku-4-5-20251001",
  "prompt_template": "...",
  "temperature": 0.5,
  "k": 3,
  "entries": [
    {
      "question_id": "v01",
      "token_level": 64,
      "model": "claude-haiku-4-5-20251001",
      "prompt_hash": "...",
      "sample": 0,
      "completion": "...",
      "score": 1.0,
      "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }
  ]
}
```

Key: `(question_id, token_level, model, prompt_hash, sample)`. `sample` defaults to **0** when missing so v3 lookups still hit.

Hard cap: abort if a v4 collect would need **more than 150** new HTTP calls. Haiku-only worst case: 10 × 2 × 3 = **60**. Two models: 120 (< 150).

---

## 7. Envelope tasks

Raw action scores need not increase with cost. Before `allocate_dp`, convert actions to a strictly increasing cost table with **cumulative max** scores and drop dominated points. Then `value_at(t)` = score of the largest remaining level `≤ t` (existing table curve). Skip is t = 0.

---

## 8. Train / test and predictor

Split: seed **2026**, shuffle the 10 questions, **6 train / 4 test**.

Features (question text and gold only): intercept, scaled character length, scaled word count, numeric-gold flag, “leetcode” substring flag.

One clipped-linear (ridge) model **per (model, level)**. Test labels never enter fit.

Allocate on **predicted** envelopes for the test questions. Score those assignments on **realized** k=3 envelopes.

---

## 9. Policies to report (test split, same B)

- Oracle DP / greedy / random on realized tables
- DP / greedy on predicted tables, **realized** score of those assignments
- Uniform cheap (all Haiku-64) and uniform long (all Haiku-256)

Do not require predicted-DP to beat greedy. README must say whether predicted-DP beat uniform-64 on realized score.

---

## 10. Offline rules

- pytest never reads `ANTHROPIC_API_KEY` and never opens sockets for collect.
- Public FastAPI `POST /allocate` stays simulated `log` / `sqrt` / `constant`.
- `POST /allocate-predicted` exists only when `ALLOCATOR_V4_LOCAL=1`. It fits/applies the predictor from a local cache (fixture if no live cache). It does **not** call Anthropic.
- Replay from cache needs **no** key.

---

## 11. Closed loop

After replay, write `benchmarks/live/v4_roundtrip.json`: predicted vs realized mean per test id. No bandit or online SGD.
