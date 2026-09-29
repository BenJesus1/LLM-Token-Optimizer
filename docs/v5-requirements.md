# v5 requirements (one-prompt model router)

Written **before** the two-model collect loop. pytest, CI, and public `POST /allocate` stay **offline**. Live calls happen only from `python -m allocator.live.collect_v5 --live`.

---

## 1. Question v5 answers

Given **one prompt**, should we call cheap Haiku or a stronger Claude to maximize expected exact-match per cost-unit? The router sees **prompt text only** (no gold). Eval is held-out realized scores from a frozen cache.

---

## 2. Provider and models

| Item | Value |
| --- | --- |
| Provider | Anthropic Messages REST |
| Cheap | `claude-haiku-4-5-20251001` |
| Quality | `claude-sonnet-4-5-20250929` (pinned from this key’s `GET /v1/models`) |
| Auth | `ANTHROPIC_API_KEY` |
| Temperature | **0** |
| k | **1** |
| `max_tokens` | **256** |
| Prompt | v4 show-work template (`V4_WORK_PROMPT_TEMPLATE`) |

If quality 404s, abort collect (do not invent another id).

---

## 3. Questions

`benchmarks/live/questions_v5.json` — **n = 20** `{id, question, gold}`, numeric golds, mix of short arithmetic and multi-step word problems. Do **not** overwrite v3/v4 question or cache files.

---

## 4. Cache

`benchmarks/live/cache_v5.json`. Same entry schema as v3 plus `model` already on the key. Prompt template stored on the cache object.

Worst case: 20 × 2 × 1 = **40** calls. Cap **`MAX_NEW_CALLS_V5 = 80`**.

---

## 5. Cost units

Haiku call = **1**. Sonnet call = **6** (output-relative proxy; not live prices). Cascade that retries pays **both** (1 + 6 = 7).

Efficiency on a split = `sum(scores) / sum(cost_units)` (0 cost → null).

---

## 6. Router

Prompt-only features: intercept, scaled length, scaled word count, word-problem cue rate, long-prompt flag. **No gold.**

Train **12** / test **8**, seed **2026**. Fit ridge **P(Haiku score = 1)** on train. `tau` grid on train only: maximize `sum(score) - 0.15 * sum(cost)`.

Policy: `p >= tau` → Haiku, else Sonnet.

---

## 7. Eval policies (test split, cache only)

- always_haiku
- always_quality
- router
- heuristic_cascade: escalate if Haiku completion has no extractable number (no gold)
- oracle_cascade: escalate iff Haiku score is 0 (**ceiling**, not deployable)

---

## 8. Offline rules

- pytest never reads the key or opens sockets.
- `POST /route` only when `ALLOCATOR_V5_LOCAL=1`. Body `{ "question": "..." }`. Returns `{ model, p_haiku, tau, reason }`. **No Anthropic.**
- Public `/allocate` unchanged.

---

## 9. DP

`allocate_dp` is **not** on the `/route` hot path.
