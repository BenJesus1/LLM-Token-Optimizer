# Live eval requirements (v3)

Written **before** the client. pytest, CI, and `POST /allocate` stay **offline**. Live calls happen only from `python -m allocator.live.collect --live`.

---

## 1. Provider and model

| Item | Value |
| --- | --- |
| Provider | Anthropic Messages REST (`https://api.anthropic.com/v1/messages`) |
| Client | `httpx` (no extra SDK) |
| Model | `claude-haiku-4-5-20251001` |
| Auth | `ANTHROPIC_API_KEY` from the environment or `.env` (never committed) |
| Temperature | 0 |
| k | 1 completion per `(question_id, token_level)` |

## 2. Questions

File: `benchmarks/live/questions.json` — **20** objects `{ "id", "question", "gold" }`. Grade-school arithmetic plus one-line facts. Frozen in git; not a live scrape.

Optional second frozen set: `benchmarks/live/questions_leetcode.json` — **20** Easy LeetCode **instances** (named problem + one example, numeric gold). Collect into `benchmarks/live/cache_leetcode.json`; do not overwrite `questions.json` / `cache.json`.

## 3. Token ladder and global budget

Levels: **64, 128, 256, 512** (`max_tokens` on the completion). Cap per question = 512.

Global budget for replay: `B = max(1, n * 512 // 4)` → **2560** when n = 20. This is the CLI replay budget, not the v2 API cap (2000).

## 4. Prompt

User message (hashed into the cache key):

```
Answer with only the final answer. No explanation.

{question}
```

`prompt_hash` = SHA-256 hex of that template with `{question}` filled, UTF-8.

## 5. Scoring

1. Normalize gold and completion: strip, lowercase, collapse internal whitespace.
2. If the gold matches `^-?\d+(\.\d+)?$` after normalize, extract the **last** integer or decimal in the completion (regex) and compare numerically.
3. Otherwise require exact string equality after normalize.
4. Score is **1.0** or **0.0** (k = 1). Missing text or extract failure → 0.0.

## 6. Cache schema

`benchmarks/live/cache.json`:

```json
{
  "model": "claude-haiku-4-5-20251001",
  "prompt_template": "...",
  "entries": [
    {
      "question_id": "q01",
      "token_level": 64,
      "model": "claude-haiku-4-5-20251001",
      "prompt_hash": "...",
      "completion": "...",
      "score": 1.0,
      "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }
  ]
}
```

Key: `(question_id, token_level, model, prompt_hash)`. Hits never call the network.

Hard cap: if a collect would need **more than 100** new HTTP calls, abort before sending any.

## 7. Table-valued tasks

Each question → `Task` with `curve="table"`, `token_cost=512`, `value_table=((64, s64), (128, s128), ...)`.

`value_at(t)` = score of the largest level `≤ t`, else 0.

Replay runs `allocate_dp`, `allocate_greedy`, `allocate_random(..., seed=2026)` at budget B.

## 8. Offline rules

- pytest never reads `ANTHROPIC_API_KEY` and never opens sockets for collect.
- Public FastAPI `/allocate` does not collect or read the API key.
- Replay from cache needs **no** key.

## 9. Call budget

Worst case: 20 × 4 × 1 = **80** new calls (< 100 cap).
