# Project Handoff: LLM Token-Budget Allocator — v5 (One-prompt model router)

**For:** Cursor
**Owner:** Ben
**Builds on:** v1–v4. v4 allocated Haiku token rungs on a batch. v5 **picks a model per prompt**.

> Public `/allocate`, pytest, and CI stay offline. Two-model collect is opt-in (`collect_v5 --live`).

---

## 1. Why

A useful product is: *this prompt → Haiku or Sonnet*, maximizing expected match per dollar — not another 64-vs-256 Haiku table.

| Goal | How v5 delivers it |
|---|---|
| One prompt in | `choose_model(question)` from prompt-only P(Haiku ok) |
| Quality can beat cheap | Collect both models at 256 tokens, show-work prompt |
| Honest eval | Held-out realized score and cost vs always-Haiku / always-Sonnet |
| No surprise bills | `/route` local-only; no live complete on that path |

**Target effort:** ~16 hours.

---

## 2. One-liner

Fit P(Haiku correct) on 12 train prompts, threshold `tau` on train cost-aware utility, route 8 test prompts, score from cache (no extra API).

---

## 3. Pins

- Cheap: `claude-haiku-4-5-20251001`
- Quality: `claude-sonnet-4-5-20250929`
- n=20, k=1, T=0, max_tokens=256, 12/8 split, λ=0.15, costs 1 and 6, cap 80 calls

---

## 4. Out of scope

Opus / 3+ models; live `/route` completions; monthly DP batch; rewriting `allocate_dp`; public live API.

---

## 5. Definition of done

- [x] Requirements written before two-model collect
- [x] Fixture replay green in pytest
- [x] Live `cache_v5.json` or fixture + `v5_results.json`
- [x] README: router vs always-Haiku vs always-quality (honest if loss)
- [x] Public `/allocate` still offline
