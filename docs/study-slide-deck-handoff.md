# Study slide-deck handoff — LLM Token-Budget Allocator (v1)

**For:** Claude Design (or any slide-design model) building a study deck  
**Also for:** Ben, studying the project without opening every file  
**Repo:** https://github.com/BenJesus1/LLM-Token-Optimizer  
**HEAD at handoff:** `84f3d84` on `master`  
**Owner:** Ben  
**License:** MIT  

This document is the source of truth for a **study slide deck**. It is not the original build-plan handoff. It describes the **shipped v1**, how it actually works, which libraries it uses, and what Ben must be able to say in an interview.

---

## 0. Prompt to paste into Claude Design

Copy everything in the fenced block below into Claude Design, then attach this file (or paste the rest of the document after it).

```
You are designing a study slide deck for Ben, who built this project and must
re-explain it in intern interviews without looking at code.

Project: LLM Token-Budget Allocator (v1)
Repo: https://github.com/BenJesus1/LLM-Token-Optimizer

Goal of the deck:
- Help Ben memorize the stack, the two DP formulations, why greedy fails,
  the honest benchmark numbers, and the limitations.
- Not a sales pitch. Not a tutorial for beginners who never saw the repo.
- Dense enough to study from; sparse enough to present (one idea per slide).

Constraints:
- ~16–20 slides. Landscape 16:9.
- Academic / systems-portfolio tone. No startup hype, no “AI-powered” fluff.
- No live LLM APIs, no GPU/KV-cache, no bandits — those are explicitly out of
  scope. Never imply this product calls OpenAI or places GPU memory.
- Use the exact numbers and recurrences from the attached handoff. Do not
  invent results. Greedy matching DP at n=100 and n=1000 is a real finding;
  do not “fix” it by claiming greedy gets worse at scale.
- Charts to embed if available: benchmarks/charts/runtime_vs_task_count.png
  and benchmarks/charts/value_gap_vs_scale.png
- Color: muted, print-friendly. Dark text on light ground. One accent color
  for “exact DP”, a second for “greedy”, a third (muted) for “random”.
  No gradients, no emoji, no rainbow.
- Typography: one sans for titles (e.g. Inter / IBM Plex Sans), one
  monospace for recurrences and file names (e.g. IBM Plex Mono / JetBrains Mono).
- Every algorithm slide must include Big-O.
- Speaker notes on every slide: 3–6 bullets Ben can read aloud, in plain
  English, no code dump.

Follow the slide outline in section 2 of the attached handoff. Prefer
diagrams over paragraphs. Prefer the five-task 0/1 example and the two-task
log “spread vs dump” example as the two worked problems.

Output: a complete slide deck (title + body + speaker notes per slide).
```

---

## 1. Design brief

### Audience

Ben, studying for Summer 2027 AI internship interviews (Google, Meta, Anthropic, OpenAI, SpaceX). Secondary audience: a hiring manager skimming a printed leave-behind. Assume Ben wrote the code with an AI pair-programmer and now must **own** every design choice.

### What “efficient study” means

The deck should make these six things automatic:

1. One-sentence pitch.
2. Stack in 15 seconds (Python 3.11, pytest, matplotlib, stdlib argparse; nothing else).
3. 0/1 knapsack recurrence vs variable-token recurrence, and why they differ.
4. The five-task greedy-failure story (value 13 vs 12).
5. Honest results: token-level greedy matches DP at 100 and 1,000; ~1.6% gap at n=10; random ~30% short.
6. What this is **not** (live APIs, GPU memory, production service).

### Tone

Calm, precise, slightly academic. Like a short systems paper turned into slides. Prefer “simulated value” over “AI quality.” Prefer “token budget” over “cost router product.”

### Visual system (Claude Design must follow)

| Role | Use |
|---|---|
| Neutral ground | Off-white / paper (`#F7F5F0` or similar) |
| Text | Near-black (`#1A1A1A`) |
| Accent A — exact DP | Deep ink blue (`#1B4D6E`) |
| Accent B — greedy | Warm ochre (`#B56B2A`) |
| Accent C — random | Cool gray (`#6B7280`) |
| Danger / limitation | Muted brick (`#8B3A3A`) — only on limitation / “not this” slides |
| Code / recurrence | Monospace on a 4% gray panel, not a terminal-green theme |

Do not use neon, glassmorphism, or “AI neural net” clip art. Architecture diagrams should look like labeled boxes and arrows, not clouds.

### Assets in the repo (embed these, do not redraw as decoration)

- `benchmarks/charts/runtime_vs_task_count.png` — log-x runtime vs n
- `benchmarks/charts/value_gap_vs_scale.png` — % shortfall vs DP
- `tasks.json` — four-task sample used by the CLI demo

---

## 2. Suggested slide outline (build this)

Speaker-note bullets are what Ben should memorize. Slide titles are exact suggestions.

### Slide 1 — Title
**LLM Token-Budget Allocator**  
Subtitle: Exact DP vs greedy vs random, under a fixed token budget  
Footer: Ben · v1.0.0 · Python 3.11 · github.com/BenJesus1/LLM-Token-Optimizer

Speaker notes:
- Portfolio DS&A + AI-systems piece, not a hosted product.
- Built to be fully explainable in an interview.

### Slide 2 — One-line problem
A service has a **fixed token (or dollar) budget B** and a list of competing LLM tasks.  
Decide **how many integer tokens each task gets** to **maximize total simulated value**.

Speaker notes:
- Real-world rhyme: monthly API spend, which requests to fully serve vs throttle vs cheap-model.
- Values here are **mocked curves**, not live model scores.

### Slide 3 — What we are not building
Bullets only:
- No OpenAI / Anthropic API calls
- No GPU / KV-cache memory placement
- No multi-armed bandits / online learning
- No production deploy, auth, or multi-tenant service
- No task DAGs, latency, or SLOs

Speaker notes:
- Interviewers will try to grow the problem. Name the fence immediately.
- The constant curve is how 0/1 knapsack **lives inside** the variable-token DP — that is the clever reduction, not GPU allocation.

### Slide 4 — Stack (15 seconds)
Three columns:

**Runtime**
- Python **3.11+**
- Installable package `allocator` 0.1.0
- Console script: `allocator`

**Libraries we import**
- **pytest** ≥ 8 — tests
- **matplotlib** ≥ 3.8 — charts (Agg backend)
- **stdlib only** for solvers: `dataclasses`, `argparse`, `json`, `csv`, `math`, `random`, `pathlib`, `time`, `sys`

**Packaging / CI**
- `pyproject.toml` + setuptools (`src/` layout)
- GitHub Actions: Ubuntu, Python 3.11, `pip install -e .` then `pytest`
- Transitive: matplotlib pulls numpy; the project never imports numpy itself

Speaker notes:
- Deliberately tiny stack so every line is defensible.
- No FastAPI, no pandas, no LLM SDKs, no numpy in our code.

### Slide 5 — Repo map
```
src/allocator/     solvers + CLI
tests/             one file beside each unit
benchmarks/        generate → measure → plot
tasks.json         sample input
.github/workflows  pytest on every push
```

File → job (keep this table on the slide, small type is OK):

| File | Job |
|---|---|
| `task.py` | Frozen Task: id, cap, weight, curve |
| `value.py` | log / sqrt / constant curves |
| `allocation.py` | Assignment + Allocation results |
| `knapsack.py` | Exact **0/1** DP |
| `allocate.py` | Exact **variable-token** DP (the CLI solver) |
| `greedy.py` | 0/1 density greedy **and** token-level greedy |
| `random_alloc.py` | Seeded random tokens |
| `cli.py` + `tasks_io.py` | argparse + JSON/CSV load |

Speaker notes:
- CLI calls `allocate_dp` and `allocate_greedy`, **not** `knapsack_dp` / `knapsack_greedy`.
- Two greedy functions live in one file on purpose. Mixing them up in an interview is the #1 trap.

### Slide 6 — End-to-end data flow
Diagram (left to right):

`tasks.json` or `.csv` → `load_tasks` → `tuple[Task, ...]` → **solvers** → `Allocation` → stdout / `results.json` / PNG

CLI path (highlight):
1. `allocator run --budget 20 --tasks tasks.json`
2. Load tasks (JSON list or CSV rows: `id`, `token_cost`, `weight`, optional `curve`)
3. Time `allocate_dp`
4. Also run `allocate_greedy` for the comparison line
5. Print chosen `(id, tokens)`, DP value, DP runtime, greedy delta
6. Errors → **stderr**, exit **1**, no traceback

Speaker notes:
- Cold clone: `git clone` → `pip install -e .` → that one command.
- Sample budget 20 on `tasks.json` spends summarize 6, review 5, translate 4, classify 5; DP value ≈ 16.1086; greedy matches.

### Slide 7 — Task + value model
Task fields:
- `id: str`
- `token_cost: int` — **cap** (max tokens this task may receive)
- `weight: float` — scales the curve
- `curve: str` — `"log"` (default), `"sqrt"`, or `"constant"`

Curves:
- **log:** `weight * log(1 + t)` — diminishing, concave
- **sqrt:** `weight * sqrt(t)` — diminishing, concave
- **constant:** 0 until `t = cap`, then `weight` — 0/1 step

Speaker notes:
- `token_cost` is not “the cost if included” for the variable-token solver; it is a cap. For 0/1 knapsack it is both cost and cap.
- Constant curve is the reduction: variable DP recovers 0/1 knapsack.
- Values are never fetched from a model.

### Slide 8 — Two problems, two DPs
Split slide.

**Left — 0/1 knapsack (`knapsack_dp`)**  
Each task: take all `token_cost` tokens or skip.  
`dp[i][w] = max( skip, take if cost ≤ w )`  
Time **O(n · B)**

**Right — resource allocation (`allocate_dp`)**  
Each task: choose `t ∈ 0..min(cap, w)`  
`dp[i][w] = max_t  dp[i-1][w-t] + value_i(t)`  
Time **O(n · B · cap)**

Speaker notes:
- Same table shape (tasks × budget). Different inner loop.
- 0/1 inner decision is binary. Variable-token inner decision is “how many tokens.”
- Name: multiple-choice knapsack / separable resource allocation. **Not** unbounded knapsack (unbounded would allow taking the same task many times with no cap).

### Slide 9 — Why variable DP is optimal (the interview slide)
One recurrence, one English paragraph.

Recurrence:
```
dp[i][w] = max_{t = 0 .. min(cap_i, w)}  dp[i-1][w - t] + value_i(t)
dp[0][·] = 0
```

English:
Any feasible assignment of the first i tasks spends some t on task i and the rest on an **optimal** assignment of the earlier tasks into leftover budget. Those cases partition the search space, so the max is exact.

Implementation details to mention if asked:
- 2D table + parallel `choice[i][w]` so we can reconstruct tokens.
- Ties keep the **smaller t** so we do not spend tokens that do not raise value.
- Zero-token tasks are omitted from the printed `Allocation`.

Speaker notes:
- Optimal because of **optimal substructure + exhaustive partition**, not because “DP is always optimal.”
- Reconstruction walks backward: remaining budget, read `choice`, subtract t.

### Slide 10 — Worked example A: spread vs dump
Two identical log tasks, weight 1, cap 3, budget 3.  
`value(t) = log(1+t)`

| Assignment | Value |
|---|---|
| (3, 0) dump | log(4) ≈ 1.386 |
| (2, 1) spread | log(3)+log(2) ≈ 1.792 |

DP (and token-level greedy) pick **(2, 1)** because of the smaller-t tie-break on the later task.

Speaker notes:
- Concave curves reward spreading. That is why this is more than textbook 0/1 knapsack.
- Caps stop one task from eating the whole budget.

### Slide 11 — Worked example B: 0/1 greedy fails
Five constant tasks, budget 10:

| Task | Cost | Value | Density |
|---|---|---|---|
| draft | 4 | 5 | 1.25 |
| review | 3 | 4 | ~1.33 |
| cite | 2 | 3 | 1.5 |
| polish | 5 | 6 | 1.2 |
| translate | 6 | 7 | ~1.17 |

- **DP:** {review, cite, polish} cost 10 value **13**
- **0/1 greedy:** density order cite → review → draft, leftover 1, cannot fit polish. Value **12**

Speaker notes:
- Greedy is the fractional-knapsack density heuristic **without splitting**.
- A high-density prefix can leave a remainder that a slightly worse-density combo fills better.
- This is where greedy actually breaks. Not at n=1000 on the mixed-curve synthetic sets.

### Slide 12 — The three token-level solvers (CLI / benchmarks)

| Solver | What it does | Time | Role |
|---|---|---|---|
| `allocate_dp` | Exact max over integer token vectors | O(n·B·cap) | Ground truth |
| `allocate_greedy` | Repeatedly take the highest-density **increment** | O(n·B) | Fast baseline |
| `allocate_random` | One token at a time to a random task with cap left | O(n·B) | Dumb baseline |

Token-level greedy increment:
- log / sqrt: +1 token (marginal gain / 1)
- constant: **jump to full cap** (unit steps in between have zero gain)

Speaker notes:
- Interview trap: “greedy” means two different functions.
- `knapsack_greedy` is 0/1 whole-task fill, O(n log n), and **is** suboptimal on the five-task instance.
- `allocate_greedy` often matches DP on concave mixed instances (see results).

### Slide 13 — Benchmarks: how numbers were made
- Scales: **10 / 100 / 1,000** tasks
- Seed **2026**; each scale uses `seed + n` so the three sets are independent
- Caps uniform 1..20, weights 0.5..5.0, curves chosen from {constant, log, sqrt}
- Budget = **one quarter of total cap** (so not everything can be fully served)
- One timed run per cell (`time.perf_counter`), written to `benchmarks/results.json`
- pytest **never** runs n=1000 (too slow for CI). Full suite: `python -m benchmarks.measure`

Speaker notes:
- Reproducible from a cold clone. Same seed → same tasks.
- Charts are committed PNGs, not a local-only matplotlib window.

### Slide 14 — Results table (honest)

| n | Budget | DP value | Greedy value | Random value | DP time | Greedy time |
|---|---|---|---|---|---|---|
| 10 | 21 | 29.95 | 29.49 | 19.80 | ~1 ms | ~0.07 ms |
| 100 | 244 | 342.35 | 342.35 | 245.24 | ~31 ms | ~6 ms |
| 1,000 | 2,716 | 3678.74 | 3678.74 | 2575.71 | ~3.9 s | ~0.66 s |

Callouts (use accent colors):
- Token-level greedy **matches DP at 100 and 1,000**
- Gap at n=10 is ~**1.6%** of DP value
- Random stays ~**30%** short at every scale
- DP is slower, as O(n·B·cap) predicts — that is what the log-x runtime chart is for

Speaker notes:
- Do **not** claim greedy collapses as n grows. On these concave-heavy draws it does not.
- Quality failure of greedy is the **0/1 hand instance**, not the scale sweep.
- Random existing is what makes “greedy is pretty good” a comparison, not a vibe.

### Slide 15 — Charts
Embed both PNGs. Captions:
- Runtime vs task count (log x). Source: `results.json`, seed 2026, one run per cell.
- Value shortfall vs DP (%). Same source. (DP − solver) / DP.

Speaker notes:
- Log x because 10 / 100 / 1000 would squash the left side on a linear axis.
- The runtime chart is the complexity slide. The gap chart is the quality slide.

### Slide 16 — Tests as the correctness story
~86 pytest tests. Philosophy: **hand-compute the answer on paper, then assert**.

Must-mention cases:
- Five-task 0/1: DP 13, greedy 12
- Two log tasks: spread beats dump
- Constant curve: `allocate_dp` recovers `knapsack_dp`
- Edges: zero budget, empty list, single task over budget, ties, unknown curve, bad JSON
- CLI: errors on stderr, exit 1, no traceback
- Charts exist in the repo (not only generated in memory)

Speaker notes:
- Tests sit **beside** each unit (`test_allocate.py` next to `allocate.py` conceptually).
- CI is the boring proof the clone works.

### Slide 17 — Limitations (say these first if the interviewer pushes)
- Independent tasks; no dependencies
- Tokens + simulated value only; no latency / SLO / queues
- Curves and weights known up front, not learned
- No live model quality
- Not a production service
- Not GPU or KV-cache allocation

Speaker notes:
- Known-up-front value is why this is DP, not a bandit.
- Stretch (not v1): dashboard, bandits, mocked multi-model cost router.

### Slide 18 — Interview close / 30-second recap
1. Problem: integer tokens under budget B, maximize simulated value.
2. Exact method: DP over tasks × leftover budget; inner max over t.
3. Faster method: token-level greedy; 0/1 density greedy is a different, weaker story.
4. Evidence: seeded 10/100/1000; greedy matches at scale here; random does not; 0/1 greedy loses 13 vs 12.
5. Scope: no APIs, no GPUs, tiny Python stack.

Optional last line: “I can work the five-task instance on the whiteboard.”

---

## 3. Stack and libraries (complete)

### What we chose to depend on

Declared in `pyproject.toml`:

```
requires-python = ">=3.11"
dependencies = [
    "matplotlib>=3.8",
    "pytest>=8.0",
]
build-system: setuptools>=68
```

That is the **entire** third-party surface the project admits.

### Standard library used in first-party code

| Module | Where | Why |
|---|---|---|
| `dataclasses` | Task, Assignment, Allocation, SyntheticSet | Frozen records |
| `math` | `value.py` | `log`, `sqrt` |
| `argparse` | `cli.py` | `allocator run --budget --tasks` |
| `json` | `tasks_io.py`, `cli.py`, `measure.py` | load tasks + results |
| `csv` | `tasks_io.py` | CSV task lists |
| `pathlib` | CLI, IO, benchmarks | files |
| `time` | `cli.py`, `measure.py` | `perf_counter` |
| `sys` | `cli.py` | stderr + exit |
| `random` | `random_alloc.py`, `generate.py` | `random.Random(seed)` |
| `collections.abc` | typing for Sequence / Mapping / Callable | |
| `typing` | `measure.py`, `plot.py` | `Any` in JSON rows |

### Libraries we explicitly do **not** use

No `openai`, `anthropic`, `numpy` (direct), `pandas`, `scipy`, `fastapi`, `flask`, `click`, `pydantic`, `torch`, `redis`, Docker, cloud SDKs.

matplotlib **does** depend on numpy internally. If asked “do you use numpy?”: “Not in our modules. Matplotlib is the only charting dependency; numpy comes along for the ride.”

### How a clone runs

```
pip install -e .
allocator run --budget 20 --tasks tasks.json
python -m pytest
python -m benchmarks.measure    # writes results.json; slow at n=1000
python -m benchmarks.plot       # writes the two PNGs from results.json
```

Entry point: `[project.scripts] allocator = "allocator.cli:main"`

### CI

`.github/workflows/ci.yml` — on push and pull_request, `ubuntu-latest`, Python 3.11, pip cache, `pip install -e .`, `python -m pytest`. The 1,000-task DP is **not** in CI.

---

## 4. How the project works in its entirety

### 4.1 Purpose

A **token-budget allocator**: given budget B and n tasks, assign a non-negative integer token count to each task, never exceeding that task’s cap, never exceeding B in total, maximizing the sum of simulated values.

It exists as a **portfolio DS&A artifact** that rhymes with real LLM cost-routing (which requests to fully serve, shorten, or send to a cheaper model) without billing a real API.

### 4.2 Domain objects

**Task** (`task.py`) — frozen dataclass. Validates non-empty id, non-negative cost and weight, known curve name. `value_at(t)` is the only place constant-curve 0/1 stepping is applied (needs the cap). `Task.constant(...)` is sugar for 0/1 examples. Property `value` is value at full cap (0/1 include).

**Assignment** — one task plus the tokens it received.  
**Allocation** — tuple of assignments in **input order**, **omitting zeros**. Totals: `total_cost`, `total_value`. Lookup: `tokens_for(id)`.

Helpers: `allocation_from_tasks` (0/1: every selected task gets its full cap) and `allocation_from_counts` (variable-token vector).

### 4.3 Value curves (`value.py`)

Named functions in `CURVES`. `evaluate_curve` dispatches by name. Direct `constant_curve` is “weight if t>0 else 0”; **Task.value_at overrides that** so constant means “pay the full cap or get nothing.” If a slide mentions constant, use the Task definition, not the raw function.

Diminishing returns (log, sqrt): first extra token is worth more than the 100th. That is why spreading can beat dumping.

### 4.4 Solver family

There are **five** public solvers. Group them as two problems.

**Problem A — 0/1 include/exclude** (classic knapsack)

- `knapsack_dp` — exact. Table `dp[i][w]`. Recurrence skip vs take. Reconstructs by walking backward; **skip on tie** so zero-value junk is not taken. O(n·B) time and memory (2D on purpose, for reconstruction).
- `knapsack_greedy` — sort by value/token (∞ if cost 0 and value > 0), take whole tasks that fit. O(n log n). Can be suboptimal.

**Problem B — integer tokens per task** (what the CLI and benchmarks use)

- `allocate_dp` — exact. For each task i and budget w, try every t from 0 to min(cap, w). Store best t in `choice`. Reconstruct backward. Ties keep smaller t. O(n·B·cap).
- `allocate_greedy` — while budget remains, scan all tasks for the best increment (density, dt). Apply it. Constant tasks jump `dt = remaining cap`. O(n·B).
- `allocate_random` — required `seed`. Uniform among tasks still under cap. One token per step.

**Invariant every solver must keep:** total tokens ≤ B, per-task tokens ≤ cap, no negative tokens. Greedy is allowed to lose on value; it is not allowed to overspend.

### 4.5 Why two DPs exist

Phase 1 of the project was textbook 0/1 knapsack. Phase 2 replaced fixed values with curves, which is meaningless unless a task can receive **part** of its cap. `allocate_dp` is that generalization. The constant curve proves the generalization still contains 0/1: tests assert `allocate_dp` matches `knapsack_dp` on the five-task instance, and selected tasks receive exactly `token_cost`.

### 4.6 I/O and CLI

`load_tasks(path)`:
- `.json` — must be a **list of objects**
- `.csv` — header row, DictReader
- required fields: `id`, `token_cost`, `weight`
- `curve` optional, default `log`
- anything else: ValueError

CLI (`allocator run --budget N --tasks FILE`):
- required subcommand `run`
- times only DP
- always also runs token-level greedy for the summary line
- success: stdout allocation + “Total value (DP)” + “Runtime (DP)” + greedy comparison
- failure: `error: ...` on stderr, exit 1 (missing file, bad JSON, empty list, negative budget, unknown curve, …)

Sample `tasks.json`:

```json
[
  {"id": "summarize", "token_cost": 12, "weight": 2.5, "curve": "log"},
  {"id": "review",    "token_cost": 8,  "weight": 1.8, "curve": "sqrt"},
  {"id": "translate", "token_cost": 10, "weight": 2.0, "curve": "log"},
  {"id": "classify",  "token_cost": 5,  "weight": 4.0, "curve": "constant"}
]
```

Budget 20: summarize 6, review 5, translate 4, classify 5, value ≈ 16.1086, greedy matches.

### 4.7 Benchmark pipeline

1. **generate** — `generate_tasks(n, seed=...)` then budget = max(1, total_cap // 4). Suite: n in (10, 100, 1000), seed `2026 + n`.
2. **measure** — run dp, greedy, random; record n, budget, seed, solver, runtime_s, total_value, total_cost. Save JSON.
3. **plot** — matplotlib Agg (no GUI). Two figures, labeled axes, legend, caption with seed. Does **not** re-run n=1000.

Published cells (from committed `results.json`):

| n | budget | instance seed | dp value | greedy value | random value | dp s | greedy s | random s |
|---|---|---|---|---|---|---|---|---|
| 10 | 21 | 2036 | 29.9516 | 29.4865 | 19.8002 | 0.00100 | 0.000072 | 0.000028 |
| 100 | 244 | 2126 | 342.3533 | 342.3533 | 245.2363 | 0.03120 | 0.00550 | 0.00066 |
| 1000 | 2716 | 3026 | 3678.7430 | 3678.7430 | 2575.7075 | 3.9075 | 0.6590 | 0.0621 |

Runtimes are one-machine wall times; values are the numbers to treat as canonical. README rounds values to 2 decimals and times to human units — slides may use either, but do not mix a rounded value with a fake extra digit.

### 4.8 Tests

Beside each function, not a late dump. Hand-computed comments **before** asserts. pytest.ini: `testpaths = tests`, `pythonpath = src` and `.` so both `allocator` and `benchmarks` import.

### 4.9 Git history (why it looks like this)

Incremental commits by phase, not one dump. That is intentional for the portfolio. Headlines, newest last in time (oldest first below):

1. README skeleton, license, gitignore  
2. src / tests / benchmarks packages  
3. Pin Python 3.11 + pytest + matplotlib  
4. GitHub Actions CI  
5. Task record  
6. 0/1 knapsack DP  
7. 0/1 greedy  
8. Shared solver edge cases  
9. Value curves  
10. Variable-token DP  
11. Re-verify three-task hand instance  
12. Seeded generator  
13. Measure 10/100/1000  
14. Runtime plot  
15. Value-gap plot  
16. Commit the PNGs  
17. argparse CLI + tasks.json  
18. CLI summary vs token-level greedy  
19. CLI helper docs  
20. Fail CLI on bad input without traceback  
21. README research write-up  

HEAD message: *Write the README as a research artifact so an interviewer can recast the DP, the greedy baseline, and the real (not invented) benchmark gaps.*

---

## 5. Plain-language study cards (Ben)

Use these as speaker-note sources. No code.

### What is dynamic programming here?

Break the hard problem into overlapping subproblems: “best value using the first i tasks with leftover budget w.” Solve small i and small w first, store answers in a table, never recompute. The final cell `dp[n][B]` is the answer. Then walk backward to see **which** tokens produced it.

### Why is the table optimal?

Because every legal way to use task i is some token count t, and after choosing t the leftover problem is **exactly** the already-solved subproblem on the earlier tasks. We try all t, so we cannot miss a better split.

### When does 0/1 greedy fail?

When taking the “best bang for the buck” items leaves a hole that a slightly worse combination would have filled. Remainder waste. Fractional knapsack would split the next item; we cannot, so leftover budget dies.

### When does token-level greedy work well?

On smooth diminishing (concave) curves, the next-best extra token is a good local rule, and empirically it tied DP at n=100 and n=1,000 on our mixed synthetic sets. It is still a heuristic. We keep DP as the proof of optimality.

### Why random?

To show “faster than DP” is not enough. Random is fast and about 30% worse. Greedy is also fast and, here, nearly exact. The interesting gap is **quality vs speed**, not speed alone.

### Complexity in one breath

“0/1 DP is tasks times budget. Variable-token DP multiplies by the cap because each task has many choices of t. Token-level greedy is tasks times budget with no third loop. 0/1 greedy is a sort.”

---

## 6. Diagrams Claude Design should draw (not screenshots)

1. **Budget bar** — a rectangle of width B, chunks labeled with task ids and token counts (CLI sample or five-task 0/1).
2. **DP table cartoon** — rows = after task i, columns = leftover w, arrows “from (i-1, w-t)”. Do not fill with real numbers except on the three-task log example if space allows (see `tests/test_allocate.py` comments for a filled row).
3. **Greedy remainder trap** — density-sorted items dropping into a bin of size 10, leftover 1, vs the 13-value packing.
4. **Pipeline** — generate → measure → results.json → plot → README.
5. **Stack stack** — three layers: CPython 3.11 / our package / pytest+matplotlib. Tiny on purpose.

Do not draw GPUs, transformers, or network globes.

---

## 7. Copy constraints (do not hallucinate)

Safe claims:
- Exact DP for integer token allocation under a cap.
- 0/1 knapsack as the constant-curve special case.
- Token-level greedy matched DP at 100 and 1,000 on seed-2026 mixed curves.
- 0/1 greedy is suboptimal on the published five-task instance (13 vs 12).
- Random ~30% short.
- Python 3.11, pytest, matplotlib, argparse CLI.
- MIT license, GitHub public repo.

Unsafe / forbidden claims:
- “Calls GPT / measures real quality”
- “Allocates KV-cache / GPU HBM”
- “Learns values online / bandits”
- “Greedy gets worse as n grows” (false on our data)
- “Production cost router used at company X”
- Extra libraries (pandas, numpy-as-ours, FastAPI, Docker)
- Test count other than “on the order of 80+”; prefer “the unit tests include hand-computed cases” if unsure
- Runtime numbers as if they were scientific averages of many trials — they are **one run per cell**

---

## 8. Quick facts box (put on a cheat-sheet slide or appendix)

| Item | Value |
|---|---|
| Package | `allocator` 0.1.0 |
| Python | ≥ 3.11 |
| Direct deps | pytest, matplotlib |
| CLI | `allocator run --budget N --tasks FILE` |
| Exact CLI solver | `allocate_dp` |
| CLI comparison | `allocate_greedy` (token-level) |
| 0/1 exact | `knapsack_dp` |
| 0/1 greedy | `knapsack_greedy` |
| Variable DP time | O(n · B · cap) |
| 0/1 DP time | O(n · B) |
| Token greedy time | O(n · B) |
| 0/1 greedy time | O(n log n) |
| Curves | log, sqrt, constant |
| Benchmark seed | 2026 |
| Scales | 10, 100, 1000 |
| Budget rule | total_cap // 4 |
| Five-task 0/1 | DP 13, greedy 12 |
| Spread vs dump | log(3)+log(2) > log(4) |
| License | MIT |
| Repo | https://github.com/BenJesus1/LLM-Token-Optimizer |

---

## 9. File checklist for the designer

If files can be attached to the design session, attach:

1. This handoff  
2. `README.md`  
3. `benchmarks/charts/runtime_vs_task_count.png`  
4. `benchmarks/charts/value_gap_vs_scale.png`  
5. `tasks.json`  
6. Optional: `src/allocator/allocate.py` and `src/allocator/knapsack.py` for recurrence fidelity only — do not paste long code onto slides  

Ben studies from the **speaker notes**, not from screenshots of source.
