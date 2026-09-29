"""Pinned live-eval constants. Keep in sync with docs/live-eval-requirements.md and docs/v4-requirements.md."""

MODEL = "claude-haiku-4-5-20251001"
TOKEN_LEVELS: tuple[int, ...] = (64, 128, 256, 512)
MAX_NEW_CALLS = 100
PROMPT_TEMPLATE = "Answer with only the final answer. No explanation.\n\n{question}"
API_URL = "https://api.anthropic.com/v1/messages"
RANDOM_SEED = 2026

# v4 (predict-then-allocate). Do not change v3 TOKEN_LEVELS / MAX_NEW_CALLS / temperature 0.
V4_TOKEN_LEVELS: tuple[int, ...] = (64, 256)
V4_K = 3
V4_TEMPERATURE = 0.5
MAX_NEW_CALLS_V4 = 150
V4_TRAIN_N = 6
V4_TEST_N = 4
COST_HAIKU_64 = 1
COST_HAIKU_256 = 4
V4_LOCAL_ENV = "ALLOCATOR_V4_LOCAL"
V4_CACHE_ENV = "ALLOCATOR_V4_CACHE"
V4_QUESTIONS_ENV = "ALLOCATOR_V4_QUESTIONS"
# Used only for the harder v4 cache so 64 tokens can truncate before the answer.
V4_WORK_PROMPT_TEMPLATE = (
    "Solve step by step. Write every calculation. "
    "Do not put the final answer until the last line. "
    "On the last line write: Answer: <number>\n\n{question}"
)


def global_budget(n: int, *, cap: int = TOKEN_LEVELS[-1]) -> int:
    """Quarter of n times the per-question cap (same rule as v1 generate)."""

    return max(1, n * cap // 4) if n else 0


def render_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


# v5 one-prompt router. Do not change v3/v4 collect defaults.
MODEL_CHEAP = MODEL
MODEL_QUALITY = "claude-sonnet-4-5-20250929"
V5_MODELS: tuple[str, ...] = (MODEL_CHEAP, MODEL_QUALITY)
V5_MAX_TOKENS = 256
V5_K = 1
V5_TEMPERATURE = 0.0
MAX_NEW_CALLS_V5 = 80
V5_TRAIN_N = 12
V5_TEST_N = 8
COST_HAIKU_CALL = 1
COST_QUALITY_CALL = 6
V5_LAMBDA = 0.15
V5_LOCAL_ENV = "ALLOCATOR_V5_LOCAL"
V5_CACHE_ENV = "ALLOCATOR_V5_CACHE"
V5_QUESTIONS_ENV = "ALLOCATOR_V5_QUESTIONS"
