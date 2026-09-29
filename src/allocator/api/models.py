"""Pydantic request/response models. Domain rules stay on ``Task`` / solvers."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaskIn(BaseModel):
    """One competing task. Mirrors ``allocator.task.Task`` fields."""

    id: str
    token_cost: int
    weight: float
    curve: str = "log"


class AllocateRequest(BaseModel):
    """Body of ``POST /allocate``."""

    budget: int
    tasks: list[TaskIn]


class AssignmentOut(BaseModel):
    """One DP assignment with simulated value."""

    id: str
    tokens: int
    value: float


class GreedyComparison(BaseModel):
    """Token-level greedy totals vs DP (same comparison the CLI prints)."""

    total_value: float
    total_cost: int
    delta_vs_dp: float
    delta_vs_dp_pct: float | None = Field(
        default=None,
        description="Percent of greedy value; null when greedy value is 0",
    )


class AllocateResponse(BaseModel):
    """DP allocation plus greedy comparison."""

    assignments: list[AssignmentOut]
    total_cost: int
    total_value: float
    dp_runtime_s: float
    greedy: GreedyComparison
    solvers: list[str]


class HealthResponse(BaseModel):
    """Deploy probe."""

    status: str


class PredictedQuestionIn(BaseModel):
    """One v4 question. Predictor uses text/gold features only."""

    id: str
    question: str
    gold: str


class AllocatePredictedRequest(BaseModel):
    """Body of local ``POST /allocate-predicted``. Budget is cost units."""

    budget: int
    questions: list[PredictedQuestionIn]


class RouteRequest(BaseModel):
    """Body of local ``POST /route``. Prompt text only."""

    question: str


class RouteResponse(BaseModel):
    """Chosen Claude id plus the Haiku probability and threshold used."""

    model: str
    p_haiku: float
    tau: float
    reason: str
