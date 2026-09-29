"""FastAPI app: ``POST /allocate`` and ``GET /health``."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from allocator.live.config import V4_LOCAL_ENV, V5_LOCAL_ENV

from .models import (
    AllocatePredictedRequest,
    AllocateRequest,
    AllocateResponse,
    HealthResponse,
    RouteRequest,
    RouteResponse,
)
from .service import SOLVER_NAMES, run_allocate

logger = logging.getLogger("allocator.api")


def _log_line(
    *,
    task_count: int | None,
    budget: int | None,
    solvers: list[str] | None,
    latency_s: float,
    status_code: int,
) -> None:
    """One structured JSON line per request (stdout via the logging config)."""

    logger.info(
        json.dumps(
            {
                "task_count": task_count,
                "budget": budget,
                "solvers": solvers,
                "latency_s": latency_s,
                "status_code": status_code,
            }
        )
    )


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Time every request; body-derived fields are filled on /allocate."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        if request.url.path == "/health":
            return response
        extra = getattr(request.state, "allocate_log", None)
        task_count = extra["task_count"] if extra else None
        budget = extra["budget"] if extra else None
        solvers = extra["solvers"] if extra else None
        _log_line(
            task_count=task_count,
            budget=budget,
            solvers=solvers,
            latency_s=time.perf_counter() - started,
            status_code=response.status_code,
        )
        return response


def create_app() -> FastAPI:
    """Build the public demo app."""

    logging.basicConfig(level=logging.INFO)
    application = FastAPI(
        title="LLM Token-Budget Allocator",
        version="0.5.0",
        description="Exact DP plus token-level greedy over a shared token budget.",
    )
    application.add_middleware(RequestLogMiddleware)

    @application.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request.state.allocate_log = {
            "task_count": None,
            "budget": None,
            "solvers": None,
        }
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Deploy probe. Does not run a solver."""

        return HealthResponse(status="ok")

    @application.post("/allocate", response_model=AllocateResponse)
    def allocate(request: Request, body: AllocateRequest) -> AllocateResponse:
        """Allocate a token budget using ``allocate_dp`` and ``allocate_greedy``."""

        request.state.allocate_log = {
            "task_count": len(body.tasks),
            "budget": body.budget,
            "solvers": list(SOLVER_NAMES),
        }
        try:
            return run_allocate(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if os.environ.get(V4_LOCAL_ENV) == "1":
        from .service import run_allocate_predicted

        @application.post("/allocate-predicted", response_model=AllocateResponse)
        def allocate_predicted(
            request: Request, body: AllocatePredictedRequest
        ) -> AllocateResponse:
            """Local-only: DP on predicted envelopes. Does not call Anthropic."""

            request.state.allocate_log = {
                "task_count": len(body.questions),
                "budget": body.budget,
                "solvers": list(SOLVER_NAMES),
            }
            try:
                return run_allocate_predicted(body)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    if os.environ.get(V5_LOCAL_ENV) == "1":
        from .service import run_route

        @application.post("/route", response_model=RouteResponse)
        def route(request: Request, body: RouteRequest) -> RouteResponse:
            """Local-only: pick Haiku or quality from prompt text. No Anthropic."""

            request.state.allocate_log = {
                "task_count": 1,
                "budget": None,
                "solvers": ["route"],
            }
            try:
                return run_route(body)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    return application


app = create_app()
