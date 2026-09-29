"""FastAPI app: router registration, middleware, startup warm-up.

    uvicorn app.main:app --reload
"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from slowapi.middleware import SlowAPIMiddleware

from app.api import approvals, ask, health, runs
from app.core.config import get_settings
from app.core.errors import install_error_handlers
from app.core.limits import limiter
from app.core.logging import configure_logging, log_event, trace_id_var


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging(get_settings().log_level)

    def _warm() -> None:
        try:
            from tools.forecast import load_models

            load_models()
        except Exception as e:
            log_event("forecast_models_unavailable", error=str(e))

    await asyncio.to_thread(_warm)
    log_event("startup", provider=get_settings().llm_provider, model=get_settings().default_model)
    yield


app = FastAPI(
    title="RetailOps Agent",
    version="1.0.0",
    description="Tool-using LLM agent for retail sales, inventory and returns analysis.",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
install_error_handlers(app)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    trace_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    trace_id_var.set(trace_id)
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    log_event("http_request", method=request.method, path=request.url.path, status=response.status_code,
              latency_ms=int((time.perf_counter() - started) * 1000))
    return response


app.include_router(health.router)
app.include_router(ask.router)
app.include_router(runs.router)
app.include_router(approvals.router)
