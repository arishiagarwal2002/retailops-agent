"""Uniform JSON error responses that carry the trace id."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.core.logging import log_event, trace_id_var


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RateLimitExceeded)
    async def _rate_limited(_: Request, exc: RateLimitExceeded):
        return JSONResponse(status_code=429, content={"error": f"rate limit exceeded: {exc.detail}",
                                                      "trace_id": trace_id_var.get()})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log_event("unhandled_error", error=repr(exc))
        return JSONResponse(status_code=500, content={"error": "internal error", "trace_id": trace_id_var.get()})
