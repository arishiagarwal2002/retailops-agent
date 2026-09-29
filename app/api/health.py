"""GET /health - liveness plus database, data and model checks."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import ro_engine, rw_engine

router = APIRouter(tags=["health"])


def _checks() -> dict:
    checks: dict = {}
    try:
        with rw_engine().connect() as c:
            checks["policy_chunks"] = c.execute(text("SELECT count(*) FROM policy_chunks")).scalar()
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {type(e).__name__}"
    try:
        with ro_engine().connect() as c:
            checks["products"] = c.execute(text("SELECT count(*) FROM products")).scalar()
        checks["read_only_role"] = "ok"
    except Exception as e:
        checks["read_only_role"] = f"error: {type(e).__name__}"
    try:
        from tools.forecast import load_models

        models, meta = load_models()
        checks["forecast_models"] = f"ok ({len(models)} quantiles, origin {meta['last_complete_week']})"
    except Exception as e:
        checks["forecast_models"] = f"error: {e}"
    s = get_settings()
    checks["llm"] = (f"configured ({s.llm_provider}: {s.default_model})" if s.llm_key_configured
                     else f"not configured (provider {s.llm_provider}: set its API key in .env)")
    return checks


@router.get("/health")
async def health():
    checks = await asyncio.to_thread(_checks)
    ok = all(not str(v).startswith("error") for v in checks.values())  # a missing LLM key is reported, not fatal
    return JSONResponse(status_code=200 if ok else 503, content={"status": "ok" if ok else "degraded", **checks})


@router.get("/", include_in_schema=False)
async def index():
    """Landing route. Without it, opening the base URL in a browser returns a bare 404."""
    return {
        "service": "RetailOps Agent",
        "description": "Tool-using LLM agent for retail sales, inventory and returns analysis.",
        "docs": "/docs",
        "health": "/health",
        "demo_ui": "the Streamlit app (default http://localhost:8501)",
        "endpoints": {
            "POST /v1/ask": "ask a question synchronously (X-API-Key header required)",
            "POST /v1/runs": "ask asynchronously, returns a run_id",
            "GET /v1/runs/{id}": "status, answer and full tool trace",
            "GET /v1/runs": "recent runs (audit log)",
            "GET /v1/approvals": "pending actions awaiting human approval",
            "POST /v1/approvals/{id}": "approve or reject a pending action",
        },
    }
