"""POST /v1/ask - synchronous question -> answer."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request

from agent.runner import run_agent
from agent.state import AgentConfig
from app.api.schemas import AskRequest, RunOut
from app.api.service import cache_get, cache_put, load_run
from app.core.auth import Principal, require_api_key
from app.core.config import get_settings
from app.core.limits import enforce_spend_cap, limiter

router = APIRouter(prefix="/v1", tags=["ask"])


@router.post("/ask", response_model=RunOut)
@limiter.limit(lambda: get_settings().rate_limit)
async def ask(request: Request, body: AskRequest, principal: Principal = Depends(require_api_key)) -> RunOut:
    s = get_settings()
    model = (body.model if s.llm_provider == "anthropic" else None) or s.default_model
    if body.use_cache and (hit := await asyncio.to_thread(cache_get, body.question, model)):
        return hit
    await asyncio.to_thread(enforce_spend_cap)
    result = await asyncio.to_thread(run_agent, body.question, None, AgentConfig(model=model), None, principal.key_id)
    out = await asyncio.to_thread(load_run, result.run_id)
    if out is None:
        raise HTTPException(500, "run was not recorded")
    # Key the cache by the model that actually answered, and never write when caching is off.
    if body.use_cache and out.model == model:
        await asyncio.to_thread(cache_put, body.question, model, out)
    return out
