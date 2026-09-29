"""Asynchronous runs: POST returns a run_id immediately; GET polls status, answer and trace."""
from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select

from agent.runner import create_run_row, run_agent
from agent.state import AgentConfig
from app.api.schemas import AskRequest, RunAccepted, RunList, RunOut
from app.api.service import load_run
from app.core.auth import Principal, require_api_key
from app.core.config import get_settings
from app.core.limits import enforce_spend_cap, limiter
from app.core.logging import log_event
from app.db.models import Run
from app.db.session import rw_session

router = APIRouter(prefix="/v1", tags=["runs"])

# Hold references so background tasks are not garbage-collected mid-run.
_TASKS: set[asyncio.Task] = set()


@router.post("/runs", response_model=RunAccepted, status_code=status.HTTP_202_ACCEPTED)
@limiter.limit(lambda: get_settings().rate_limit)
async def start_run(request: Request, body: AskRequest, principal: Principal = Depends(require_api_key)) -> RunAccepted:
    await asyncio.to_thread(enforce_spend_cap)
    s = get_settings()
    model = (body.model if s.llm_provider == "anthropic" else None) or s.default_model
    run_id = str(uuid.uuid4())
    # Insert the row before returning so an immediate GET finds it.
    await asyncio.to_thread(create_run_row, run_id, body.question, model, principal.key_id)

    async def _go() -> None:
        try:
            await asyncio.to_thread(run_agent, body.question, None, AgentConfig(model=model), run_id,
                                    principal.key_id, False)
        except Exception as e:  # run_agent records its own failures; this is a last resort
            log_event("async_run_crashed", run_id=run_id, error=repr(e))

    task = asyncio.create_task(_go())
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return RunAccepted(run_id=run_id, poll=f"/v1/runs/{run_id}")


@router.get("/runs/{run_id}", response_model=RunOut)
async def get_run(run_id: str, _: Principal = Depends(require_api_key)) -> RunOut:
    out = await asyncio.to_thread(load_run, run_id)
    if out is None:
        raise HTTPException(404, "run not found")
    return out


@router.get("/runs", response_model=RunList)
async def list_runs(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                    _: Principal = Depends(require_api_key)) -> RunList:
    def _list() -> list[RunOut]:
        with rw_session() as s:
            ids = s.execute(select(Run.run_id).order_by(Run.created_at.desc()).limit(limit).offset(offset)).scalars().all()
        return [load_run(i, include_trace=False) for i in ids]

    return RunList(items=await asyncio.to_thread(_list), limit=limit, offset=offset)
