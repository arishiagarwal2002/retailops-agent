"""Per-key rate limiting (slowapi) and a hard daily LLM spend cap enforced in code."""
from __future__ import annotations

import datetime as dt

from fastapi import HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.logging import log_event
from app.db.models import Run
from app.db.session import rw_session


def _rate_key(request: Request) -> str:
    return request.headers.get("x-api-key") or get_remote_address(request)


limiter = Limiter(key_func=_rate_key)


def spent_today_gbp() -> float:
    start = dt.datetime.now(dt.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    with rw_session() as s:
        total = s.execute(select(func.coalesce(func.sum(Run.cost_gbp), 0.0)).where(Run.created_at >= start)).scalar()
    return float(total or 0.0)


def enforce_spend_cap() -> None:
    cap = get_settings().daily_spend_cap_gbp
    spent = spent_today_gbp()
    if spent >= cap:
        log_event("spend_cap_hit", spent_gbp=round(spent, 4), cap_gbp=cap)
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"daily LLM spend cap of £{cap:.2f} reached; try again tomorrow (UTC)",
        )
