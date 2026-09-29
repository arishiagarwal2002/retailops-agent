"""Shared helpers for the API routers: loading runs, response cache."""
from __future__ import annotations

import datetime as dt
import hashlib

from sqlalchemy import delete, select

from agent.verifier import CITATION_RE, normalize
from app.api.schemas import PendingActionOut, RunOut
from app.core.config import get_settings
from app.db.models import PendingAction, ResponseCache, Run
from app.db.session import rw_session

CACHEABLE_STATUSES = {"completed"}


def citations(answer: str | None) -> list[str]:
    return list(dict.fromkeys(CITATION_RE.findall(normalize(answer or ""))))


def action_out(a: PendingAction) -> PendingActionOut:
    return PendingActionOut(id=a.id, run_id=a.run_id, action_type=a.action_type, payload=a.payload_json,
                            status=a.status, created_at=a.created_at, approved_at=a.approved_at,
                            approved_by=a.approved_by, decision_note=a.decision_note)


def load_run(run_id: str, include_trace: bool = True) -> RunOut | None:
    with rw_session() as s:
        r = s.get(Run, run_id)
        if r is None:
            return None
        actions = s.execute(select(PendingAction).where(PendingAction.run_id == run_id)).scalars().all()
        return RunOut(
            run_id=r.run_id, question=r.question, answer=r.answer, status=r.status,
            citations=citations(r.answer), trace=r.trace_json if include_trace else None,
            tool_calls=r.tool_calls or 0, tokens_in=r.tokens_in or 0, tokens_out=r.tokens_out or 0,
            cost_gbp=r.cost_gbp or 0.0, latency_ms=r.latency_ms or 0, model=r.model, created_at=r.created_at,
            pending_actions=[action_out(a) for a in actions],
        )


def _cache_key(question: str, model: str) -> str:
    norm = " ".join(question.lower().split())
    return hashlib.sha256(f"{model}|{norm}".encode()).hexdigest()


def cache_get(question: str, model: str) -> RunOut | None:
    ttl = get_settings().response_cache_ttl_s
    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=ttl)
    key = _cache_key(question, model)
    with rw_session() as s:
        row = s.get(ResponseCache, key)
        if row is None:
            return None
        if row.created_at < cutoff:
            s.execute(delete(ResponseCache).where(ResponseCache.question_hash == key))
            return None
        out = RunOut.model_validate(row.response_json)
    out.cached = True
    return out


def cache_put(question: str, model: str, run: RunOut) -> None:
    if run.status not in CACHEABLE_STATUSES or run.pending_actions:
        return
    key = _cache_key(question, model)
    with rw_session() as s:
        s.merge(ResponseCache(question_hash=key, model=model, response_json=run.model_dump(mode="json"),
                              created_at=dt.datetime.now(dt.UTC)))
