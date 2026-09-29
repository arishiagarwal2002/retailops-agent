"""POST /v1/approvals/{id} - the ONLY path by which the agent's proposals change state."""
from __future__ import annotations

import asyncio
import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from agent.runner import resume_run
from app.api.schemas import ApprovalRequest, PendingActionOut
from app.api.service import action_out
from app.core.auth import Principal, require_api_key
from app.core.logging import log_event
from app.db.models import PendingAction, PurchaseOrder
from app.db.session import rw_session

router = APIRouter(prefix="/v1", tags=["approvals"])


def decide(action_id: int, decision: str, approver: str, note: str | None) -> PendingActionOut:
    with rw_session() as s:
        # Row lock: two approvers clicking at once cannot both place the order.
        action = s.execute(select(PendingAction).where(PendingAction.id == action_id).with_for_update()).scalar_one_or_none()
        if action is None:
            raise HTTPException(404, "pending action not found")
        if action.status != "pending":
            raise HTTPException(409, f"action already {action.status}")
        action.status = "approved" if decision == "approve" else "rejected"
        action.approved_at = dt.datetime.now(dt.UTC)
        action.approved_by = approver
        action.decision_note = note
        if action.status == "approved" and action.action_type == "reorder":
            p = action.payload_json
            s.add(PurchaseOrder(pending_action_id=action.id, stock_code=p["stock_code"], quantity=p["qty"],
                                supplier_id=p.get("supplier_id")))
        s.flush()
        out = action_out(action)
    log_event("approval_decision", action_id=action_id, decision=out.status, approver=approver)
    return out


@router.post("/approvals/{action_id}", response_model=PendingActionOut)
async def approve(action_id: int, body: ApprovalRequest, principal: Principal = Depends(require_api_key)):
    out = await asyncio.to_thread(decide, action_id, body.decision, principal.name, body.note)
    # Resume the paused graph (if this process still holds it) so the run records the outcome.
    await asyncio.to_thread(resume_run, out.run_id,
                            {"action_id": action_id, "decision": out.status, "by": principal.name})
    return out


@router.get("/approvals", response_model=list[PendingActionOut])
async def list_approvals(status: str = Query("pending", pattern="^(pending|approved|rejected|all)$"),
                         _: Principal = Depends(require_api_key)):
    def _list():
        with rw_session() as s:
            q = select(PendingAction).order_by(PendingAction.created_at.desc()).limit(100)
            if status != "all":
                q = q.where(PendingAction.status == status)
            return [action_out(a) for a in s.execute(q).scalars().all()]

    return await asyncio.to_thread(_list)
