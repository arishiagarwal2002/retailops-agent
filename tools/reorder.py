"""draft_reorder: records a PENDING purchase order. It never places anything.

The only path from pending to placed is POST /v1/approvals/{id} by a human
(see app/api/approvals.py).
"""
from __future__ import annotations

from sqlalchemy import text

from app.db.models import PendingAction
from app.db.session import ro_engine, rw_session

MAX_QTY = 100_000


class ReorderError(ValueError):
    pass


def draft_reorder(stock_code: str, qty: int, reason: str, run_id: str) -> dict:
    stock_code = stock_code.strip().upper()
    if not 1 <= qty <= MAX_QTY:
        raise ReorderError(f"qty must be between 1 and {MAX_QTY}")
    with ro_engine().connect() as conn:
        row = conn.execute(
            text("SELECT p.description, p.avg_unit_price, s.on_hand, s.reorder_point, s.supplier_id "
                 "FROM products p LEFT JOIN stock_levels s USING (stock_code) WHERE p.stock_code = :c"),
            {"c": stock_code},
        ).first()
    if row is None:
        raise ReorderError(f"unknown stock_code '{stock_code}'")
    payload = {
        "stock_code": stock_code,
        "qty": int(qty),
        "reason": reason[:500],
        "supplier_id": row.supplier_id,
        "on_hand": row.on_hand,
        "reorder_point": row.reorder_point,
        "est_value_gbp": round(float(row.avg_unit_price or 0) * qty, 2),
    }
    with rw_session() as s:
        action = PendingAction(run_id=run_id, action_type="reorder", payload_json=payload, status="pending")
        s.add(action)
        s.flush()
        action_id = action.id
    return {
        "pending_action_id": action_id,
        "status": "pending_approval",
        "payload": payload,
        "message": "Draft recorded. Nothing has been ordered. A human must approve it via "
                   f"POST /v1/approvals/{action_id}.",
    }
