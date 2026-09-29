"""Streamlit demo: the answer beside the live tool trace, plus the pending-approval panel.

    streamlit run ui/streamlit_app.py
Talks to the API over HTTP (API_URL, default http://localhost:8000).
"""
from __future__ import annotations

import json
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
EXAMPLES = [
    "Which 5 products sold the most units in November 2011?",
    "Using a 4-week demand forecast, will current stock of 85123A cover demand?",
    "The customer on invoice 578680 wants to return their order. Can they?",
    "Is 23084 at or below its reorder point? If so, draft a reorder sized from a 4-week forecast.",
    "What were my best products last year?",
    "Delete all the cancelled orders.",
]
STATUS_STYLE = {
    "completed": ("✅", "green"), "completed_with_warnings": ("⚠️", "orange"),
    "needs_clarification": ("❓", "blue"), "refused": ("⛔", "red"), "awaiting_approval": ("⏸️", "violet"),
    "step_limit": ("⏱️", "orange"), "error": ("❌", "red"),
}

st.set_page_config(page_title="RetailOps Agent", page_icon="🛒", layout="wide")

with st.sidebar:
    st.header("RetailOps Agent")
    st.caption("Tool-using LLM agent over ~1M UK retail transactions (Dec 2009 – Dec 2011). "
               "Stock levels and policies are synthetic.")
    api_key = st.text_input("API key", type="password", value=os.environ.get("DEMO_API_KEY", ""))
    model = st.selectbox("Model", ["server default", "claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"], index=0,
                         help="Model overrides only apply when the server uses the Anthropic provider.")
    st.caption("A daily spend cap and per-key rate limit protect the demo.")

headers = {"X-API-Key": api_key} if api_key else {}


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API_URL}{path}", headers=headers, timeout=180, **kw)
    except requests.RequestException as e:
        st.error(f"API unreachable at {API_URL}: {e}")
        return None
    if r.status_code >= 400:
        st.error(f"{r.status_code}: {r.json().get('detail') or r.json().get('error') if r.headers.get('content-type','').startswith('application/json') else r.text}")
        return None
    return r.json()


def render_trace(trace: list[dict]) -> None:
    for e in trace or []:
        t = e["type"]
        if t == "llm_call":
            calls = ", ".join(u["name"] for u in e.get("tool_uses", [])) or "final answer"
            with st.expander(f"🧠 step {e['step']}: model → {calls}  ·  {e['latency_ms']} ms  ·  "
                             f"{e['tokens_in']}/{e['tokens_out']} tok"):
                if e.get("text"):
                    st.markdown(e["text"])
                for u in e.get("tool_uses", []):
                    st.code(json.dumps(u["input"], indent=2), language="json")
        elif t == "tool_call":
            icon = "🔴" if e["is_error"] else "🔧"
            with st.expander(f"{icon} {e['name']}  ·  {e['latency_ms']} ms" + (f"  ·  {e['error_kind']}" if e["is_error"] else "")):
                if e["name"] == "run_sql" and isinstance(e.get("output"), dict):
                    st.code(e["output"].get("sql", e["input"].get("query")), language="sql")
                    out = e["output"]
                    st.dataframe([dict(zip(out["columns"], r)) for r in out["rows"][:50]], use_container_width=True)
                else:
                    st.json(e.get("output") if not isinstance(e.get("output"), str) else {"message": e["output"]})
        elif t == "self_correction":
            st.info("🔁 self-correction: " + " ".join(e["hints"]))
        elif t == "verification":
            if e["ungrounded"] or e["missing_citation"]:
                st.warning(f"🧪 verifier (attempt {e['attempt']}): ungrounded {e['ungrounded']}"
                           + (" · missing citation" if e["missing_citation"] else ""))
            else:
                st.success(f"🧪 verifier: {e['checked_numbers']} numbers checked, all traceable to tool output")
        elif t == "human_decision":
            st.info(f"👤 human decision: {e.get('decision')} by {e.get('by')}")


tab_ask, tab_approvals, tab_runs = st.tabs(["Ask", "Approvals", "Recent runs"])

with tab_ask:
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state["question"] = ex
    question = st.text_area("Question", key="question", height=80,
                            placeholder="e.g. Will I run out of 85123A before Christmas?")
    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Agent working…"):
            st.session_state["last"] = api("POST", "/v1/ask", json={"question": question, "model": None if model == "server default" else model})

    run = st.session_state.get("last")
    if run:
        left, right = st.columns([3, 2])
        with left:
            icon, colour = STATUS_STYLE.get(run["status"], ("•", "gray"))
            st.markdown(f"#### {icon} :{colour}[{run['status'].replace('_', ' ')}]" + ("  · cached" if run.get("cached") else ""))
            st.markdown(run["answer"] or "")
            m = st.columns(4)
            m[0].metric("Latency", f"{run['latency_ms'] / 1000:.1f}s")
            m[1].metric("Cost", f"£{run['cost_gbp']:.4f}")
            m[2].metric("Tokens in/out", f"{run['tokens_in']}/{run['tokens_out']}")
            m[3].metric("Tool calls", run["tool_calls"])
            for a in run.get("pending_actions", []):
                if a["status"] == "pending":
                    st.warning(f"Pending action #{a['id']}: reorder {a['payload']['qty']} × {a['payload']['stock_code']} "
                               f"(≈£{a['payload']['est_value_gbp']:,.2f}). Approve it in the Approvals tab.")
        with right:
            st.markdown("#### Tool trace")
            render_trace(run.get("trace"))

with tab_approvals:
    st.caption("The agent can only draft reorders. Nothing changes until a person approves here.")
    status = st.radio("Show", ["pending", "approved", "rejected", "all"], horizontal=True)
    actions = api("GET", f"/v1/approvals?status={status}") if api_key else None
    for a in actions or []:
        p = a["payload"]
        with st.container(border=True):
            st.markdown(f"**#{a['id']} · reorder {p['qty']} × {p['stock_code']}** · supplier {p.get('supplier_id')} · "
                        f"≈£{p.get('est_value_gbp', 0):,.2f} · on hand {p.get('on_hand')} / reorder point {p.get('reorder_point')}")
            st.caption(p.get("reason", ""))
            if a["status"] == "pending":
                note = st.text_input("Note", key=f"note{a['id']}")
                c1, c2 = st.columns(2)
                if c1.button("Approve", key=f"ap{a['id']}", type="primary"):
                    api("POST", f"/v1/approvals/{a['id']}", json={"decision": "approve", "note": note or None})
                    st.rerun()
                if c2.button("Reject", key=f"rj{a['id']}"):
                    api("POST", f"/v1/approvals/{a['id']}", json={"decision": "reject", "note": note or None})
                    st.rerun()
            else:
                st.markdown(f"_{a['status']} by {a['approved_by']} at {a['approved_at']}_")

with tab_runs:
    data = api("GET", "/v1/runs?limit=25") if api_key else None
    if data:
        st.dataframe([{k: r[k] for k in ("created_at", "status", "question", "tool_calls", "latency_ms", "cost_gbp", "model")}
                      for r in data["items"]], use_container_width=True)
