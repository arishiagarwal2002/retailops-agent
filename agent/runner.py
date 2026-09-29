"""Run the agent end to end: persist a runs row, invoke the graph, record the outcome.

Used by the API (sync and async paths) and by the evaluation harness, so
evaluation measures exactly what is served.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field

from langgraph.types import Command

from agent.graph import build_graph
from agent.llm import LLM, make_llm
from agent.observability import export_run
from agent.state import AgentConfig
from app.core.config import get_settings
from app.core.logging import log_event, trace_id_var
from app.db.models import Run
from app.db.session import rw_session


@dataclass
class RunResult:
    run_id: str
    question: str
    answer: str
    status: str
    trace: list[dict]
    tool_calls: list[str]
    tokens_in: int
    tokens_out: int
    cost_gbp: float
    latency_ms: int
    steps: int
    pending_action_ids: list[int] = field(default_factory=list)
    model: str = ""
    config_name: str = "agent"

    def to_dict(self) -> dict:
        return asdict(self)


# run_id -> (graph, config) for runs paused at the approval interrupt.
_PAUSED: dict[str, tuple[object, dict]] = {}


def create_run_row(run_id: str, question: str, model: str, api_key_id: int | None = None) -> None:
    with rw_session() as s:
        s.add(Run(run_id=run_id, question=question, status="running", model=model, api_key_id=api_key_id))


def finish_run_row(r: RunResult) -> None:
    with rw_session() as s:
        row = s.get(Run, r.run_id)
        if row is None:
            return
        row.answer = r.answer
        row.status = r.status
        row.trace_json = r.trace
        row.tool_calls = len(r.tool_calls)
        row.tokens_in = r.tokens_in
        row.tokens_out = r.tokens_out
        row.cost_gbp = r.cost_gbp
        row.latency_ms = r.latency_ms


def _result(run_id: str, question: str, state: dict, started: float, model: str, cfg: AgentConfig,
            interrupted: bool) -> RunResult:
    trace = state.get("trace", [])
    status = "awaiting_approval" if interrupted else state.get("status", "error")
    return RunResult(
        run_id=run_id,
        question=question,
        answer=state.get("answer", ""),
        status=status,
        trace=trace,
        tool_calls=[e["name"] for e in trace if e["type"] == "tool_call"],
        tokens_in=state.get("tokens_in", 0),
        tokens_out=state.get("tokens_out", 0),
        cost_gbp=round(state.get("cost_usd", 0.0) * get_settings().usd_to_gbp, 6),
        latency_ms=int((time.perf_counter() - started) * 1000),
        steps=state.get("steps", 0),
        pending_action_ids=state.get("pending_action_ids", []),
        model=model,
        config_name=cfg.name,
    )


def run_agent(question: str, llm: LLM | None = None, cfg: AgentConfig = AgentConfig(),
              run_id: str | None = None, api_key_id: int | None = None, create_row: bool = True) -> RunResult:
    run_id = run_id or str(uuid.uuid4())
    trace_id_var.set(run_id)
    llm = llm or make_llm(cfg.model)
    if create_row:
        create_run_row(run_id, question, llm.model, api_key_id)
    started = time.perf_counter()
    graph = build_graph(llm, cfg)
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": 60}
    initial = {
        "run_id": run_id, "question": question,
        "messages": [{"role": "user", "content": question}],
        "trace": [], "steps": 0, "status": "running", "pending_action_ids": [],
        "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
    }
    try:
        state = graph.invoke(initial, config)
        interrupted = bool(state.get("__interrupt__"))
        result = _result(run_id, question, state, started, llm.model, cfg, interrupted)
        if interrupted:
            _PAUSED[run_id] = (graph, config)
    except Exception as e:
        log_event("run_failed", run_id=run_id, error=repr(e))
        reason = str(e).splitlines()[0][:200] if str(e) else ""
        if "authentication" in reason.lower() or "api_key" in reason.lower():
            reason = "the LLM is not configured (set the provider's API key in .env)"
        result = RunResult(run_id, question, f"The run failed: {type(e).__name__}: {reason}", "error", [], [], 0, 0, 0.0,
                           int((time.perf_counter() - started) * 1000), 0, [], llm.model, cfg.name)
    finish_run_row(result)
    log_event("run_finished", run_id=run_id, status=result.status, steps=result.steps,
              latency_ms=result.latency_ms, cost_gbp=result.cost_gbp, tools=result.tool_calls)
    export_run(result)
    return result


def resume_run(run_id: str, decision: dict) -> RunResult | None:
    """Resume a run paused at await_approval. Returns None if the process no longer holds it."""
    paused = _PAUSED.pop(run_id, None)
    if paused is None:
        return None
    graph, config = paused
    started = time.perf_counter()
    state = graph.invoke(Command(resume=decision), config)
    snapshot = graph.get_state(config).values
    prev_latency = 0
    with rw_session() as s:
        row = s.get(Run, run_id)
        prev_latency = row.latency_ms if row else 0
    result = _result(run_id, snapshot["question"], state, started, "", AgentConfig(), False)
    result.status = "completed"
    result.latency_ms = prev_latency  # keep the agent's own latency; approval wait is not agent time
    with rw_session() as s:
        row = s.get(Run, run_id)
        if row:
            row.answer = result.answer
            row.status = result.status
            row.trace_json = result.trace
    return result
