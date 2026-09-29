"""Graph nodes: route/plan, execute tools, self-correct, verify, await approval.

Nodes are built by a factory so that the LLM and the ablation switches
(self-correction, verifier) are injected rather than global.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

from langgraph.types import interrupt

from agent import prompts
from agent.llm import LLM
from agent.state import AgentConfig, AgentState
from agent.verifier import has_citation, tool_output_texts, ungrounded_numbers
from app.core.config import get_settings
from app.core.logging import log_event
from tools.registry import anthropic_tools, execute

MAX_TOOL_OUTPUT_CHARS = 12_000


def _truncate(content) -> str:
    s = content if isinstance(content, str) else json.dumps(content, default=str)
    if len(s) > MAX_TOOL_OUTPUT_CHARS:
        s = s[:MAX_TOOL_OUTPUT_CHARS] + f'... [truncated {len(s) - MAX_TOOL_OUTPUT_CHARS} chars; use a narrower query]'
    return s


def make_nodes(llm: LLM, cfg: AgentConfig):
    settings = get_settings()
    tools = anthropic_tools()
    system = prompts.system_prompt()

    # ------------------------------------------------------------ [1] route / plan
    def route(state: AgentState) -> dict:
        if state.get("steps", 0) >= settings.step_limit:
            return {
                "status": "step_limit",
                "answer": "I could not complete this within the step budget "
                          f"({settings.step_limit} steps). The tool trace shows how far I got.",
            }
        started = time.perf_counter()
        resp = llm.chat(system, state["messages"], tools)
        latency = int((time.perf_counter() - started) * 1000)
        cost = resp.usage.cost_usd(llm.model)
        event = {
            "type": "llm_call", "step": state.get("steps", 0) + 1, "latency_ms": latency,
            "tokens_in": resp.usage.input_tokens + resp.usage.cache_read_tokens + resp.usage.cache_write_tokens,
            "tokens_out": resp.usage.output_tokens, "stop_reason": resp.stop_reason,
            "text": resp.text[:2000], "tool_uses": [{"name": t["name"], "input": t["input"]} for t in resp.tool_uses],
        }
        log_event("llm_call", run_id=state["run_id"], step=event["step"], latency_ms=latency,
                  tokens_in=event["tokens_in"], tokens_out=event["tokens_out"])
        update = {
            "messages": state["messages"] + [{"role": "assistant", "content": resp.content}],
            "trace": state.get("trace", []) + [event],
            "steps": state.get("steps", 0) + 1,
            "tokens_in": state.get("tokens_in", 0) + event["tokens_in"],
            "tokens_out": state.get("tokens_out", 0) + event["tokens_out"],
            "cost_usd": state.get("cost_usd", 0.0) + cost,
        }
        if not resp.tool_uses:
            update["answer"] = resp.text
        return update

    def after_route(state: AgentState) -> str:
        if state.get("status") == "step_limit":
            return "end"
        last = state["messages"][-1]
        if any(b.get("type") == "tool_use" for b in last["content"]):
            return "execute"
        return "verify"

    # ------------------------------------------------------------ [2] tool execution
    def execute_tools(state: AgentState) -> dict:
        uses = [b for b in state["messages"][-1]["content"] if b.get("type") == "tool_use"]
        with ThreadPoolExecutor(max_workers=max(1, len(uses))) as pool:
            outcomes = list(pool.map(lambda b: execute(b["name"], b.get("input") or {}, state["run_id"]), uses))

        results, events = [], []
        terminal = None
        pending = list(state.get("pending_action_ids", []))
        for block, o in zip(uses, outcomes):
            content = o.content
            if o.is_error and not cfg.self_correct:
                # Ablation: without self-correction the model only learns that the call failed.
                content = "error: tool call failed"
            results.append({"type": "tool_result", "tool_use_id": block["id"],
                            "content": _truncate(content), "is_error": o.is_error})
            events.append({"type": "tool_call", "name": o.name, "input": o.input, "is_error": o.is_error,
                           "error_kind": o.error_kind, "latency_ms": o.latency_ms,
                           "output": o.content if not isinstance(o.content, str) else o.content[:4000]})
            log_event("tool_call", run_id=state["run_id"], tool=o.name, is_error=o.is_error,
                      error_kind=o.error_kind, latency_ms=o.latency_ms)
            if o.terminal and not o.is_error:
                terminal = o
            if o.name == "draft_reorder" and not o.is_error:
                pending.append(o.content["pending_action_id"])

        update: dict = {
            "messages": state["messages"] + [{"role": "user", "content": results}],
            "trace": state.get("trace", []) + events,
            "pending_action_ids": pending,
        }
        if terminal is not None:
            if terminal.name == "ask_clarification":
                q = terminal.content["question"]
                opts = terminal.content.get("options") or []
                update["answer"] = q + ("\n\nOptions: " + "; ".join(opts) if opts else "")
                update["status"] = "needs_clarification"
            else:
                update["answer"] = "I can't help with that: " + terminal.content["reason"]
                update["status"] = "refused"
        return update

    def after_execute(state: AgentState) -> str:
        if state.get("status") in ("needs_clarification", "refused"):
            return "end"
        return "self_correct" if cfg.self_correct else "route"

    # ------------------------------------------------------------ [3] self-correction
    def self_correct(state: AgentState) -> dict:
        """Append corrective guidance to the tool-result message after failures or empty results."""
        new_events = [e for e in state["trace"] if e["type"] == "tool_call"]
        n_new = sum(1 for b in state["messages"][-1]["content"] if b.get("type") == "tool_result")
        recent = new_events[-n_new:] if n_new else []
        hints: list[str] = []
        sql_failures = state.get("sql_failures", 0)
        empty_used = state.get("empty_retry_used", False)
        for e in recent:
            if e["name"] == "run_sql" and e["is_error"] and e["error_kind"] in ("guardrail", "db", "validation"):
                sql_failures += 1
                if sql_failures <= settings.sql_max_retries:
                    hints.append(prompts.SQL_RETRY_HINT.format(attempt=sql_failures,
                                                               max_retries=settings.sql_max_retries))
                else:
                    hints.append(prompts.SQL_GIVE_UP_HINT.format(attempts=sql_failures))
            elif e["is_error"] and e["error_kind"] == "timeout":
                hints.append(prompts.TIMEOUT_HINT.format(tool=e["name"]))
            elif e["name"] == "run_sql" and not e["is_error"] and e["output"].get("row_count") == 0 and not empty_used:
                empty_used = True
                hints.append(prompts.EMPTY_RESULT_HINT)
        if not hints:
            return {"sql_failures": sql_failures, "empty_retry_used": empty_used}
        last = dict(state["messages"][-1])
        last["content"] = list(last["content"]) + [{"type": "text", "text": "\n".join(hints)}]
        return {
            "messages": state["messages"][:-1] + [last],
            "trace": state["trace"] + [{"type": "self_correction", "hints": hints}],
            "sql_failures": sql_failures,
            "empty_retry_used": empty_used,
        }

    # ------------------------------------------------------------ [4] verifier
    def verify(state: AgentState) -> dict:
        answer = state.get("answer", "")
        if not cfg.verifier:
            return {"status": "completed"}
        outputs = tool_output_texts(state.get("trace", []))
        bad, checked = ungrounded_numbers(answer, outputs, state["question"])
        used_policy = any(e["type"] == "tool_call" and e["name"] == "search_policy" and not e["is_error"]
                          for e in state.get("trace", []))
        missing_citation = used_policy and not has_citation(answer)
        problems = []
        if bad:
            problems.append(f"- These numbers do not appear in any tool result: {', '.join(bad)}")
        if missing_citation:
            problems.append("- You used search_policy but the answer has no [Document §x.y] citation.")
        event = {"type": "verification", "checked_numbers": checked, "ungrounded": bad,
                 "missing_citation": missing_citation, "attempt": state.get("verify_attempts", 0) + 1}
        trace = state.get("trace", []) + [event]
        if not problems:
            return {"trace": trace, "status": "completed"}
        if state.get("verify_attempts", 0) == 0:
            feedback = {"role": "user", "content": prompts.VERIFIER_FEEDBACK.format(problems="\n".join(problems))}
            return {"trace": trace, "verify_attempts": 1, "messages": state["messages"] + [feedback],
                    "status": "running"}
        warning = "\n\n> **Warning:** "
        warning += ("unverified values: " + ", ".join(bad) + ". " if bad else "")
        warning += ("policy claims lack a citation." if missing_citation else "")
        return {"trace": trace, "answer": answer + warning.rstrip(), "status": "completed_with_warnings"}

    def after_verify(state: AgentState) -> str:
        if state.get("status") == "running":
            return "route"
        return "await_approval" if state.get("pending_action_ids") else "end"

    # ------------------------------------------------------------ human approval
    def await_approval(state: AgentState) -> dict:
        """Pause the graph until a human decides. Resumed by POST /v1/approvals/{id}."""
        decision = interrupt({"pending_action_ids": state["pending_action_ids"], "answer": state.get("answer")})
        note = (f"\n\n**Update:** pending action #{decision.get('action_id')} was "
                f"{decision.get('decision')} by {decision.get('by')}.")
        return {
            "decision": decision,
            "answer": state.get("answer", "") + note,
            "trace": state.get("trace", []) + [{"type": "human_decision", **decision}],
        }

    return {
        "route": route, "after_route": after_route,
        "execute": execute_tools, "after_execute": after_execute,
        "self_correct": self_correct,
        "verify": verify, "after_verify": after_verify,
        "await_approval": await_approval,
    }
