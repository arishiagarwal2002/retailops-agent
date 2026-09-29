"""Non-agentic baseline: single-shot text-to-SQL.

One LLM call writes one SQL query from the schema; it runs once through the
same guardrail; a second call phrases the answer from the rows. No tool
choice, no retries, no forecast, no policy search. This is what the agent
has to beat to justify its extra cost.
"""
from __future__ import annotations

import json
import re
import time
import uuid

from agent.llm import LLM, Usage, make_llm
from agent.runner import RunResult, create_run_row, finish_run_row
from app.core.config import get_settings
from tools.schema import get_schema
from tools.sql import GuardrailError, run_sql

SQL_PROMPT = """You translate questions about a UK online gift retailer into ONE PostgreSQL SELECT query.
Today is {as_of}. Revenue = quantity * unit_price; exclude cancellations (NOT is_cancellation) unless asked.
Schema:
{schema}
Reply with only the SQL inside a ```sql code block."""

ANSWER_PROMPT = """Answer the user's question in 1-3 sentences using only these query results.
Question: {question}
SQL: {sql}
Columns: {columns}
Rows (first 50): {rows}"""


def run_baseline(question: str, llm: LLM | None = None, run_id: str | None = None) -> RunResult:
    llm = llm or make_llm()
    run_id = run_id or str(uuid.uuid4())
    create_run_row(run_id, question, llm.model)
    started = time.perf_counter()
    usage = Usage()
    trace: list[dict] = []
    s = get_settings()

    resp = llm.chat(SQL_PROMPT.format(as_of=s.as_of_date, schema=json.dumps(get_schema())),
                    [{"role": "user", "content": question}])
    usage.add(resp.usage)
    trace.append({"type": "llm_call", "step": 1, "text": resp.text[:2000], "tool_uses": [],
                  "tokens_in": resp.usage.input_tokens, "tokens_out": resp.usage.output_tokens,
                  "latency_ms": 0, "stop_reason": resp.stop_reason})
    m = re.search(r"```(?:sql)?\s*(.+?)```", resp.text, flags=re.S)
    sql = (m.group(1) if m else resp.text).strip()

    status = "completed"
    try:
        out = run_sql(sql)
        trace.append({"type": "tool_call", "name": "run_sql", "input": {"query": sql}, "output": out,
                      "is_error": False, "error_kind": None, "latency_ms": out["elapsed_ms"]})
        resp2 = llm.chat("You are a concise retail analyst.", [{"role": "user", "content": ANSWER_PROMPT.format(
            question=question, sql=out["sql"], columns=out["columns"], rows=out["rows"][:50])}])
        usage.add(resp2.usage)
        trace.append({"type": "llm_call", "step": 2, "text": resp2.text[:2000], "tool_uses": [],
                      "tokens_in": resp2.usage.input_tokens, "tokens_out": resp2.usage.output_tokens,
                      "latency_ms": 0, "stop_reason": resp2.stop_reason})
        answer = resp2.text
    except (GuardrailError, Exception) as e:  # the baseline has no recovery path by design
        trace.append({"type": "tool_call", "name": "run_sql", "input": {"query": sql}, "output": str(e)[:500],
                      "is_error": True, "error_kind": "guardrail" if isinstance(e, GuardrailError) else "db",
                      "latency_ms": 0})
        answer = f"The query failed: {str(e).splitlines()[0][:200]}"
        status = "error"

    result = RunResult(
        run_id=run_id, question=question, answer=answer, status=status, trace=trace,
        tool_calls=[e["name"] for e in trace if e["type"] == "tool_call"],
        tokens_in=usage.input_tokens, tokens_out=usage.output_tokens,
        cost_gbp=round(usage.cost_gbp(llm.model), 6),
        latency_ms=int((time.perf_counter() - started) * 1000),
        steps=len([e for e in trace if e["type"] == "llm_call"]), model=llm.model, config_name="baseline",
    )
    finish_run_row(result)
    return result
