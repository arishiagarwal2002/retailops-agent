"""Tool schemas exposed to the LLM, and the single dispatch point that executes them.

Pydantic models are the one source of truth for tool arguments: they generate
the JSON schema the model sees, and validate what the model sends back.
A validation failure is returned to the model as a tool result, not raised.
"""
from __future__ import annotations

import concurrent.futures as cf
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.exc import DBAPIError, OperationalError

from tools.forecast import ForecastError, forecast_demand
from tools.policy_search import search_policy
from tools.reorder import ReorderError, draft_reorder
from tools.schema import get_schema, sample_rows
from tools.sql import GuardrailError, run_sql

TOOL_TIMEOUT_S = 30


# ------------------------------------------------------------------ arg models
class NoArgs(BaseModel):
    pass


class SampleRowsArgs(BaseModel):
    table: str = Field(description="One of: transactions, products, customers, stock_levels")
    n: int = Field(3, ge=1, le=5, description="Rows to return (max 5)")


class RunSqlArgs(BaseModel):
    query: str = Field(description="A single PostgreSQL SELECT statement (CTEs allowed). Read-only.")


class ForecastArgs(BaseModel):
    stock_code: str = Field(description="Product stock code, e.g. '85123A'")
    weeks: int = Field(4, ge=1, le=12, description="Forecast horizon in weeks (1-12)")


class PolicyArgs(BaseModel):
    query: str = Field(description="What to look up in the returns policy, supplier terms and UK consumer rights")
    k: int = Field(5, ge=1, le=8, description="Number of passages to return")


class ReorderArgs(BaseModel):
    stock_code: str = Field(description="Product stock code to reorder")
    qty: int = Field(ge=1, le=100_000, description="Units to order")
    reason: str = Field(description="Why this reorder is proposed, citing the numbers used")


class ClarifyArgs(BaseModel):
    question: str = Field(description="The clarifying question to ask the user")
    options: list[str] = Field(default_factory=list, description="Optional candidate interpretations")


class RefuseArgs(BaseModel):
    reason: str = Field(description="Short explanation of why the request cannot be done")


@dataclass
class ToolSpec:
    name: str
    description: str
    args: type[BaseModel]
    fn: Callable[..., Any]
    terminal: bool = False
    needs_run_id: bool = False


TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in [
        ToolSpec("get_schema", "Return the tables, columns, types and column descriptions of the retail "
                 "database. Call this before writing SQL if you are unsure of a column.", NoArgs, get_schema),
        ToolSpec("sample_rows", "Return up to 5 example rows from a table so you can see real value formats.",
                 SampleRowsArgs, sample_rows),
        ToolSpec("run_sql", "Execute ONE read-only PostgreSQL SELECT over transactions, products, customers "
                 "and stock_levels. Returns rows and the executed SQL. A LIMIT of 1000 is enforced. Writes, "
                 "multiple statements and other tables are rejected.", RunSqlArgs, run_sql),
        ToolSpec("forecast_demand", "Forecast weekly unit demand for one stock code for 1-12 weeks after the "
                 "last complete week of data (w/c 2011-11-28). Returns p50 point forecast and an 80% "
                 "interval (p10-p90) per week. Fails for SKUs with under 20 weeks of history.",
                 ForecastArgs, forecast_demand),
        ToolSpec("search_policy", "Search the store Returns Policy, Supplier Terms and UK Consumer Rights "
                 "Summary. Returns passages with citation labels you MUST cite.", PolicyArgs, search_policy),
        ToolSpec("draft_reorder", "Draft a purchase order for human approval. This does NOT place an order; "
                 "it records a pending action that a person must approve. Only use it when the user asks "
                 "you to reorder or asks which items to reorder AND wants it actioned.",
                 ReorderArgs, draft_reorder, needs_run_id=True),
        ToolSpec("ask_clarification", "Ask the user a clarifying question instead of guessing, when the request "
                 "is ambiguous in a way that changes the answer (e.g. 'best' by revenue or by units?). Ends "
                 "the turn.", ClarifyArgs, lambda question, options: {"question": question, "options": options},
                 terminal=True),
        ToolSpec("refuse", "Decline a request that is out of scope (not about this retailer's sales, stock, "
                 "returns or suppliers), asks to modify or delete data, or asks for secrets. Ends the turn.",
                 RefuseArgs, lambda reason: {"reason": reason}, terminal=True),
    ]
}


def anthropic_tools(names: list[str] | None = None) -> list[dict]:
    specs = [TOOLS[n] for n in names] if names else list(TOOLS.values())
    out = []
    for spec in specs:
        schema = spec.args.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        schema.setdefault("properties", {})
        out.append({"name": spec.name, "description": spec.description, "input_schema": schema})
    return out


@dataclass
class ToolOutcome:
    name: str
    input: dict
    content: Any
    is_error: bool = False
    error_kind: str | None = None  # validation | guardrail | db | timeout | tool | unknown_tool
    terminal: bool = False
    latency_ms: int = 0
    extra: dict = field(default_factory=dict)


_executor = cf.ThreadPoolExecutor(max_workers=8)


def execute(name: str, raw_input: dict, run_id: str) -> ToolOutcome:
    started = time.perf_counter()

    def done(**kw) -> ToolOutcome:
        return ToolOutcome(name=name, input=raw_input, latency_ms=int((time.perf_counter() - started) * 1000), **kw)

    spec = TOOLS.get(name)
    if spec is None:
        return done(content=f"unknown tool '{name}'", is_error=True, error_kind="unknown_tool")
    try:
        args = spec.args.model_validate(raw_input or {})
    except ValidationError as e:
        return done(content=f"invalid arguments: {e.errors(include_url=False)}", is_error=True,
                    error_kind="validation")
    kwargs = args.model_dump()
    if spec.needs_run_id:
        kwargs["run_id"] = run_id
    future = _executor.submit(spec.fn, **kwargs)
    try:
        result = future.result(timeout=TOOL_TIMEOUT_S)
    except cf.TimeoutError:
        return done(content=f"tool '{name}' timed out after {TOOL_TIMEOUT_S}s", is_error=True, error_kind="timeout")
    except GuardrailError as e:
        return done(content=f"query rejected by guardrail: {e}", is_error=True, error_kind="guardrail")
    except (ForecastError, ReorderError, ValueError) as e:
        return done(content=str(e), is_error=True, error_kind="tool")
    except OperationalError as e:
        msg = str(e.orig).splitlines()[0] if e.orig else str(e)
        kind = "timeout" if "statement timeout" in msg else "db"
        return done(content=f"database error: {msg}", is_error=True, error_kind=kind)
    except DBAPIError as e:
        msg = str(e.orig).splitlines()[0] if e.orig else str(e)
        return done(content=f"database error: {msg}", is_error=True, error_kind="db")
    return done(content=result, terminal=spec.terminal)
