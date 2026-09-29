"""run_sql and its guardrail.

Defence in depth - any one layer alone is insufficient:
  1. AST validation (this file): one statement, read-only query, allow-listed
     tables, no side-effecting functions, LIMIT enforced.
  2. A dedicated read-only Postgres role with SELECT on four tables only.
  3. Session-level statement_timeout and default_transaction_read_only.

Layer 1 gives the model a precise, fixable error message. Layers 2 and 3 hold
even if layer 1 has a parser bug.
"""
from __future__ import annotations

import datetime as dt
import decimal
import time

import sqlglot
from sqlalchemy import text
from sqlglot import exp
from sqlglot.errors import ParseError

from app.core.config import get_settings
from app.db.session import ro_engine

ALLOWED_TABLES = {"transactions", "products", "customers", "stock_levels"}

# Statement types that must never appear anywhere in the tree, including inside CTEs.
FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
    exp.Command, exp.Into, exp.Lock, exp.Copy, exp.Grant, exp.Set, exp.TruncateTable,
)
# Functions with side effects or that reach outside the allowed data.
FORBIDDEN_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink", "set_config", "current_setting", "query_to_xml",
                               "txid_", "version")


class GuardrailError(ValueError):
    """Raised when a query is rejected. The message is returned to the model verbatim."""


def validate_sql(query: str, row_limit: int | None = None) -> str:
    row_limit = row_limit or get_settings().sql_row_limit
    query = query.strip().rstrip(";").strip()
    if not query:
        raise GuardrailError("empty query")
    try:
        parsed = [p for p in sqlglot.parse(query, dialect="postgres") if p is not None]
    except ParseError as e:
        raise GuardrailError(f"could not parse SQL: {str(e).splitlines()[0]}") from e
    if len(parsed) != 1:
        raise GuardrailError("exactly one statement allowed")
    stmt = parsed[0]
    if not isinstance(stmt, exp.Query):
        raise GuardrailError(f"only SELECT permitted, got {type(stmt).__name__.upper()}")

    for node in stmt.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise GuardrailError(f"{type(node).__name__.upper()} is not permitted; the SQL tool is read-only")
        if isinstance(node, exp.Select) and node.args.get("locks"):
            raise GuardrailError("row locking (FOR UPDATE/SHARE) is not permitted")
        if isinstance(node, exp.Func):
            name = (node.sql_name() if not isinstance(node, exp.Anonymous) else node.name).lower()
            if name.startswith(FORBIDDEN_FUNCTION_PREFIXES):
                raise GuardrailError(f"function not permitted: {name}")

    cte_names = {cte.alias_or_name.lower() for cte in stmt.find_all(exp.CTE)}
    for table in stmt.find_all(exp.Table):
        name = table.name.lower()
        if table.args.get("db") or table.args.get("catalog"):
            raise GuardrailError(f"schema-qualified tables are not permitted: {table.sql()}")
        if name not in ALLOWED_TABLES and name not in cte_names:
            raise GuardrailError(
                f"table not allowed: {table.name}. Allowed tables: {', '.join(sorted(ALLOWED_TABLES))}")

    limit = stmt.args.get("limit")
    if limit is None:
        stmt = stmt.limit(row_limit)
    else:
        try:
            value = int(limit.expression.name)
            if value > row_limit:
                stmt = stmt.limit(row_limit)
        except (AttributeError, ValueError):
            stmt = stmt.limit(row_limit)
    return stmt.sql(dialect="postgres")


def _jsonable(v):
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    return v


def run_sql(query: str) -> dict:
    """Validate, execute on the read-only role, return rows. Raises GuardrailError or DB errors."""
    safe = validate_sql(query)
    started = time.perf_counter()
    with ro_engine().connect() as conn:
        result = conn.execute(text(safe.replace(":", r"\:")))
        columns = list(result.keys())
        rows = [[_jsonable(v) for v in r] for r in result.fetchall()]
    return {
        "sql": safe,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": len(rows) >= get_settings().sql_row_limit,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
