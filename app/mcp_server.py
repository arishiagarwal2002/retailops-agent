"""The agent's read-only tools exposed as an MCP server (stdio).

Lets any MCP client (Claude Desktop, Claude Code, ...) query the retail data
with the same guardrails. draft_reorder is deliberately NOT exposed: the
approval workflow belongs to the API, not to arbitrary clients.

    python -m app.mcp_server
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from tools.forecast import ForecastError
from tools.forecast import forecast_demand as _forecast
from tools.policy_search import search_policy as _search
from tools.schema import get_schema as _schema
from tools.schema import sample_rows as _sample
from tools.sql import GuardrailError
from tools.sql import run_sql as _run_sql

mcp = MCPServer("retailops", instructions="Read-only access to a UK gift retailer's sales, stock, forecasts and policies.")


@mcp.tool()
def get_schema() -> dict:
    """Tables, columns and column descriptions of the retail database."""
    return _schema()


@mcp.tool()
def sample_rows(table: str, n: int = 3) -> dict:
    """Up to 5 example rows from transactions, products, customers or stock_levels."""
    return _sample(table, n)


@mcp.tool()
def run_sql(query: str) -> dict:
    """One read-only PostgreSQL SELECT (AST-validated, read-only role, 5 s timeout, LIMIT 1000)."""
    try:
        return _run_sql(query)
    except GuardrailError as e:
        raise ToolError(f"query rejected by guardrail: {e}") from e


@mcp.tool()
def forecast_demand(stock_code: str, weeks: int = 4) -> dict:
    """Weekly demand forecast (p50 and 80% interval) for 1-12 weeks."""
    try:
        return _forecast(stock_code, weeks)
    except ForecastError as e:
        raise ToolError(str(e)) from e


@mcp.tool()
def search_policy(query: str, k: int = 5) -> dict:
    """Search returns policy, supplier terms and UK consumer rights; cite the returned labels."""
    return _search(query, k)


if __name__ == "__main__":
    mcp.run()
