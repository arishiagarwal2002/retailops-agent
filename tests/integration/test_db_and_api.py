"""Read-only role enforcement, tools against real data, and the HTTP API with a mocked LLM."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from agent.llm import ScriptedLLM, text_response, tool_response
from app.core.auth import create_api_key
from app.db.session import ro_engine
from tests.conftest import requires_db

pytestmark = requires_db


@pytest.mark.parametrize("stmt", [
    "DELETE FROM transactions WHERE false",
    "UPDATE products SET category = category WHERE false",
    "SELECT count(*) FROM runs",
    "SELECT count(*) FROM api_keys",
])
def test_read_only_role_blocks_writes_and_app_tables_even_without_the_guard(stmt):
    """Layer 2: bypass the AST guard entirely and hit the database role directly."""
    with ro_engine().connect() as conn, pytest.raises(DBAPIError):
        conn.execute(text(stmt))


def test_statement_timeout_applies():
    with ro_engine().connect() as conn, pytest.raises(Exception, match="statement timeout"):
        conn.execute(text("SELECT pg_sleep(6)"))


def test_run_sql_returns_rows_and_executed_sql():
    from tools.sql import run_sql

    out = run_sql("SELECT stock_code FROM products ORDER BY stock_code")
    assert out["sql"].endswith("LIMIT 1000") and out["row_count"] <= 1000


def test_sample_rows_hides_customer_id():
    from tools.schema import sample_rows

    out = sample_rows("transactions", 3)
    assert "customer_id" not in out["columns"] and len(out["rows"]) <= 3


def test_forecast_refuses_short_history():
    from tools.forecast import ForecastError, forecast_demand

    with pytest.raises(ForecastError, match="insufficient history"):
        forecast_demand("INJ001", 4)


@pytest.fixture()
def client(monkeypatch):
    import agent.runner

    script = ScriptedLLM(lambda msgs: tool_response("run_sql", {"query": "SELECT count(*) AS n FROM products"})
                         if len(msgs) == 1 else text_response("Done."))
    monkeypatch.setattr(agent.runner, "make_llm", lambda model=None: script)
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_auth_required(client):
    assert client.post("/v1/ask", json={"question": "hello there"}).status_code == 401
    assert client.get("/v1/runs", headers={"X-API-Key": "nope"}).status_code == 401


def test_ask_and_runs_and_approvals(client):
    h = {"X-API-Key": create_api_key("pytest")}
    r = client.post("/v1/ask", json={"question": "How many products do we have?", "use_cache": False}, headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed" and body["trace"]
    got = client.get(f"/v1/runs/{body['run_id']}", headers=h).json()
    assert got["run_id"] == body["run_id"]
    accepted = client.post("/v1/runs", json={"question": "async question here"}, headers=h)
    assert accepted.status_code == 202
    assert client.get(accepted.json()["poll"], headers=h).status_code == 200
    assert client.post("/v1/approvals/999999", json={"decision": "approve"}, headers=h).status_code == 404


def test_health(client):
    assert client.get("/health").json()["database"] == "ok"
