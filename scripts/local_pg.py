"""Start a local Postgres 16 + pgvector without Docker (uses the `pgserver` wheel).

Handy on machines without Docker. Creates the `retailops` database and owner
role, then writes the connection strings into .env.

    python -m scripts.local_pg          # start (idempotent) and write .env
    python -m scripts.local_pg --stop
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pgserver

ROOT = Path(__file__).resolve().parents[1]
PGDATA = ROOT / ".pgdata"
ENV = ROOT / ".env"


def upsert_env(values: dict[str, str]) -> None:
    text = ENV.read_text(encoding="utf-8") if ENV.exists() else ""
    for key, value in values.items():
        line = f"{key}={value}"
        if re.search(rf"^{key}=.*$", text, flags=re.M):
            text = re.sub(rf"^{key}=.*$", line, text, flags=re.M)
        else:
            text += ("" if text.endswith("\n") or not text else "\n") + line + "\n"
    ENV.write_text(text, encoding="utf-8")


def clear_stale_pid() -> None:
    """After a reboot or crash, postmaster.pid can outlive the server and block a restart."""
    pid_file = PGDATA / "postmaster.pid"
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().splitlines()[0])
    except (ValueError, IndexError):
        pid_file.unlink()
        return
    try:
        import psutil

        alive = psutil.pid_exists(pid) and "postgres" in psutil.Process(pid).name().lower()
    except ImportError:  # psutil ships with pgserver, but don't depend on it
        alive = True
    except Exception:
        alive = False
    if not alive:
        pid_file.unlink()
        print(f"removed stale postmaster.pid (pid {pid} not running)")


def main() -> None:
    if "--stop" not in sys.argv:
        clear_stale_pid()
    srv = pgserver.get_server(PGDATA, cleanup_mode=None)
    if "--stop" in sys.argv:
        srv.cleanup()
        print("stopped")
        return
    uri = srv.get_uri()  # postgresql://postgres:@127.0.0.1:<port>/postgres
    port = re.search(r":(\d+)/", uri).group(1)
    srv.psql(
        "DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='retailops') THEN "
        "CREATE ROLE retailops LOGIN CREATEROLE PASSWORD 'retailops'; END IF; END $$;"
    )
    if "retailops" not in srv.psql("SELECT datname FROM pg_database;"):
        srv.psql("CREATE DATABASE retailops OWNER retailops;")
    # CREATE EXTENSION needs superuser on a stock install, so do it here once.
    srv.psql("\\c retailops\nCREATE EXTENSION IF NOT EXISTS vector;")
    upsert_env({
        "DATABASE_URL": f"postgresql+psycopg://retailops:retailops@127.0.0.1:{port}/retailops",
        "DATABASE_URL_RO": f"postgresql+psycopg://retailops_ro:retailops_ro@127.0.0.1:{port}/retailops",
    })
    print(f"postgres running on port {port}; .env updated")


if __name__ == "__main__":
    main()
