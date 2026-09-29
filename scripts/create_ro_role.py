"""Create (or update) the read-only Postgres role the SQL tool connects as.

Idempotent. Takes the role name and password from DATABASE_URL_RO and runs
as the owner in DATABASE_URL, which needs CREATEROLE.

    python -m scripts.create_ro_role
"""
from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.core.config import get_settings

RETAIL_TABLES = ("transactions", "products", "customers", "stock_levels")


def main() -> None:
    s = get_settings()
    ro = make_url(s.database_url_ro)
    role, password = ro.username, ro.password or ""
    engine = create_engine(s.database_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        exists = conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": role}).scalar()
        pw = password.replace("'", "''")
        if exists:
            conn.execute(text(f"ALTER ROLE {role} WITH LOGIN PASSWORD '{pw}'"))
        else:
            conn.execute(text(f"CREATE ROLE {role} WITH LOGIN PASSWORD '{pw}'"))
        # Defaults applied to every session of this role, independent of the client.
        conn.execute(text(f"ALTER ROLE {role} SET statement_timeout = '{s.sql_timeout_ms}ms'"))
        conn.execute(text(f"ALTER ROLE {role} SET default_transaction_read_only = on"))
        db = conn.execute(text("SELECT current_database()")).scalar()
        conn.execute(text(f"GRANT CONNECT ON DATABASE {db} TO {role}"))
        conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {role}"))
        present = conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename = ANY(:t)"),
            {"t": list(RETAIL_TABLES)},
        ).scalars().all()
        if present:
            conn.execute(text(f"GRANT SELECT ON {', '.join(present)} TO {role}"))
    print(f"read-only role '{role}' ready (SELECT on: {', '.join(present) or 'none yet'})")


if __name__ == "__main__":
    main()
