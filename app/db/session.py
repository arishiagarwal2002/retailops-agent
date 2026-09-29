"""Two engines: an owner (read-write) engine for app state, and a read-only engine for the SQL tool.

The read-only engine is the second of three independent SQL defences
(AST guard -> read-only role -> statement timeout). Every connection it
hands out is also forced into a read-only transaction with a timeout.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


@lru_cache
def rw_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True, pool_size=5)


@lru_cache
def ro_engine() -> Engine:
    s = get_settings()
    engine = create_engine(s.database_url_ro, pool_pre_ping=True, pool_size=5)

    @event.listens_for(engine, "connect")
    def _harden(dbapi_conn, _record):  # noqa: ANN001
        with dbapi_conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {int(s.sql_timeout_ms)}")
            cur.execute("SET default_transaction_read_only = on")
        dbapi_conn.commit()

    return engine


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=rw_engine(), expire_on_commit=False)


@contextmanager
def rw_session() -> Iterator[Session]:
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
