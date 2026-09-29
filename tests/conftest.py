import pytest
from sqlalchemy import text


def _db_ready() -> bool:
    try:
        from app.db.session import ro_engine

        with ro_engine().connect() as conn:
            return conn.execute(text("SELECT count(*) FROM products")).scalar() > 0
    except Exception:
        return False


DB_READY = _db_ready()
requires_db = pytest.mark.skipif(not DB_READY, reason="Postgres with loaded data not available")
