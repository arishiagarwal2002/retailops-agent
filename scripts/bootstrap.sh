#!/bin/sh
# Idempotent database bootstrap: schema, read-only role, data, policy vectors, demo API key.
set -e
python -m alembic upgrade head
python -m scripts.create_ro_role
if [ "$(python -c 'from sqlalchemy import text; from app.db.session import rw_engine; print(rw_engine().connect().execute(text("select count(*) from transactions")).scalar())')" = "0" ]; then
  python -m data.load
fi
python -m retrieval.store
if [ -n "$DEMO_API_KEY_NAME" ]; then python -m scripts.create_api_key "$DEMO_API_KEY_NAME"; fi
echo "bootstrap complete"
