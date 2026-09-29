"""Load the cleaned Parquet files into Postgres with COPY.

Truncates and reloads the four retail tables, then ANALYZEs them so the
planner has fresh statistics.

    python -m data.load
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import psycopg
from sqlalchemy.engine import make_url

from app.core.config import get_settings

PROCESSED = Path(__file__).parent / "processed"

TABLES: list[tuple[str, list[str]]] = [
    ("products", ["stock_code", "description", "category", "avg_unit_price", "first_seen", "last_seen"]),
    ("stock_levels", ["stock_code", "on_hand", "reorder_point", "lead_time_days", "supplier_id"]),
    ("customers", ["customer_id", "country", "first_order", "last_order", "order_count"]),
    ("transactions", ["invoice_no", "stock_code", "quantity", "invoice_date", "unit_price",
                      "customer_id", "country", "is_cancellation"]),
]


def psycopg_dsn(sqlalchemy_url: str) -> str:
    return make_url(sqlalchemy_url).set(drivername="postgresql").render_as_string(hide_password=False)


def copy_frame(cur: psycopg.Cursor, table: str, cols: list[str], df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df[cols].to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '\\N')") as cp:
        while chunk := buf.read(1 << 20):
            cp.write(chunk)


def main() -> None:
    dsn = psycopg_dsn(get_settings().database_url)
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE transactions, stock_levels, customers, products RESTART IDENTITY CASCADE")
        for table, cols in TABLES:
            df = pd.read_parquet(PROCESSED / f"{table}.parquet")
            if "customer_id" in df:
                df["customer_id"] = df["customer_id"].astype("Int64")
            copy_frame(cur, table, cols, df)
            print(f"loaded {len(df):>9,} rows into {table}")
        conn.commit()
    with psycopg.connect(dsn, autocommit=True) as conn:
        for table, _ in TABLES:
            conn.execute(f"ANALYZE {table}")
    print("done")


if __name__ == "__main__":
    main()
