"""Initial schema: retail tables, pgvector, app state.

Revision ID: 0001
Revises:
Create Date: 2026-09-28
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBED_DIM = 384
RETAIL_TABLES = ("transactions", "products", "customers", "stock_levels")


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "transactions",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("invoice_no", sa.String(16), nullable=False),
        sa.Column("stock_code", sa.String(32), nullable=False),
        sa.Column("quantity", sa.Integer, nullable=False),
        sa.Column("invoice_date", sa.DateTime, nullable=False),
        sa.Column("unit_price", sa.Float, nullable=False),
        sa.Column("customer_id", sa.Integer, nullable=True),
        sa.Column("country", sa.String(64), nullable=False),
        sa.Column("is_cancellation", sa.Boolean, nullable=False),
    )
    op.create_index("ix_transactions_invoice_date", "transactions", ["invoice_date"])
    op.create_index("ix_transactions_stock_code", "transactions", ["stock_code"])
    op.create_index("ix_transactions_customer_id", "transactions", ["customer_id"])

    op.create_table(
        "products",
        sa.Column("stock_code", sa.String(32), primary_key=True),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("category", sa.String(64), nullable=False),
        sa.Column("avg_unit_price", sa.Float, nullable=False),
        sa.Column("first_seen", sa.Date, nullable=False),
        sa.Column("last_seen", sa.Date, nullable=False),
    )
    op.create_table(
        "customers",
        sa.Column("customer_id", sa.Integer, primary_key=True),
        sa.Column("country", sa.String(64), nullable=False),
        sa.Column("first_order", sa.Date, nullable=False),
        sa.Column("last_order", sa.Date, nullable=False),
        sa.Column("order_count", sa.Integer, nullable=False),
    )
    op.create_table(
        "stock_levels",
        sa.Column("stock_code", sa.String(32), sa.ForeignKey("products.stock_code"), primary_key=True),
        sa.Column("on_hand", sa.Integer, nullable=False),
        sa.Column("reorder_point", sa.Integer, nullable=False),
        sa.Column("lead_time_days", sa.Integer, nullable=False),
        sa.Column("supplier_id", sa.String(8), nullable=False),
    )

    op.create_table(
        "policy_chunks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("doc_name", sa.String(128), nullable=False),
        sa.Column("section", sa.String(256), nullable=False),
        sa.Column("chunk_text", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBED_DIM)),
    )
    op.create_table(
        "embedding_cache",
        sa.Column("text_hash", sa.String(64), primary_key=True),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("embedding", Vector(EMBED_DIM)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "runs",
        sa.Column("run_id", sa.String(36), primary_key=True),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("answer", sa.Text),
        sa.Column("trace_json", JSONB),
        sa.Column("tool_calls", sa.Integer, server_default="0"),
        sa.Column("tokens_in", sa.Integer, server_default="0"),
        sa.Column("tokens_out", sa.Integer, server_default="0"),
        sa.Column("cost_gbp", sa.Float, server_default="0"),
        sa.Column("latency_ms", sa.Integer, server_default="0"),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("model", sa.String(64)),
        sa.Column("api_key_id", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_runs_created_at", "runs", ["created_at"])

    op.create_table(
        "pending_actions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("runs.run_id"), nullable=False),
        sa.Column("action_type", sa.String(32), nullable=False),
        sa.Column("payload_json", JSONB, nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("approved_at", sa.DateTime(timezone=True)),
        sa.Column("approved_by", sa.String(64)),
        sa.Column("decision_note", sa.Text),
    )
    op.create_table(
        "purchase_orders",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "pending_action_id", sa.Integer, sa.ForeignKey("pending_actions.id"), unique=True, nullable=False
        ),
        sa.Column("stock_code", sa.String(32), nullable=False),
        sa.Column("quantity", sa.Integer, nullable=False),
        sa.Column("supplier_id", sa.String(8)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("key_hash", sa.String(64), unique=True, nullable=False),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "response_cache",
        sa.Column("question_hash", sa.String(64), primary_key=True),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("response_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Least privilege: the read-only role (created by scripts/create_ro_role.py)
    # can SELECT the four retail tables and nothing else.
    op.execute(
        f"""
        DO $$
        BEGIN
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'retailops_ro') THEN
            GRANT USAGE ON SCHEMA public TO retailops_ro;
            GRANT SELECT ON {", ".join(RETAIL_TABLES)} TO retailops_ro;
          END IF;
        END $$;
        """
    )


def downgrade() -> None:
    for table in (
        "response_cache",
        "api_keys",
        "purchase_orders",
        "pending_actions",
        "runs",
        "embedding_cache",
        "policy_chunks",
        "stock_levels",
        "customers",
        "products",
        "transactions",
    ):
        op.drop_table(table)
