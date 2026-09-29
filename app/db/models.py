"""SQLAlchemy models. Retail tables are read by the agent; the rest are app state."""
from __future__ import annotations

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBED_DIM = 384  # BAAI/bge-small-en-v1.5; must match settings.embed_dim


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------- retail data
class Transaction(Base):
    __tablename__ = "transactions"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    invoice_no: Mapped[str] = mapped_column(String(16))
    stock_code: Mapped[str] = mapped_column(String(32))
    quantity: Mapped[int] = mapped_column(Integer)
    invoice_date: Mapped[datetime] = mapped_column(DateTime)
    unit_price: Mapped[float] = mapped_column(Float)
    customer_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    country: Mapped[str] = mapped_column(String(64))
    is_cancellation: Mapped[bool] = mapped_column(Boolean)

    __table_args__ = (
        Index("ix_transactions_invoice_date", "invoice_date"),
        Index("ix_transactions_stock_code", "stock_code"),
        Index("ix_transactions_customer_id", "customer_id"),
    )


class Product(Base):
    __tablename__ = "products"
    stock_code: Mapped[str] = mapped_column(String(32), primary_key=True)
    description: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64))
    avg_unit_price: Mapped[float] = mapped_column(Float)
    first_seen: Mapped[date] = mapped_column(Date)
    last_seen: Mapped[date] = mapped_column(Date)


class Customer(Base):
    __tablename__ = "customers"
    customer_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    country: Mapped[str] = mapped_column(String(64))
    first_order: Mapped[date] = mapped_column(Date)
    last_order: Mapped[date] = mapped_column(Date)
    order_count: Mapped[int] = mapped_column(Integer)


class StockLevel(Base):
    __tablename__ = "stock_levels"
    stock_code: Mapped[str] = mapped_column(
        String(32), ForeignKey("products.stock_code"), primary_key=True
    )
    on_hand: Mapped[int] = mapped_column(Integer)
    reorder_point: Mapped[int] = mapped_column(Integer)
    lead_time_days: Mapped[int] = mapped_column(Integer)
    supplier_id: Mapped[str] = mapped_column(String(8))


# ---------------------------------------------------------------- retrieval
class PolicyChunk(Base):
    __tablename__ = "policy_chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_name: Mapped[str] = mapped_column(String(128))
    section: Mapped[str] = mapped_column(String(256))
    chunk_text: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(Vector(EMBED_DIM))


class EmbeddingCache(Base):
    __tablename__ = "embedding_cache"
    text_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(128))
    embedding = mapped_column(Vector(EMBED_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------- app state
class Run(Base):
    __tablename__ = "runs"
    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    trace_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_gbp: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    api_key_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_runs_created_at", "created_at"),)


class PendingAction(Base):
    __tablename__ = "pending_actions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.run_id"))
    action_type: Mapped[str] = mapped_column(String(32))
    payload_json: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)


class PurchaseOrder(Base):
    """The only state change the agent can cause, and only via an approved PendingAction."""

    __tablename__ = "purchase_orders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pending_action_id: Mapped[int] = mapped_column(ForeignKey("pending_actions.id"), unique=True)
    stock_code: Mapped[str] = mapped_column(String(32))
    quantity: Mapped[int] = mapped_column(Integer)
    supplier_id: Mapped[str | None] = mapped_column(String(8), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ResponseCache(Base):
    __tablename__ = "response_cache"
    question_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(64))
    response_json: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
