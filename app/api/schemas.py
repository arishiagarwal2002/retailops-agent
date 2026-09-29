"""Request and response models for the HTTP API."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ALLOWED_MODELS = ("claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5")  # when LLM_PROVIDER=anthropic


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    # None = the server's default model. Other models are only accepted for the Anthropic provider,
    # so a public demo can't be pointed at an expensive model.
    model: Literal["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"] | None = None
    use_cache: bool = True


class PendingActionOut(BaseModel):
    id: int
    run_id: str
    action_type: str
    payload: dict
    status: str
    created_at: datetime | None = None
    approved_at: datetime | None = None
    approved_by: str | None = None
    decision_note: str | None = None


class RunOut(BaseModel):
    run_id: str
    question: str
    answer: str | None
    status: str
    citations: list[str] = []
    trace: list[dict] | None = None
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_gbp: float = 0.0
    latency_ms: int = 0
    model: str | None = None
    created_at: datetime | None = None
    pending_actions: list[PendingActionOut] = []
    cached: bool = False


class RunAccepted(BaseModel):
    run_id: str
    status: str = "running"
    poll: str


class RunList(BaseModel):
    items: list[RunOut]
    limit: int
    offset: int


class ApprovalRequest(BaseModel):
    decision: Literal["approve", "reject"]
    note: str | None = Field(default=None, max_length=500)
