"""Typed graph state. Everything is plain JSON so the checkpointer can persist it."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class AgentState(TypedDict, total=False):
    run_id: str
    question: str
    messages: list[dict]          # Anthropic-format conversation
    trace: list[dict]             # every LLM call, tool call, correction and verification, in order
    steps: int                    # LLM calls made (bounded by step_limit)
    sql_failures: int
    empty_retry_used: bool
    verify_attempts: int
    answer: str
    status: str                   # running | completed | completed_with_warnings | needs_clarification
                                  # | refused | step_limit | error | awaiting_approval
    pending_action_ids: list[int]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    decision: dict                # human decision on a pending action, set on resume


@dataclass(frozen=True)
class AgentConfig:
    model: str | None = None
    self_correct: bool = True
    verifier: bool = True
    name: str = "agent"
