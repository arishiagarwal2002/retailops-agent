"""LLM clients. Messages use the Anthropic Messages API shape throughout.

- AnthropicLLM: the real client (Claude via the official SDK).
- ScriptedLLM: deterministic stand-in for unit/integration tests and CI.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from app.core.config import get_settings

# USD per million tokens (input, output). Cache reads bill at 0.1x input, writes at 1.25x.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-5-5": (4.00, 20.00),
    "scripted": (0.0, 0.0),
}
FREE_PROVIDERS = {"gemini", "groq", "ollama"}  # billed at 0: free tier or local


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens

    def cost_usd(self, model: str) -> float:
        if not model.startswith("claude-"):
            return 0.0  # free-tier (Gemini/Groq) or local (Ollama) models
        pin, pout = PRICING_USD_PER_MTOK.get(model, (5.0, 25.0))
        return (
            self.input_tokens * pin
            + self.cache_write_tokens * pin * 1.25
            + self.cache_read_tokens * pin * 0.1
            + self.output_tokens * pout
        ) / 1_000_000

    def cost_gbp(self, model: str) -> float:
        return self.cost_usd(model) * get_settings().usd_to_gbp


@dataclass
class LLMResponse:
    content: list[dict]  # Anthropic content blocks as plain dicts
    stop_reason: str
    usage: Usage = field(default_factory=Usage)

    @property
    def text(self) -> str:
        return "\n".join(b["text"] for b in self.content if b.get("type") == "text").strip()

    @property
    def tool_uses(self) -> list[dict]:
        return [b for b in self.content if b.get("type") == "tool_use"]


class LLM(Protocol):
    model: str

    def chat(self, system: str, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int | None = None) -> LLMResponse: ...


class AnthropicLLM:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        import anthropic

        s = get_settings()
        self.model = model or s.default_model
        self.max_tokens = s.llm_max_tokens
        self._client = anthropic.Anthropic(api_key=api_key or s.anthropic_api_key, max_retries=3, timeout=120)

    def chat(self, system: str, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int | None = None) -> LLMResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            # The system prompt and tool list are identical on every call, so cache them.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        resp = self._client.messages.create(**kwargs)
        u = resp.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        content = [b.model_dump(exclude_none=True) for b in resp.content]
        return LLMResponse(content=content, stop_reason=resp.stop_reason or "end_turn", usage=usage)


class ScriptedLLM:
    """Replays responses produced by a function of the conversation so far.

    `script(messages) -> LLMResponse`. Used by tests and the CI smoke run when
    no API key is configured.
    """

    model = "scripted"

    def __init__(self, script: Callable[[list[dict]], LLMResponse]):
        self._script = script
        self.calls: list[list[dict]] = []

    def chat(self, system: str, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int | None = None) -> LLMResponse:
        self.calls.append(messages)
        return self._script(messages)


def text_response(text: str, tokens: tuple[int, int] = (100, 20)) -> LLMResponse:
    return LLMResponse([{"type": "text", "text": text}], "end_turn", Usage(*tokens))


def tool_response(name: str, args: dict, call_id: str = "t1", preface: str = "",
                  tokens: tuple[int, int] = (100, 20)) -> LLMResponse:
    blocks: list[dict] = [{"type": "text", "text": preface}] if preface else []
    blocks.append({"type": "tool_use", "id": call_id, "name": name, "input": args})
    return LLMResponse(blocks, "tool_use", Usage(*tokens))


def make_llm(model: str | None = None) -> LLM:
    """Build the configured provider's client (LLM_PROVIDER in .env)."""
    s = get_settings()
    if s.llm_provider == "anthropic":
        return AnthropicLLM(model=model)
    from agent.openai_compat import OpenAICompatLLM

    key = {"gemini": s.gemini_api_key, "groq": s.groq_api_key}.get(s.llm_provider)
    base = s.ollama_base_url if s.llm_provider == "ollama" else None
    return OpenAICompatLLM(s.llm_provider, model=model or s.default_model, api_key=key, base_url=base,
                           max_tokens=s.llm_max_tokens, min_interval_s=s.llm_min_interval_s)
