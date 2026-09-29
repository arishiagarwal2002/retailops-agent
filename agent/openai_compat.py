"""Client for OpenAI-compatible chat APIs: Google Gemini (free tier), Groq (free tier), Ollama (local).

The agent keeps Anthropic-shaped messages internally (content blocks, tool_use,
tool_result). This adapter translates them to the OpenAI chat format on the way
out and translates responses back, so the graph, verifier and eval harness are
provider-agnostic.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid

from agent.llm import LLMResponse, Usage
from app.core.logging import log_event

PROVIDERS: dict[str, dict] = {
    # min_interval_s is set by the binding free-tier limit, measured rather than guessed.
    # Gemini: 5 requests/min per model. Groq: 8,000 tokens/min, and this agent averages ~2,400
    # tokens per request (see docs/failure_analysis.md), so ~3.3 requests/min is sustainable.
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "default_model": "gemini-3.8-flash", "key_env": "GEMINI_API_KEY", "min_interval_s": 12.5},
    "groq": {"base_url": "https://api.groq.com/openai/v1",
             "default_model": "openai/gpt-oss-120b", "key_env": "GROQ_API_KEY", "min_interval_s": 18.0},
    "ollama": {"base_url": "http://localhost:11434/v1",
               "default_model": "qwen2.5:7b", "key_env": None, "min_interval_s": 0.0},
}

# Waiting is the normal state on a token-capped free tier, so the give-up rule is a time budget
# rather than an attempt count: a single heavy request can legitimately need several minutes of
# token-bucket refill. Only genuinely stuck requests should fail.
RATE_LIMIT_BUDGET_S = 900.0
MAX_RATE_LIMIT_WAITS = 60


class QuotaExhaustedError(RuntimeError):
    """The provider's daily free quota is used up; waiting a minute will not help."""


class _Pacer:
    """Process-wide spacing of requests per (provider, model), shared across threads.

    Two layers: a fixed minimum interval, and an adaptive hold learned from the provider's
    x-ratelimit-* headers. Free tiers are usually capped on tokens-per-minute rather than
    requests, so waiting for the token bucket to refill beats absorbing a stream of 429s.
    """

    _lock = threading.Lock()
    _next_slot: dict[str, float] = {}
    # Resume only once the token bucket has this much of its limit free.
    LOW_TOKEN_FRACTION = 0.25

    @classmethod
    def wait(cls, key: str, interval: float) -> None:
        with cls._lock:
            now = time.monotonic()
            slot = max(now, cls._next_slot.get(key, 0.0))
            cls._next_slot[key] = slot + max(interval, 0.0)
        if slot > now:
            time.sleep(slot - now)

    @classmethod
    def observe_headers(cls, key: str, headers) -> None:
        """Hold the next request when the remaining token budget is nearly spent."""
        try:
            limit = _parse_int(headers.get("x-ratelimit-limit-tokens"))
            remaining = _parse_int(headers.get("x-ratelimit-remaining-tokens"))
            reset_s = _parse_duration_s(headers.get("x-ratelimit-reset-tokens"))
        except Exception:
            return
        if not limit or remaining is None or reset_s is None:
            return
        if remaining < limit * cls.LOW_TOKEN_FRACTION:
            with cls._lock:
                cls._next_slot[key] = max(cls._next_slot.get(key, 0.0), time.monotonic() + reset_s + 0.5)


def _parse_int(value) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _parse_duration_s(value) -> float | None:
    """Parse Groq-style durations: '577ms', '7.66s', '2h44m9.6s'."""
    if value is None:
        return None
    text = str(value).strip()
    if text.endswith("ms") and "h" not in text and "m" not in text[:-2]:
        try:
            return float(text[:-2]) / 1000.0
        except ValueError:
            return None
    total, found = 0.0, False
    for amount, unit in re.findall(r"([\d.]+)\s*(h|m|s)", text):
        total += float(amount) * {"h": 3600, "m": 60, "s": 1}[unit]
        found = True
    return total if found else None


def _retry_delay_s(message: str) -> float | None:
    m = re.search(r"retry in ([\d.]+)s", message) or re.search(r"'retryDelay': '([\d.]+)s'", message)
    return float(m.group(1)) if m else None


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content if b.get("type") == "text")


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if m["role"] == "assistant":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            text = "\n".join(b["text"] for b in blocks if b.get("type") == "text" and b.get("text"))
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
                     for b in blocks if b.get("type") == "tool_use"]
            msg: dict = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        if isinstance(content, str):
            out.append({"role": "user", "content": content})
            continue
        # A user turn may hold tool results plus extra text (self-correction hints).
        extra = []
        for b in content:
            if b.get("type") == "tool_result":
                body = b["content"] if isinstance(b["content"], str) else _text_of(b["content"])
                if b.get("is_error"):
                    body = f"ERROR: {body}"
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": body})
            elif b.get("type") == "text":
                extra.append(b["text"])
        if extra:
            out.append({"role": "user", "content": "\n".join(extra)})
    return out


def to_openai_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools]


def from_openai_message(message, usage) -> LLMResponse:
    blocks: list[dict] = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    for call in message.tool_calls or []:
        try:
            args = json.loads(call.function.arguments or "{}")
            if not isinstance(args, dict):
                args = {"_raw": args}
        except json.JSONDecodeError:
            # Malformed JSON goes back as an argument-validation error, like any bad tool call.
            args = {"_raw": call.function.arguments}
        blocks.append({"type": "tool_use", "id": call.id or f"call_{uuid.uuid4().hex[:12]}",
                       "name": call.function.name, "input": args})
    has_tools = any(b["type"] == "tool_use" for b in blocks)
    u = Usage(input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
              output_tokens=getattr(usage, "completion_tokens", 0) or 0) if usage else Usage()
    return LLMResponse(content=blocks or [{"type": "text", "text": ""}],
                       stop_reason="tool_use" if has_tools else "end_turn", usage=u)


class OpenAICompatLLM:
    def __init__(self, provider: str, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, max_tokens: int = 4096, min_interval_s: float | None = None):
        from openai import OpenAI

        cfg = PROVIDERS[provider]
        self.provider = provider
        self.model = model or cfg["default_model"]
        self.max_tokens = max_tokens
        self.min_interval_s = cfg["min_interval_s"] if min_interval_s is None else min_interval_s
        # 429s are handled below using the server's own retry delay; SDK retries only cover network errors.
        self._client = OpenAI(api_key=api_key or "ollama", base_url=base_url or cfg["base_url"],
                              max_retries=2, timeout=120)

    def chat(self, system: str, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int | None = None) -> LLMResponse:
        kwargs: dict = {"model": self.model, "messages": to_openai_messages(system, messages),
                        "max_tokens": max_tokens or self.max_tokens}
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
        from openai import InternalServerError, RateLimitError

        key = f"{self.provider}:{self.model}"
        waited = 0.0
        for _ in range(MAX_RATE_LIMIT_WAITS):
            _Pacer.wait(key, self.min_interval_s)
            try:
                # with_raw_response exposes the x-ratelimit-* headers, which let us pace off the
                # provider's own accounting instead of a guessed interval.
                raw = self._client.chat.completions.with_raw_response.create(**kwargs)
                _Pacer.observe_headers(key, raw.headers)
                resp = raw.parse()
                return from_openai_message(resp.choices[0].message, resp.usage)
            except RateLimitError as e:
                msg = str(e)
                delay = _retry_delay_s(msg)
                log_event("llm_rate_limited", provider=self.provider, model=self.model, retry_delay_s=delay,
                          detail=msg[:600])
                # A 429 can mention the daily quota and still ask for a short wait; only give up when
                # the server offers no near-term retry.
                if "PerDay" in msg and (delay is None or delay > 300):
                    raise QuotaExhaustedError(
                        f"{self.provider} daily free quota for {self.model} is used up; it resets daily. "
                        "Try again tomorrow, or set LLM_MODEL to another model (quotas are per model).") from e
                nap = min(delay or 20.0, 90.0) + 1.0
                if waited + nap > RATE_LIMIT_BUDGET_S:
                    raise RuntimeError(
                        f"{self.provider} rate-limited {self.model} for over "
                        f"{RATE_LIMIT_BUDGET_S / 60:.0f} min; giving up on this request") from e
                waited += nap
                time.sleep(nap)
            except InternalServerError as e:  # noqa: PERF203
                # 503 "high demand" on free tiers is transient: back off and try again.
                if getattr(e, "status_code", 500) not in (500, 503):
                    raise
                waited += 15.0
                if waited > RATE_LIMIT_BUDGET_S:
                    raise
                time.sleep(15.0)
        raise RuntimeError(f"{self.provider} kept rate-limiting {self.model} after "
                           f"{MAX_RATE_LIMIT_WAITS} waits ({waited:.0f}s)")
