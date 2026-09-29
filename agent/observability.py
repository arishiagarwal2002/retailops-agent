"""Optional Langfuse export. A no-op unless LANGFUSE_PUBLIC_KEY / SECRET_KEY are set.

Each run becomes one `agent` observation with a child per LLM call
(`generation`, with token usage and cost), tool call (`tool`) and verifier
check (`guardrail`). Exported after the run from the recorded trace, so
tracing can never change agent behaviour or fail a request.
"""
from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import log_event


@lru_cache
def _client():
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host)
    except Exception as e:  # pragma: no cover
        log_event("langfuse_init_failed", error=str(e))
        return None


def export_run(result) -> None:
    lf = _client()
    if lf is None:
        return
    try:
        root = lf.start_observation(name="retailops-run", as_type="agent", input=result.question,
                                    output=result.answer,
                                    metadata={"run_id": result.run_id, "status": result.status,
                                              "config": result.config_name, "model": result.model})
        for e in result.trace:
            if e["type"] == "llm_call":
                obs = root.start_observation(
                    name=f"llm step {e['step']}", as_type="generation", model=result.model,
                    output=e.get("text") or e.get("tool_uses"),
                    usage_details={"input": e["tokens_in"], "output": e["tokens_out"]},
                    metadata={"latency_ms": e["latency_ms"], "stop_reason": e["stop_reason"]})
            elif e["type"] == "tool_call":
                obs = root.start_observation(
                    name=e["name"], as_type="tool", input=e["input"], output=e["output"],
                    level="ERROR" if e["is_error"] else "DEFAULT",
                    metadata={"latency_ms": e["latency_ms"], "error_kind": e["error_kind"]})
            elif e["type"] == "verification":
                obs = root.start_observation(name="verifier", as_type="guardrail", output=e,
                                             level="WARNING" if e["ungrounded"] or e["missing_citation"] else "DEFAULT")
            else:
                obs = root.start_observation(name=e["type"], as_type="event", output=e)
            obs.end()
        root.end()
        lf.flush()
    except Exception as e:  # tracing must never break a request
        log_event("langfuse_export_failed", error=str(e))
