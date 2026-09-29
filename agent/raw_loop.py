"""The agent loop with no framework - kept deliberately, for reference.

This is what LangGraph abstracts: send tools, receive tool_use blocks,
execute them, append tool_result blocks, repeat until the model answers or
the step budget runs out. It has no self-correction, no verifier and no
approval interrupt; graph.py adds those as explicit nodes.

    python -m agent.raw_loop "Which 5 products sold the most units in November 2011?"
"""
from __future__ import annotations

import json
import sys
import uuid

from agent.llm import LLM, make_llm
from agent.prompts import system_prompt
from tools.registry import anthropic_tools, execute

STEP_LIMIT = 8


def raw_agent(question: str, llm: LLM, verbose: bool = True) -> str:
    run_id = f"raw-{uuid.uuid4()}"
    tools = anthropic_tools([n for n in anthropic_tools_names() if n != "draft_reorder"])
    messages: list[dict] = [{"role": "user", "content": question}]
    for step in range(1, STEP_LIMIT + 1):
        resp = llm.chat(system_prompt(), messages, tools)
        messages.append({"role": "assistant", "content": resp.content})
        if not resp.tool_uses:  # the model answered
            return resp.text
        results = []
        for call in resp.tool_uses:
            outcome = execute(call["name"], call["input"], run_id)
            if verbose:
                print(f"[step {step}] {call['name']}({json.dumps(call['input'])[:120]}) "
                      f"-> {'ERROR ' if outcome.is_error else ''}{str(outcome.content)[:160]}")
            if outcome.terminal:
                return json.dumps(outcome.content)
            results.append({
                "type": "tool_result",
                "tool_use_id": call["id"],  # must match the tool_use block
                "content": outcome.content if isinstance(outcome.content, str) else json.dumps(outcome.content, default=str),
                "is_error": outcome.is_error,
            })
        # All results for one assistant turn go back in a single user message.
        messages.append({"role": "user", "content": results})
    return f"Could not complete within {STEP_LIMIT} steps."


def anthropic_tools_names() -> list[str]:
    return [t["name"] for t in anthropic_tools()]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which 5 products sold the most units in November 2011?"
    print(raw_agent(q, make_llm()))
