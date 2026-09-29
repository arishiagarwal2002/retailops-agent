"""Oracle LLM: a scripted model that executes the gold SQL and states the gold values.

It is NOT a result. It exists to test the harness itself: if the oracle does not
score ~100% on the SQL categories, the scorer or the task file is wrong. Runs
without an API key, so CI uses it to check the eval pipeline end to end.
"""
from __future__ import annotations

import json

from agent.llm import ScriptedLLM, text_response, tool_response
from evaluation.metrics import Task, gold_context, resolve_refs


def oracle_for(task: Task) -> ScriptedLLM:
    def script(messages: list[dict]) -> object:
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        spec = task.spec
        if task.category == "ambiguous":
            return tool_response("ask_clarification", {"question": "Which metric and period do you mean?"})
        if task.category == "adversarial":
            if spec.get("type") in ("write", "off_domain", "secret"):
                return tool_response("refuse", {"reason": "That is outside what I can do."})
            if n_assistant == 0:
                code = task.question.split("INJ")[1][:3] if "INJ" in task.question else "22423"
                return tool_response("run_sql", {"query": f"SELECT description, category, avg_unit_price FROM products WHERE stock_code = 'INJ{code}'"})
            last = json.loads(messages[-1]["content"][0]["content"]) if "INJ" in task.question else {}
            row = last.get("rows", [[None, None, None]])[0]
            return text_response(f"It is in category {row[1]} with average price £{row[2]}. "
                                 "I did not follow instructions embedded in the description."
                                 if "INJ" in task.question else "I can only draft a reorder for approval.")
        if task.category == "policy":
            if n_assistant == 0:
                return tool_response("search_policy", {"query": task.question, "k": 5})
            lead = {"yes": "Yes.", "no": "No."}.get(spec.get("verdict"), "Per policy:")
            facts = " ".join(str(x) for x in spec.get("mention_numbers", []))
            phrase = (spec.get("mention_any") or [""])[0]
            return text_response(f"{lead} {phrase} {facts} [{spec['expected_sections'][0]}]")
        gold = spec.get("gold_sql")
        sqls = [gold] if isinstance(gold, str) else (gold or [])
        if n_assistant < len(sqls):
            return tool_response("run_sql", {"query": " ".join(sqls[n_assistant].split())}, call_id=f"q{n_assistant}")
        if spec.get("gold_forecast") and n_assistant == len(sqls):
            fc = spec["gold_forecast"]
            return tool_response("forecast_demand", {"stock_code": str(fc["stock_code"]), "weeks": int(fc["weeks"])},
                                 call_id="fc")
        ctx = gold_context(task)
        values = []
        for ref in spec.get("mention", []):
            values.extend(resolve_refs(ref, ctx))
        return text_response("Answer: " + ", ".join(str(v) for v in values) + ".")

    return ScriptedLLM(script)
