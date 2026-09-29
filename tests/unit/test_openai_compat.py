"""Translation between the agent's Anthropic-shaped messages and the OpenAI chat format."""
import json
from types import SimpleNamespace as NS

from agent.openai_compat import OpenAICompatLLM, from_openai_message, to_openai_messages, to_openai_tools


def test_messages_round_trip_through_openai_format():
    messages = [
        {"role": "user", "content": "stock of 85123A?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "c1", "name": "run_sql", "input": {"query": "SELECT 1"}},
            {"type": "tool_use", "id": "c2", "name": "get_schema", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "column x does not exist", "is_error": True},
            {"type": "tool_result", "tool_use_id": "c2", "content": '{"tables": {}}', "is_error": False},
            {"type": "text", "text": "Fix the query and retry."},
        ]},
    ]
    out = to_openai_messages("SYS", messages)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "tool", "user"]
    assert out[2]["content"] == "Checking."
    assert [c["id"] for c in out[2]["tool_calls"]] == ["c1", "c2"]
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"query": "SELECT 1"}
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "ERROR: column x does not exist"}
    assert out[5]["content"] == "Fix the query and retry."


def test_tools_become_functions():
    tools = to_openai_tools([{"name": "run_sql", "description": "d", "input_schema": {"type": "object"}}])
    assert tools == [{"type": "function", "function": {"name": "run_sql", "description": "d",
                                                        "parameters": {"type": "object"}}}]


def _call(name, args, cid="x1"):
    return NS(id=cid, function=NS(name=name, arguments=args))


def test_response_with_tool_calls_and_bad_json():
    msg = NS(content=None, tool_calls=[_call("run_sql", '{"query": "SELECT 1"}'), _call("get_schema", "{not json", "x2")])
    r = from_openai_message(msg, NS(prompt_tokens=50, completion_tokens=7))
    assert r.stop_reason == "tool_use"
    assert r.tool_uses[0]["input"] == {"query": "SELECT 1"}
    assert r.tool_uses[1]["input"] == {"_raw": "{not json"}  # becomes a validation error downstream
    assert (r.usage.input_tokens, r.usage.output_tokens) == (50, 7)


def test_text_response_and_free_cost():
    r = from_openai_message(NS(content="Answer.", tool_calls=None), None)
    assert r.stop_reason == "end_turn" and r.text == "Answer."
    assert r.usage.cost_usd("gemini-3.8-flash") == 0.0


def test_client_sends_translated_request():
    llm = OpenAICompatLLM("gemini", api_key="test")
    seen = {}

    def create(**kw):
        seen.update(kw)
        parsed = NS(choices=[NS(message=NS(content="ok", tool_calls=None))],
                    usage=NS(prompt_tokens=1, completion_tokens=1))
        return NS(headers={}, parse=lambda: parsed)

    llm._client = NS(chat=NS(completions=NS(with_raw_response=NS(create=create))))
    r = llm.chat("SYS", [{"role": "user", "content": "hi"}],
                 [{"name": "t", "description": "d", "input_schema": {"type": "object"}}])
    assert r.text == "ok"
    assert seen["model"] == "gemini-3.8-flash"
    assert seen["messages"][0] == {"role": "system", "content": "SYS"}
    assert seen["tools"][0]["function"]["name"] == "t"
