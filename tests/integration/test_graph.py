"""Full-graph tests against the real database with a scripted (mocked) LLM."""
from sqlalchemy import select

from agent.llm import ScriptedLLM, text_response, tool_response
from agent.runner import resume_run, run_agent
from agent.state import AgentConfig
from app.db.models import PendingAction, PurchaseOrder
from app.db.session import rw_session
from tests.conftest import requires_db

pytestmark = requires_db

GOOD_SQL = "SELECT on_hand, reorder_point FROM stock_levels WHERE stock_code = '85123A'"


def by_turn(*responses):
    def script(messages):
        n = sum(1 for m in messages if m["role"] == "assistant")
        return responses[min(n, len(responses) - 1)]
    return ScriptedLLM(script)


def _on_hand() -> int:
    from sqlalchemy import text

    from app.db.session import ro_engine

    with ro_engine().connect() as c:
        return c.execute(text("SELECT on_hand FROM stock_levels WHERE stock_code='85123A'")).scalar()


def test_sql_error_triggers_self_correction_then_answer():
    on_hand = _on_hand()
    llm = by_turn(
        tool_response("run_sql", {"query": "SELECT onhand FROM stock_levels"}),
        tool_response("run_sql", {"query": GOOD_SQL}),
        text_response(f"85123A has {on_hand:,} units on hand."),
    )
    r = run_agent("stock of 85123A?", llm)
    assert r.status == "completed"
    assert r.tool_calls == ["run_sql", "run_sql"]
    corrections = [e for e in r.trace if e["type"] == "self_correction"]
    assert corrections and "attempt 1 of 2" in corrections[0]["hints"][0]
    # the model saw the verbatim database error
    assert "does not exist" in llm.calls[1][-1]["content"][0]["content"]


def test_without_self_correction_error_detail_is_hidden():
    llm = by_turn(tool_response("run_sql", {"query": "SELECT onhand FROM stock_levels"}), text_response("Unknown."))
    run_agent("stock?", llm, AgentConfig(self_correct=False))
    assert llm.calls[1][-1]["content"][0]["content"] == "error: tool call failed"


def test_verifier_catches_invented_number_and_regenerates_once():
    on_hand = _on_hand()
    llm = by_turn(
        tool_response("run_sql", {"query": GOOD_SQL}),
        text_response("85123A has 987,654 units on hand."),
        text_response(f"85123A has {on_hand:,} units on hand."),
    )
    r = run_agent("stock of 85123A?", llm)
    verifications = [e for e in r.trace if e["type"] == "verification"]
    assert verifications[0]["ungrounded"] == ["987,654"]
    assert verifications[1]["ungrounded"] == []
    assert r.status == "completed"


def test_verifier_warns_after_second_failure():
    llm = by_turn(tool_response("run_sql", {"query": GOOD_SQL}), text_response("It has 987,654 units."))
    r = run_agent("stock?", llm)
    assert r.status == "completed_with_warnings"
    assert "unverified values: 987,654" in r.answer


def test_step_limit_returns_budget_message_not_an_answer():
    llm = by_turn(tool_response("get_schema", {}))
    r = run_agent("loop forever", llm)
    assert r.status == "step_limit"
    assert r.steps == 8


def test_clarification_and_refusal_end_the_run():
    r = run_agent("best products?", by_turn(tool_response("ask_clarification", {"question": "By revenue or units?"})))
    assert r.status == "needs_clarification" and "revenue or units" in r.answer
    r = run_agent("delete stuff", by_turn(tool_response("refuse", {"reason": "the data is read-only"})))
    assert r.status == "refused"


def test_reorder_pauses_for_approval_and_resumes():
    llm = by_turn(
        tool_response("draft_reorder", {"stock_code": "85123A", "qty": 120, "reason": "below reorder point"}),
        text_response("I drafted a reorder of 120 units of 85123A for approval; nothing has been ordered."),
    )
    r = run_agent("reorder 85123A", llm)
    assert r.status == "awaiting_approval"
    action_id = r.pending_action_ids[0]
    with rw_session() as s:
        assert s.get(PendingAction, action_id).status == "pending"
        assert s.execute(select(PurchaseOrder).where(PurchaseOrder.pending_action_id == action_id)).first() is None
    resumed = resume_run(r.run_id, {"action_id": action_id, "decision": "rejected", "by": "pytest"})
    assert resumed.status == "completed"
    assert "rejected by pytest" in resumed.answer


def test_parallel_tool_calls_return_in_one_message():
    from agent.llm import LLMResponse, Usage

    both = LLMResponse([
        {"type": "tool_use", "id": "a", "name": "get_schema", "input": {}},
        {"type": "tool_use", "id": "b", "name": "run_sql", "input": {"query": GOOD_SQL}},
    ], "tool_use", Usage(10, 10))
    llm = by_turn(both, text_response("Done."))
    run_agent("two tools", llm)
    results = llm.calls[1][-1]["content"]
    assert [b["tool_use_id"] for b in results if b["type"] == "tool_result"] == ["a", "b"]


def test_full_run_through_openai_compatible_adapter():
    """The whole graph works through the Gemini/Groq adapter (fake HTTP client, real DB and tools)."""
    from types import SimpleNamespace as NS

    from agent.openai_compat import OpenAICompatLLM

    on_hand = _on_hand()
    turns = iter([
        NS(content=None, tool_calls=[NS(id="g1", function=NS(name="run_sql", arguments=f'{{"query": "{GOOD_SQL}"}}'))]),
        NS(content=f"85123A has {on_hand:,} units on hand.", tool_calls=None),
    ])
    llm = OpenAICompatLLM("gemini", api_key="test", min_interval_s=0)

    def create(**kw):
        parsed = NS(choices=[NS(message=next(turns))], usage=NS(prompt_tokens=10, completion_tokens=5))
        return NS(headers={}, parse=lambda: parsed)

    llm._client = NS(chat=NS(completions=NS(with_raw_response=NS(create=create))))
    r = run_agent("stock of 85123A?", llm)
    assert r.status == "completed", r.answer
    assert r.tool_calls == ["run_sql"]
    assert r.cost_gbp == 0.0
