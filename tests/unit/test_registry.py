from tools.registry import TOOLS, anthropic_tools, execute


def test_tool_schemas_are_valid_json_schema_objects():
    tools = anthropic_tools()
    assert {t["name"] for t in tools} == set(TOOLS)
    for t in tools:
        assert t["input_schema"]["type"] == "object"
        assert t["description"]


def test_invalid_arguments_return_error_result_not_exception():
    out = execute("sample_rows", {"table": "products", "n": 50}, run_id="t")
    assert out.is_error and out.error_kind == "validation"


def test_unknown_tool():
    out = execute("drop_everything", {}, run_id="t")
    assert out.is_error and out.error_kind == "unknown_tool"


def test_guardrail_error_is_reported_before_touching_the_db():
    out = execute("run_sql", {"query": "DELETE FROM transactions"}, run_id="t")
    assert out.is_error and out.error_kind == "guardrail"
    assert "only SELECT" in out.content


def test_terminal_tools():
    out = execute("ask_clarification", {"question": "By revenue or units?"}, run_id="t")
    assert out.terminal and not out.is_error
