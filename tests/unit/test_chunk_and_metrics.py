from pathlib import Path

from evaluation.metrics import is_subsequence, load_tasks, mentions_value, recall_at_5, verdict_of
from retrieval.chunk import chunk_directory, chunk_markdown

DOC = """# Returns Policy

## 3. Refunds

### 3.1 Refund method
Refunds go to the original payment method.

### 3.2 Refund timing
We refund within 14 days.
"""


def test_chunks_carry_section_citations():
    chunks = chunk_markdown(DOC)
    assert [c.citation for c in chunks] == ["[Returns Policy §3.1]", "[Returns Policy §3.2]"]
    assert "14 days" in chunks[1].text


def test_long_sections_are_split_with_overlap():
    body = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_markdown(f"# Doc\n\n## 1. A\n\n### 1.1 Long\n{body}\n")
    assert len(chunks) >= 3
    assert all(c.section == "§1.1 Long" for c in chunks)


def test_policy_corpus_chunks():
    chunks = chunk_directory(Path("data/policy"))
    docs = {c.doc_name for c in chunks}
    assert docs == {"Returns Policy", "Supplier Terms", "UK Consumer Rights Summary"}
    assert len(chunks) > 50


def test_benchmark_has_80_tasks_with_unique_ids():
    tasks = load_tasks()
    assert len(tasks) == 80
    assert len({t.id for t in tasks}) == 80
    counts = {c: sum(t.category == c for t in tasks) for c in {t.category for t in tasks}}
    assert counts == {"sql_single": 30, "multi_step": 15, "policy": 15, "ambiguous": 10, "adversarial": 10}


def test_mentions_value_numeric_tolerance_and_strings():
    assert mentions_value("Revenue was £1,452,116.", 1452115.98)
    assert mentions_value("about £1.45m", 1452115.98)
    assert not mentions_value("about £1.2m", 1452115.98)
    assert mentions_value("The winner is the Netherlands.", "Netherlands")
    assert mentions_value("Top code 85123A", "85123A")


def test_verdict_detection_uses_first_sentence():
    assert verdict_of("No - the 30-day window has expired. Yes they could have...") == "no"
    assert verdict_of("Yes, they are still within the Christmas returns period.") == "yes"
    assert verdict_of("They are not eligible for a return.") == "no"
    assert verdict_of("Refunds are paid to the original card.") is None


def test_subsequence_and_recall():
    assert is_subsequence(["run_sql", "forecast_demand"], ["run_sql", "run_sql", "forecast_demand"])
    assert not is_subsequence(["forecast_demand", "run_sql"], ["run_sql", "forecast_demand"])
    assert recall_at_5(["Returns Policy §2.1", "Returns Policy §1.3"], ["Returns Policy §2.1"]) == 0.5


def test_refusal_detected_with_curly_apostrophe():
    """The model writes "can't" with U+2019; the decline detector must still match."""
    from agent.verifier import normalize
    from evaluation.metrics import DECLINE_RE

    assert DECLINE_RE.search(normalize("I’m sorry, but I can’t delete data from the database."))


def test_verdict_and_mentions_handle_unicode_formatting():
    from evaluation.metrics import mentions_value, verdict_of

    assert mentions_value("Stock is 1 590 units.", 1590)
    assert verdict_of("Yes – the order can still be returned.") == "yes"
