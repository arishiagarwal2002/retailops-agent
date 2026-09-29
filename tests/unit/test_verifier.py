from agent.verifier import extract_numbers, has_citation, ungrounded_numbers

SOURCES = ['{"rows": [["85123A", 12345.678, 0.2534, 1576.9, 84.67]]}']


def test_exact_rounded_and_abbreviated_numbers_are_grounded():
    answer = "Revenue was £12,345.68 (about £12.3k); forecast 1,577 units; UK share 84.67%."
    bad, checked = ungrounded_numbers(answer, SOURCES)
    assert bad == []
    assert checked == 4


def test_fraction_rendered_as_percent_is_grounded():
    bad, _ = ungrounded_numbers("Share was 25.3%.", SOURCES)
    assert bad == []


def test_invented_number_is_flagged():
    bad, _ = ungrounded_numbers("We sold 9,999 units.", SOURCES)
    assert bad == ["9,999"]


def test_small_integers_years_dates_and_sections_are_exempt():
    answer = "The top 5 items in 2011 as of 2011-12-09 [Returns Policy §3.2]."
    bad, checked = ungrounded_numbers(answer, SOURCES)
    assert bad == [] and checked == 0


def test_numbers_from_question_are_allowed():
    bad, _ = ungrounded_numbers("An order from 20 days ago is fine.", SOURCES, question="order from 20 days ago")
    assert bad == []


def test_citation_detection():
    assert has_citation("Refunds take 14 days [Returns Policy §3.2].")
    assert has_citation("[UK Consumer Rights Summary §2.1]")
    assert not has_citation("See section 3.2 of the returns policy.")


def test_million_suffix():
    nums = extract_numbers("Revenue was £1.45m")
    assert nums[0].value * nums[0].multiplier == 1.45e6


# --- Regression: real formatting seen from models, which used to cause false flags.
def test_narrow_nobreak_space_thousands_separator():
    """Models write "1 590" with U+202F; it must read as 1590, not 1 and 590."""
    answer = "Stock on hand: 1 590 units, reorder point 2 184."
    bad, checked = ungrounded_numbers(answer, ['{"rows": [[1590, 2184]]}'])
    assert bad == [] and checked == 2


def test_adjacent_json_numbers_are_not_merged():
    """Stripping commas globally turned [[1590,2184]] into 15902184."""
    bad, _ = ungrounded_numbers("It is 1,590 units.", ['{"rows": [[1590, 2184]]}'])
    assert bad == []
    bad, _ = ungrounded_numbers("It is 15,902,184 units.", ['{"rows": [[1590, 2184]]}'])
    assert bad == ["15,902,184"]


def test_citation_bracket_variants():
    assert has_citation("within 30 days【Returns Policy §2.1】")
    assert has_citation("[Returns Policy §2.1]")
    assert not has_citation("see section 3.2 of the policy")


def test_curly_apostrophe_normalised():
    from agent.verifier import normalize

    assert normalize("I’m sorry, but I can’t do that.") == "I'm sorry, but I can't do that."
