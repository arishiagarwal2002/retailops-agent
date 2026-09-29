"""Deterministic groundedness checks.

numeric: every number in the answer must match a number in some tool output
(or the question), allowing for rounding at the precision the answer uses and
for unit suffixes (k, m) and percentages. Integers 0-10 and years are exempt:
they are mostly ordinals and list sizes and would dominate the false-positive rate.

citation: if search_policy was used, the answer must contain a [Doc §x.y] label.

The same functions are used by the verifier node and by the eval metric
"numeric hallucination rate", so the metric measures exactly what the node checks.

Model output is normalised first (see `normalize`). Models routinely format
numbers with narrow no-break spaces ("1 590"), use curly apostrophes, and
bracket citations with CJK lenticular marks. Treating those as different
characters produced false hallucination and missing-citation flags.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Unicode spaces used as digit-group separators, and punctuation variants.
_DIGIT_SPACES = "\u202f\u00a0\u2009\u2007\u2008\u2005\u2006\u2002\u2003\u2004\u2000\u2001"
_DIGIT_SPACE_RE = re.compile(rf"(?<=\d)[{_DIGIT_SPACES}](?=\d{{3}}\b)")
_PUNCT_MAP = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
                            "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
                            **{c: " " for c in _DIGIT_SPACES}})
# Bracket styles models use for citations: [..] 【..】 〔..〕 ⟦..⟧
_OPEN_BRACKETS, _CLOSE_BRACKETS = "\\[\u3010\u3014\u27e6", "\\]\u3011\u3015\u27e7"


def normalize(text: str) -> str:
    """Canonicalise model formatting so matching is about content, not typography."""
    if not text:
        return text
    return _DIGIT_SPACE_RE.sub("", text).translate(_PUNCT_MAP)


NUMBER_RE = re.compile(
    r"(?<![\w.§])[-+]?£?\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s?(%|k\b|m\b|million\b|thousand\b|bn\b)?",
    re.IGNORECASE,
)
CITATION_RE = re.compile(rf"[{_OPEN_BRACKETS}][^{_OPEN_BRACKETS}{_CLOSE_BRACKETS}]{{3,80}}?"
                         rf"§\s?\d+(?:\.\d+)*[{_CLOSE_BRACKETS}]")
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b")
SECTION_RE = re.compile(r"§\s?\d+(?:\.\d+)*")
STOCK_CODE_RE = re.compile(r"\b\d{5}[A-Z]{0,2}\b")
MULTIPLIERS = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9}


@dataclass(frozen=True)
class Num:
    raw: str
    value: float
    decimals: int
    multiplier: float
    is_percent: bool


def extract_numbers(text: str) -> list[Num]:
    text = DATE_RE.sub(" ", normalize(text))
    text = CITATION_RE.sub(" ", text)
    text = SECTION_RE.sub(" ", text)
    out = []
    for m in NUMBER_RE.finditer(text):
        whole, frac, suffix = m.group(1), m.group(2) or "", (m.group(3) or "").lower()
        value = float(whole.replace(",", "") + frac)
        out.append(Num(m.group(0).strip(), value, max(len(frac) - 1, 0),
                       MULTIPLIERS.get(suffix, 1.0), suffix == "%"))
    return out


# Comma-grouped first, so "1,234" reads as one number. Stripping commas globally would
# instead merge JSON array neighbours: [[1590,2184]] -> 15902184.
SOURCE_NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _source_values(sources: list[str]) -> list[float]:
    vals: list[float] = []
    for s in sources:
        for m in SOURCE_NUMBER_RE.finditer(normalize(s)):
            try:
                vals.append(float(m.group(0).replace(",", "")))
            except ValueError:
                pass
    return vals


def _exempt(n: Num) -> bool:
    if n.multiplier == 1 and not n.is_percent and n.decimals == 0:
        if 0 <= n.value <= 10:
            return True
        if 1990 <= n.value <= 2030:
            return True
    return False


def is_grounded(n: Num, source_values: list[float]) -> bool:
    target = n.value * n.multiplier
    tol = 0.5 * (10 ** -n.decimals) * n.multiplier + 1e-9
    candidates = [target]
    if n.is_percent:
        candidates.append(target / 100)  # tool gave a fraction (0.253 -> 25.3%)
    for t in candidates:
        t_tol = tol if t == target else tol / 100
        for v in source_values:
            av = abs(v)
            if abs(av - abs(t)) <= t_tol or (av and abs(av - abs(t)) / av <= 0.005):
                return True
    return False


def ungrounded_numbers(answer: str, tool_outputs: list[str], question: str = "") -> tuple[list[str], int]:
    """Return (ungrounded raw strings, number of checked numbers)."""
    values = _source_values(tool_outputs + [question])
    checked, bad = 0, []
    for n in extract_numbers(answer):
        if _exempt(n):
            continue
        checked += 1
        if not is_grounded(n, values):
            bad.append(n.raw)
    return bad, checked


def has_citation(answer: str) -> bool:
    return bool(CITATION_RE.search(normalize(answer)))


def tool_output_texts(trace: list[dict]) -> list[str]:
    return [json.dumps(e.get("output"), default=str) for e in trace if e.get("type") == "tool_call"
            and not e.get("is_error")]
