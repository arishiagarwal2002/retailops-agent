# Failure analysis

A running log, written as failures are found rather than reconstructed afterwards. Part A lists real failures found while building and testing the system. Part B is filled from `evaluation/results/*/per_task.csv` after full LLM evaluation runs. Filter rows with `success == False` and group them by cause.

## A. Found during the build (real, with fixes)

| # | Failure | Example | Root cause | Fix |
|---|---|---|---|---|
| 1 | Wrong product categories | "WORLD WAR 2 GLIDERS ASSTD DESIGNS" was categorised **Home Decor** and became the top Home Decor product | Keyword rule `"SIGN" in description` matched inside "DE**SIGN**S" | Whole-word, plural-tolerant regex matching; re-derived the products table (`data/clean.py`) |
| 2 | "Best day of 2011" is a cancelled order | 9 Dec 2011 has £198k gross revenue, almost all from one 80,995-unit order that was cancelled the same day | Gross revenue excludes cancellation rows by definition, so an order-and-cancel pair inflates gross figures | The gold answer keeps the gross definition. The system prompt states gross vs net explicitly, and the agent must say which one it used |
| 3 | Mocked answers served to real users | `/v1/ask` returned "Done." for a real question | The integration test's scripted LLM response was written to the shared response cache under the production model's key | Cache writes only when `use_cache` is on AND the answering model equals the cache key's model |
| 4 | Reranking lowered recall | Recall@5 fell from 80.0% (dense) to 73.3% (dense top-20 → cross-encoder top-5); Hit@1 rose from 73.3% to 80.0% | On a 65-chunk corpus the MS-MARCO cross-encoder favours lexically similar sections (e.g. §2.3 "return postage" for a returns-window question) | Kept reranking for its Hit@1 gain, which matters for citations. Recorded as a measured trade-off instead of assuming the DevDocs result transfers |
| 5 | Retrieval misses invoice-style questions | "The customer on invoice 578680 ... can they return it?" retrieves nothing about return windows | The raw question has no policy vocabulary; the evidence depends on the invoice date | The agent writes its own `search_policy` query after looking up the invoice. In-loop Recall@5 is reported separately from raw-question Recall@5 |
| 6 | Unhelpful error with no LLM key | Runs failed with a bare `TypeError` | The SDK raises `TypeError` when no credential is found | Clear message, plus `/health` reports `llm: not configured` |

## A2. The harness was wrong before the agent was

The first real evaluation run (10-task smoke, `openai/gpt-oss-120b`) scored **40%**. Reading every failing
trace showed that **five of the six failures were scorer bugs, not agent mistakes**. After fixing them the
same agent build scored **90%**, with one genuine failure left. The score moved 50 points without the agent
changing at all.

This is the argument for inspecting failures instead of trusting the number. A harness that is wrong in the
agent's favour inflates the score; this one was wrong against it, which is the direction that makes you throw
away work that was already correct.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 7 | Correct figures flagged as hallucinated | The model formats thousands with U+202F (narrow no-break space): `1 590`. The number parser read `1` and `590` | `normalize()` strips Unicode digit-group spaces before any matching (`agent/verifier.py`) |
| 8 | Real tool values looked invented | `_source_values` stripped **all** commas from tool output, so the JSON array `[[1590,2184]]` became the single number `15902184` | Match comma-grouped numbers first, then strip commas per match |
| 9 | Correct citations reported missing | The model cited as `【Returns Policy §2.1】` (CJK lenticular brackets), not `[...]` | Citation regex accepts `[]` `【】` `〔〕` `⟦⟧`; the prompt also asks for plain square brackets |
| 10 | A correct refusal scored as compliance | The agent wrote "I can't delete data" with a curly apostrophe (U+2019); `DECLINE_RE` only matched the ASCII `'` | Normalise quotes and dashes before matching |
| 11 | A correct answer failed on tool order | "Will stock cover the forecast?" was scored as SQL → forecast, but the two lookups are independent and the agent did forecast → SQL | `unordered_tools: true` for tasks with no data dependency. Order is still scored where a later call's arguments depend on an earlier result (e.g. `multi_013`) |
| 12 | A brittle phrase list | The "should refuse to forecast" task matched a fixed list of phrasings; the agent's wording ("requires a longer sales history than the six weeks available") was correct but unlisted | `mention_regex`: an insufficiency word near *history/data/weeks*, so any honest phrasing passes and a fabricated forecast still fails |

The one **genuine** agent failure in that run: on "what percentage of 2011 revenue came from the UK", it hit
`function round(double precision, integer) does not exist` twice, self-corrected, produced the right query,
and then ran out of its 8-step budget. Root cause: `quantity * unit_price` is double precision and Postgres
`ROUND(x, 2)` needs numeric. The cast rule is now part of the `transactions` schema notes the agent reads,
and the task passes. Self-correction worked; it just wasn't cheap enough to fit the budget.

## B. From evaluation runs (fill after `python -m evaluation.run_eval`)

Expected candidates, to be confirmed or refuted with counts:

| Failure mode | How to detect in per_task.csv | Count | Example run_id | Fix |
|---|---|---:|---|---|
| Date-boundary semantics ("last quarter" inclusive/exclusive) | sql/multi failures whose SQL uses `<=` on an end date | | | |
| Cancellations treated as returns | policy/sql answers mentioning "return" for `C` invoices | | | |
| Over-long plans on one-query questions | `steps` > 4 on sql_single | | | |
| Forecasting a short-history SKU | multi_009 failing with forecast numbers present | | | |
| Missing citation when policy is mixed with a computed date | pol_012-014 with `citation_present == False` | | | |
| Guessing instead of clarifying | ambiguous tasks with status `completed` | | | |
