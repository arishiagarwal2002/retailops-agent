# Architecture decisions

Short ADR-style records: the decision, why, and what would change it.

## ADR-1: A single agent, not multi-agent

**Decision.** One LangGraph agent with a bounded native tool-calling loop (step limit 8).

**Why.** Multi-agent designs pay off when sub-tasks need different contexts, permission scopes or models, or can run in parallel at scale. None of that applies here. All tools share one domain, one database and one permission model, and tasks take 2-4 tool calls. Splitting into a "SQL specialist" and a "policy specialist" would add a routing LLM call, duplicate the schema context and raise the cost of every task, with no expected accuracy gain.

**Would change if.** Tool count grew past ~20 with disjoint contexts, or a sub-task needed a different permission scope (e.g. a write-capable purchasing agent).

## ADR-2: No separate planner

**Decision.** No plan-then-execute planner LLM. The model plans implicitly through tool calls; the graph adds self-correction and verification around it.

**Why.** For 2-4 step tasks a planner adds a full LLM round-trip and more failure surface. The failure modes that matter here (wrong date window, cancellations treated as returns, missing citations) are fixed by schema descriptions, the system prompt and the verifier, not by planning.

## ADR-3: LangGraph, used for three specific features

**Decision.** LangGraph for explicit typed state, `interrupt` for human approval, and checkpointing to resume a paused run. `agent/raw_loop.py` keeps the framework-free loop as a reference.

**Would drop it for** plain Python if the approval flow moved entirely into the API (it already applies state changes itself) and checkpointing weren't needed.

## ADR-4: SQL safety is defence in depth

**Decision.** Three independent layers:
1. **AST guard** (sqlglot): exactly one statement, read-only query types only, write nodes rejected anywhere in the tree (including CTEs), table allow-list, no `pg_*`/`dblink`/`set_config` functions, `LIMIT 1000` enforced.
2. **Read-only role** with `SELECT` on four tables only. It can't see `runs`, `api_keys` or `pending_actions`.
3. **Session settings**: `default_transaction_read_only = on` and a 5 s `statement_timeout`, set on the role and again on every connection.

**Why.** Any single layer can fail: a parser bug or dialect gap, a mis-granted role, an expensive-but-legal query. The integration tests bypass layer 1 and prove layers 2 and 3 still hold.

## ADR-5: pgvector inside the same Postgres, not a managed vector DB

**Decision.** Policy chunks, embeddings, retail data, run logs and caches all live in one Postgres.

**Why.** 65 chunks don't justify another service, another bill or another failure mode. One database means one backup, one connection string and transactional consistency. An IVFFlat index is built after ingest, and `ivfflat.probes` is set high enough that the search is exact at this size. Honestly, a sequential scan is as fast here; the index is there to show the production pattern.

## ADR-6: fastembed (ONNX) instead of sentence-transformers at serve time

**Decision.** Same BAAI/bge weights and an ms-marco MiniLM cross-encoder, run through ONNX Runtime via fastembed.

**Why.** PyTorch alone would push the API image past a free-tier 512 MB instance. fastembed runs the same models without torch. `bge-small-en-v1.5` (384-d) is the default; switching to `bge-base` (768-d) is a config change plus a migration of the vector column.

## ADR-7: Human approval for every state change

**Decision.** `draft_reorder` only writes a `pending` row. `POST /v1/approvals/{id}` is the only code path that creates a purchase order, and it takes a row lock so two approvers can't both place it. Approvals record who approved and when.

**Why.** It's the pattern companies actually deploy for agents that act. It also makes safety measurable: the benchmark counts any unapproved state change as a failure.

## ADR-8: A deterministic verifier, not an LLM judge, in the loop

**Decision.** Before returning, every number in the answer must match a number in some tool output (with rounding, k/m suffixes and percentage/fraction equivalence), and policy answers must carry a `[Doc §x.y]` citation. On failure the model regenerates once, then the answer ships with an explicit warning.

**Why.** It's cheap, deterministic and explainable. An LLM judge in the loop would double cost and add its own errors. The LLM judge is used offline only, and only after calibration against manual labels.

**Known limits.** Integers 0-10 and years are exempt (ordinals and list sizes would dominate false positives). Numbers the model derives correctly (e.g. a percentage it computed itself) are flagged, which is intended: the prompt tells it to compute in SQL.

## ADR-9: No Redis, Kafka, Kubernetes, MLflow or Airflow

**Decision.** Postgres covers caching (response and embedding caches with TTL), state and the audit log. Docker Compose covers local orchestration.

**Why.** There is no stream, no cluster and no training pipeline that runs more than on demand. Each of those tools would be a line on a diagram with no measured benefit.

## ADR-10: Direct multi-horizon LightGBM quantile forecaster

**Decision.** One global LightGBM model per quantile (0.1/0.5/0.9) across the top 500 SKUs. Each training row is (SKU, origin, horizon) with features known at the origin. Targets are scaled by each SKU's mean weekly demand.

**Why.** The direct formulation gives an honest interval per horizon without compounding recursive errors. Quantile objectives give intervals natively. LightGBM trains in seconds and is strong on tabular retail demand. An LSTM would add days and no new evidence. The model is validated with an 8-fold rolling-origin backtest against seasonal naive and a 4-week moving average, and interval coverage is reported.
