"""Run the benchmark.

    python -m evaluation.run_eval --config agent --runs 3
    python -m evaluation.run_eval --config baseline --runs 3
    python -m evaluation.run_eval --config no_selfcorrect --runs 3
    python -m evaluation.run_eval --config no_verifier --runs 3
    python -m evaluation.run_eval --config agent --model claude-opus-5 --runs 3
    python -m evaluation.run_eval --config agent --smoke --gate evaluation/results/gate.json
    python -m evaluation.run_eval --config oracle           # harness self-test, no API key

Writes evaluation/results/<config>__<model>__<timestamp>/{per_task.csv, runs.jsonl, summary.json}
and regenerates evaluation/results/RESULTS.md from every summary.json present.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from agent.baseline import run_baseline
from agent.llm import make_llm
from agent.prompts import PROMPT_VERSION
from agent.runner import run_agent
from agent.state import AgentConfig
from app.core.config import get_settings
from app.core.logging import configure_logging
from evaluation.metrics import CATEGORIES, Task, load_tasks, score

RESULTS = Path(__file__).parent / "results"
SMOKE_IDS = ["sql_001", "sql_004", "sql_011", "sql_021", "multi_001", "multi_009", "pol_001", "pol_013",
             "amb_001", "adv_001"]

CONFIGS = {
    "agent": AgentConfig(name="agent"),
    "no_selfcorrect": AgentConfig(name="no_selfcorrect", self_correct=False),
    "no_verifier": AgentConfig(name="no_verifier", verifier=False),
}


def run_one(task: Task, config: str, model: str) -> dict:
    if config == "baseline":
        result = run_baseline(task.question, make_llm(model))
    elif config == "oracle":
        from evaluation.oracle import oracle_for

        result = run_agent(task.question, oracle_for(task), AgentConfig(name="oracle"))
    else:
        cfg = CONFIGS[config]
        result = run_agent(task.question, make_llm(model), AgentConfig(model=model, name=cfg.name,
                                                                       self_correct=cfg.self_correct,
                                                                       verifier=cfg.verifier))
    s = score(task, result)
    s.update({"run_id": result.run_id, "latency_ms": result.latency_ms, "steps": result.steps,
              "tokens_in": result.tokens_in, "tokens_out": result.tokens_out, "cost_gbp": result.cost_gbp,
              "answer": result.answer, "tools": result.tool_calls})
    return {"score": s, "result": result.to_dict()}


def aggregate(rows: list[dict]) -> dict:
    def rate(items, key):
        vals = [r[key] for r in items if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    checked = sum(r["numbers_checked"] for r in rows)
    calls = sum(r["tool_calls"] for r in rows)
    writes = sum(r["write_attempts"] for r in rows)
    lat = [r["latency_ms"] for r in rows]
    policy = [r for r in rows if r["category"] == "policy"]
    return {
        "n_tasks": len(rows),
        "task_success": rate(rows, "success"),
        "success_by_category": {c: rate([r for r in rows if r["category"] == c], "success")
                                for c in CATEGORIES if any(r["category"] == c for r in rows)},
        "tool_selection_accuracy": rate(rows, "tool_selection_ok"),
        "argument_validity_rate": (sum(r["tool_calls_valid"] for r in rows) / calls) if calls else None,
        "numeric_hallucination_rate": (sum(r["numbers_ungrounded"] for r in rows) / checked) if checked else None,
        "execution_match_sql": rate([r for r in rows if r["category"] == "sql_single"], "execution_match"),
        "policy_recall_at_5": rate(policy, "recall_at_5"),
        "policy_citation_rate": rate(policy, "citation_present"),
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "steps_mean": float(np.mean([r["steps"] for r in rows])),
        "cost_per_task_gbp": float(np.mean([r["cost_gbp"] for r in rows])),
        "cost_by_category_gbp": {c: float(np.mean([r["cost_gbp"] for r in rows if r["category"] == c]))
                                 for c in CATEGORIES if any(r["category"] == c for r in rows)},
        "tokens_in_mean": float(np.mean([r["tokens_in"] for r in rows])),
        "tokens_out_mean": float(np.mean([r["tokens_out"] for r in rows])),
        "write_attempts": writes,
        "write_attempts_blocked_rate": (sum(r["write_attempts_blocked"] for r in rows) / writes) if writes else None,
    }


def mean_std(values: list) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None}
    return {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def combine(per_run: list[dict]) -> dict:
    out: dict = {}
    for key, v in per_run[0].items():
        if isinstance(v, dict):
            out[key] = {k: mean_std([r[key].get(k) for r in per_run]) for k in v}
        else:
            out[key] = mean_std([r[key] for r in per_run])
    return out


CSV_FIELDS = ["run", "task_id", "category", "success", "status", "tool_selection_ok", "mentions_ok",
              "execution_match", "verdict", "citation_present", "recall_at_5", "numbers_checked",
              "numbers_ungrounded", "ungrounded_examples", "tool_calls", "tool_calls_valid", "write_attempts",
              "write_attempts_blocked", "forbidden_tool_called", "latency_ms", "steps", "tokens_in", "tokens_out",
              "cost_gbp", "tools", "citations", "run_id", "answer"]

# A run in one of these states never reached a scored answer: the provider was unreachable or
# rate-limited us out. Those are harness/infrastructure failures, not agent mistakes.
INFRASTRUCTURE_STATUSES = {"error"}


def write_outputs(out_dir: Path, all_rows: list[dict], per_run: list[dict], **meta) -> dict:
    """Write per_task.csv and summary.json, then refresh RESULTS.md."""
    with (out_dir / "per_task.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    summary = {**meta, "prompt_version": PROMPT_VERSION, "metrics": combine(per_run), "per_run": per_run}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_results_md()
    return summary


def retry_errors(run_dir: Path, concurrency: int = 1) -> int:
    """Re-run only the tasks whose run ERRORED, and rewrite that run's outputs in place.

    Errored runs never produced a scored answer (the provider was unreachable or rate-limited),
    so leaving them in reports an infrastructure fault as an agent failure. Tasks that *did*
    produce an answer are never retried, whatever they scored - re-rolling scored failures until
    they pass would make the benchmark meaningless. The rewritten summary records what was retried.
    """
    records = [json.loads(line) for line in (run_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not records:
        print(f"{run_dir}: no runs recorded")
        return 1
    old_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config, model = old_summary["config"], old_summary["model"]

    targets = [(i, r) for i, r in enumerate(records) if r["result"]["status"] in INFRASTRUCTURE_STATUSES]
    if not targets:
        print(f"{run_dir}: no errored runs to retry")
        return 0
    by_id = {t.id: t for t in load_tasks()}
    print(f"retrying {len(targets)} errored task(s): {', '.join(r['score']['task_id'] for _, r in targets)}")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        redone = list(pool.map(lambda ir: run_one(by_id[ir[1]["score"]["task_id"]], config, model), targets))

    retried, still_failing = [], []
    for (idx, old), fresh in zip(targets, redone):
        task_id = old["score"]["task_id"]
        if fresh["result"]["status"] in INFRASTRUCTURE_STATUSES:
            still_failing.append(task_id)
        fresh["score"]["run"] = old["run"]
        records[idx] = {"run": old["run"], "retried_from": old["result"]["run_id"], **fresh}
        retried.append(task_id)
        print(f"  {task_id}: {old['result']['status']} -> {fresh['result']['status']} "
              f"(success={fresh['score']['success']})")

    with (run_dir / "runs.jsonl").open("w", encoding="utf-8") as jf:
        for r in records:
            jf.write(json.dumps(r, default=str) + "\n")

    per_run = [aggregate([r["score"] for r in records if r["run"] == i])
               for i in sorted({r["run"] for r in records})]
    meta = {k: old_summary[k] for k in ("config", "model", "runs", "n_tasks", "smoke", "timestamp")}
    meta["retried_tasks"] = sorted(set(old_summary.get("retried_tasks", []) + retried))
    meta["retry_note"] = ("Tasks listed in retried_tasks errored before producing an answer (provider "
                          "unreachable or rate-limited) and were re-run. Scored failures are never retried.")
    if still_failing:
        meta["still_erroring"] = still_failing
    summary = write_outputs(run_dir, [r["score"] for r in records], per_run, **meta)
    print(f"task success now {summary['metrics']['task_success']['mean']:.1%}"
          + (f" | still erroring: {', '.join(still_failing)}" if still_failing else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="agent", choices=[*CONFIGS, "baseline", "oracle"])
    p.add_argument("--model", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--category", action="append", choices=CATEGORIES)
    p.add_argument("--smoke", action="store_true", help="10-task subset used in CI")
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=None,
                   help="parallel tasks; defaults to 4 on Anthropic, 1 on free tiers (rate limits)")
    p.add_argument("--gate", type=Path, help="JSON with min task_success; exit 1 if below minus 3 points")
    p.add_argument("--retry-errors", type=Path, metavar="RUN_DIR",
                   help="re-run only the ERRORED tasks of an existing run and rewrite its outputs")
    p.add_argument("--resume", type=Path, metavar="RUN_DIR",
                   help="continue an unfinished run, skipping tasks it already completed "
                        "(for daily-quota-limited free tiers)")
    args = p.parse_args(argv)
    configure_logging("WARNING")

    if args.retry_errors:
        c = args.concurrency or (4 if get_settings().llm_provider == "anthropic" else 1)
        return retry_errors(args.retry_errors, concurrency=c)

    model = "scripted" if args.config == "oracle" else (args.model or get_settings().default_model)
    if args.concurrency is None:
        args.concurrency = 4 if get_settings().llm_provider == "anthropic" or args.config == "oracle" else 1
    tasks = load_tasks(args.category or CATEGORIES)
    if args.smoke:
        tasks = [t for t in tasks if t.id in SMOKE_IDS]
    if args.limit:
        tasks = tasks[: args.limit]

    # Resume: keep every task already finished, redo only what is missing or errored.
    done: list[dict] = []
    if args.resume:
        out_dir = args.resume
        jsonl = out_dir / "runs.jsonl"
        if jsonl.exists():
            done = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line]
        done = [r for r in done if r["result"]["status"] not in INFRASTRUCTURE_STATUSES]
        finished = {(r["run"], r["score"]["task_id"]) for r in done}
        stamp = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))["timestamp"] \
            if (out_dir / "summary.json").exists() else dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"resuming {out_dir}: {len(done)} task-runs already complete")
    else:
        finished = set()
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        # Model ids may contain "/" (e.g. openai/gpt-oss-120b), which would nest the results directory.
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
        out_dir = RESULTS / f"{args.config}__{slug}__{stamp}{'__smoke' if args.smoke else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run, all_rows = [], []
    mode = "a" if args.resume else "w"
    with (out_dir / "runs.jsonl").open(mode, encoding="utf-8") as jf:
        for run_idx in range(args.runs):
            todo = [t for t in tasks if (run_idx, t.id) not in finished]
            if args.resume and not todo:
                print(f"run {run_idx + 1}/{args.runs}: already complete")
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                outputs = list(pool.map(lambda t: run_one(t, args.config, model), todo))
            rows = [r["score"] for r in done if r["run"] == run_idx]
            for o in outputs:
                o["score"]["run"] = run_idx
                rows.append(o["score"])
                jf.write(json.dumps({"run": run_idx, **o}, default=str) + "\n")
                jf.flush()  # so progress survives an interrupted run
            agg = aggregate(rows)
            per_run.append(agg)
            all_rows.extend(rows)
            done_note = "" if len(rows) == len(tasks) else f" [{len(rows)}/{len(tasks)} tasks so far]"
            print(f"run {run_idx + 1}/{args.runs}: task success {agg['task_success']:.1%}{done_note} "
                  + " ".join(f"{c}={v:.0%}" for c, v in agg["success_by_category"].items()))

    summary = write_outputs(out_dir, all_rows, per_run, config=args.config, model=model, runs=args.runs,
                            n_tasks=len(tasks), smoke=args.smoke, stamp=stamp,
                            complete=all(len([r for r in all_rows if r["run"] == i]) == len(tasks)
                                         for i in range(args.runs)))
    print(f"results: {out_dir}")
    if not summary["complete"]:
        print(f"INCOMPLETE - continue tomorrow with:\n"
              f"  python -m evaluation.run_eval --config {args.config} --resume {out_dir}")

    if args.gate:
        gate = json.loads(args.gate.read_text(encoding="utf-8"))
        got = summary["metrics"]["task_success"]["mean"]
        floor = gate["task_success"] - 0.03
        if got < floor:
            print(f"GATE FAILED: task success {got:.1%} < {floor:.1%} (baseline {gate['task_success']:.1%} - 3 pts)")
            return 1
        print(f"gate passed: {got:.1%} >= {floor:.1%}")
    return 0


def _fmt(m: dict | None, pct: bool = True, digits: int = 1) -> str:
    if not m or m.get("mean") is None:
        return "-"
    if pct:
        return f"{m['mean']:.{digits}%} ± {m['std'] * 100:.{digits}f}"
    return f"{m['mean']:.{digits}f} ± {m['std']:.{digits}f}"


def write_results_md() -> None:
    summaries = []
    for f in sorted(RESULTS.glob("*/summary.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        if s.get("smoke") or s["config"] == "oracle":
            continue
        summaries.append(s)
    # Keep the latest summary per (config, model).
    latest: dict[tuple, dict] = {}
    for s in summaries:
        latest[(s["config"], s["model"])] = s
    lines = ["# Evaluation results", "",
             "Generated by `python -m evaluation.run_eval`. Mean ± std over runs. Benchmark: 80 tasks "
             "(30 SQL, 15 multi-step, 15 policy, 10 ambiguous, 10 adversarial).", ""]
    if not latest:
        lines.append("_No full runs recorded yet._")
    else:
        cats = list(CATEGORIES)
        lines += ["| Config | Model | Runs | Tasks | Task success | " + " | ".join(cats) +
                  " | Tool sel. | Arg valid | Num. halluc. | Recall@5 | p50 s | p95 s | £/task |",
                  "|" + "---|" * (14 + len(cats))]
        for (config, model), s in sorted(latest.items()):
            m = s["metrics"]
            lines.append(
                f"| {config} | {model} | {s['runs']} | {s['n_tasks']} | {_fmt(m['task_success'])} | "
                + " | ".join(_fmt(m["success_by_category"].get(c), digits=0) for c in cats)
                + f" | {_fmt(m['tool_selection_accuracy'])} | {_fmt(m['argument_validity_rate'])} | "
                f"{_fmt(m['numeric_hallucination_rate'])} | {_fmt(m['policy_recall_at_5'])} | "
                f"{m['latency_p50_ms']['mean'] / 1000:.1f} | {m['latency_p95_ms']['mean'] / 1000:.1f} | "
                f"{m['cost_per_task_gbp']['mean']:.4f} |")
        # Disclose retries in the published table, not only in summary.json.
        notes = [(c, mo, s) for (c, mo), s in sorted(latest.items()) if s.get("retried_tasks")]
        if notes:
            lines += ["", "**Retried tasks.** These runs errored before producing an answer (provider "
                      "unreachable or rate-limited) and were re-run; tasks that produced an answer were "
                      "never retried, whatever they scored."]
            for config, model, s in notes:
                line = f"- `{config}` / `{model}`: {', '.join(s['retried_tasks'])}"
                if s.get("still_erroring"):
                    line += f" (still erroring after retry: {', '.join(s['still_erroring'])})"
                lines.append(line)
    (RESULTS / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
