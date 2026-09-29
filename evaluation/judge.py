"""LLM-as-judge for policy-answer groundedness, calibrated against manual labels.

An uncalibrated judge score is never reported. The workflow is:

  1. python -m evaluation.judge export  --run-dir evaluation/results/<agent run>
       -> writes judge_labels.csv (answer + retrieved passages, empty `human_label` column)
  2. Fill `human_label` by hand (grounded / ungrounded) for at least 20 rows.
  3. python -m evaluation.judge calibrate --labels evaluation/results/judge_labels.csv
       -> runs the judge on the same rows, reports agreement and Cohen's kappa.
  4. Only if agreement is acceptable, `python -m evaluation.judge score --run-dir ...`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from agent.llm import make_llm
from app.core.config import get_settings

# A stronger model than the agent's default when on Anthropic; otherwise the configured free model.
JUDGE_MODEL = "claude-sonnet-5" if get_settings().llm_provider == "anthropic" else None
JUDGE_PROMPT = """You are grading whether an answer about store policy is GROUNDED in the passages.
An answer is grounded if every policy claim it makes is supported by the passages, and it does not add
policy rules that are not in them. Arithmetic on dates or order values is fine if it follows the passages.

Passages:
{passages}

Answer:
{answer}

Reply with exactly one word: grounded or ungrounded."""


def _policy_rows(run_dir: Path) -> list[dict]:
    rows = []
    for line in (run_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["score"]["category"] != "policy":
            continue
        passages = []
        for e in rec["result"]["trace"]:
            if e.get("type") == "tool_call" and e.get("name") == "search_policy" and not e.get("is_error"):
                passages += [f"{r['citation']} {r['text']}" for r in e["output"]["results"]]
        rows.append({"task_id": rec["score"]["task_id"], "run": rec["run"], "run_id": rec["result"]["run_id"],
                     "answer": rec["result"]["answer"], "passages": "\n\n".join(passages)})
    return rows


def judge(answer: str, passages: str) -> str:
    llm = make_llm(JUDGE_MODEL)
    resp = llm.chat("You are a strict, careful grader.", [{"role": "user", "content": JUDGE_PROMPT.format(
        passages=passages or "(none)", answer=answer)}], max_tokens=16)
    return "grounded" if resp.text.strip().lower().startswith("grounded") else "ungrounded"


def cohens_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["export", "calibrate", "score"])
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--labels", type=Path, default=Path("evaluation/results/judge_labels.csv"))
    args = p.parse_args()

    if args.action == "export":
        rows = _policy_rows(args.run_dir)
        with args.labels.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["task_id", "run", "run_id", "human_label", "answer", "passages"])
            w.writeheader()
            for r in rows:
                w.writerow({**r, "human_label": ""})
        print(f"wrote {len(rows)} rows to {args.labels}; fill human_label with grounded/ungrounded")
    elif args.action == "calibrate":
        rows = [r for r in csv.DictReader(args.labels.open(encoding="utf-8")) if r["human_label"].strip()]
        if len(rows) < 20:
            raise SystemExit(f"need at least 20 labelled rows, have {len(rows)}")
        human = [r["human_label"].strip().lower() for r in rows]
        model = [judge(r["answer"], r["passages"]) for r in rows]
        agree = sum(h == m for h, m in zip(human, model)) / len(rows)
        out = {"n": len(rows), "agreement": agree, "cohens_kappa": cohens_kappa(human, model), "judge_model": JUDGE_MODEL or get_settings().default_model}
        Path("evaluation/results/judge_calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
    else:
        cal = Path("evaluation/results/judge_calibration.json")
        if not cal.exists():
            raise SystemExit("calibrate the judge first; uncalibrated scores are not reported")
        rows = _policy_rows(args.run_dir)
        labels = [judge(r["answer"], r["passages"]) for r in rows]
        rate = labels.count("grounded") / len(labels)
        cal_d = json.loads(cal.read_text(encoding="utf-8"))
        print(f"groundedness (judge): {rate:.1%} over {len(rows)} policy answers "
              f"(judge agreement with manual labels {cal_d['agreement']:.0%}, kappa {cal_d['cohens_kappa']:.2f})")


if __name__ == "__main__":
    main()
