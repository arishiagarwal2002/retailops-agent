"""retry_errors must re-run infrastructure failures only, never scored failures.

Re-rolling tasks that produced an answer until they happen to pass would make the
benchmark meaningless, so that boundary is enforced here.
"""
import json

import pytest

from evaluation import run_eval


def _record(task_id, status, success, run_id):
    return {"run": 0,
            "score": {"task_id": task_id, "category": "sql_single", "success": success, "status": status,
                      "tool_selection_ok": True, "mentions_ok": success, "numbers_checked": 0,
                      "numbers_ungrounded": 0, "tool_calls": 1, "tool_calls_valid": 1, "write_attempts": 0,
                      "write_attempts_blocked": 0, "latency_ms": 10, "steps": 1, "tokens_in": 1,
                      "tokens_out": 1, "cost_gbp": 0.0, "run": 0},
            "result": {"run_id": run_id, "status": status, "answer": "a", "trace": []}}


@pytest.fixture()
def run_dir(tmp_path):
    d = tmp_path / "agent__m__x"
    d.mkdir()
    records = [
        _record("sql_001", "error", False, "r1"),          # infrastructure failure -> retry
        _record("sql_002", "completed", False, "r2"),       # ran, scored a failure -> must NOT retry
        _record("sql_003", "completed", True, "r3"),        # passed -> must NOT retry
    ]
    (d / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    (d / "summary.json").write_text(json.dumps({
        "config": "agent", "model": "m", "runs": 1, "n_tasks": 3, "smoke": False, "timestamp": "x",
        "metrics": {}, "per_run": []}), encoding="utf-8")
    return d


def test_retries_only_errored_tasks(run_dir, monkeypatch):
    called = []

    def fake_run_one(task, config, model):
        called.append(task.id)
        return {"score": {**_record(task.id, "completed", True, "new")["score"], "success": True,
                          "status": "completed"},
                "result": {"run_id": "new", "status": "completed", "answer": "fixed", "trace": []}}

    monkeypatch.setattr(run_eval, "run_one", fake_run_one)
    monkeypatch.setattr(run_eval, "write_results_md", lambda: None)
    assert run_eval.retry_errors(run_dir) == 0

    # Only the errored task was re-run.
    assert called == ["sql_001"]

    records = [json.loads(line) for line in (run_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[0]["result"]["status"] == "completed"
    assert records[0]["retried_from"] == "r1"
    # The scored failure is untouched.
    assert records[1]["score"]["success"] is False and "retried_from" not in records[1]

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["retried_tasks"] == ["sql_001"]
    assert "never retried" in summary["retry_note"]


def test_no_errors_is_a_noop(run_dir, monkeypatch):
    records = [json.loads(line) for line in (run_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    records[0]["result"]["status"] = "completed"
    (run_dir / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    monkeypatch.setattr(run_eval, "run_one", lambda *a: pytest.fail("must not re-run anything"))
    assert run_eval.retry_errors(run_dir) == 0


def test_still_erroring_tasks_are_recorded(run_dir, monkeypatch):
    def fake_run_one(task, config, model):
        return {"score": {**_record(task.id, "error", False, "new")["score"]},
                "result": {"run_id": "new", "status": "error", "answer": "", "trace": []}}

    monkeypatch.setattr(run_eval, "run_one", fake_run_one)
    monkeypatch.setattr(run_eval, "write_results_md", lambda: None)
    run_eval.retry_errors(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["still_erroring"] == ["sql_001"]
