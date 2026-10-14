"""Trusted numerical retries never grant a new candidate submission."""

import hashlib
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.harness.feedback import FeedbackServer
from mriseqbench.harness.grading import EvaluationIncomplete, grade_frozen
from mriseqbench.harness.runner import run_one
from mriseqbench.models import FeedbackConfig
from mriseqbench.process import execute

ROOT = Path(__file__).resolve().parents[2]


def s1_case():
    cat = Catalog(ROOT)
    return cat.resolve(cat.get("tasks", "gre_t1w"), cat.get("physics", "static_b0"))


@pytest.mark.parametrize("verdict", ["passed", "failed"])
@pytest.mark.parametrize(
    "interruption", ["error", "timeout", "numerically_unresolved", "evaluator_error"]
)
def test_worker_recovers_same_frozen_submission_without_extra_attempt(
    monkeypatch, tmp_path, verdict, interruption
):
    calls = []

    def worker(command, directory, budget, *args, **kwargs):
        snapshot = Path(command[command.index("--submission") + 1])
        calls.append((snapshot, snapshot.read_bytes(), budget))
        status = interruption if len(calls) == 1 else verdict
        if status not in {"error", "timeout"}:
            (directory / "result.json").write_text(
                json.dumps(
                    {
                        "status": status,
                        "checks": [],
                        "metrics": {},
                    }
                )
            )
        return {"status": status if status in {"error", "timeout"} else "completed"}

    monkeypatch.setattr("mriseqbench.harness.feedback.execute", worker)
    with FeedbackServer(
        s1_case(), FeedbackConfig(), ROOT, tmp_path / "control"
    ) as server:
        reply = server.handle("submit", server.token, b"original candidate")[1]
        assert reply["status"] == (
            "retry_required" if verdict == "failed" else "submitted"
        )
        assert len(calls) == 2
        assert calls[0] == calls[1]
        assert calls[0][2] is None
        assert server.stats()["submissions_used"] == 1
        attempt = server.attempts[0]
        assert attempt["sha256"] == hashlib.sha256(b"original candidate").hexdigest()
        result = attempt["evaluation"]
        assert result["status"] == verdict
        assert result["worker_recoveries"] == 1
        assert (
            json.loads((Path(attempt["directory"]) / "result.json").read_text())
            == result
        )


@pytest.mark.parametrize(
    "failure", ["error", "timeout", "evaluator_error", "numerically_unresolved"]
)
def test_interrupted_job_has_no_scored_result(tmp_path, failure):
    calls = []

    def worker(command, directory, *args, **kwargs):
        calls.append(command)
        (directory / "result.json").write_text(json.dumps({"status": failure}))
        return {
            "status": "completed" if failure not in {"timeout", "error"} else failure
        }

    with pytest.raises(EvaluationIncomplete, match="resume frozen submission"):
        grade_frozen(
            tmp_path / "sequence.seq", s1_case(), ROOT, tmp_path / "grade", 1, worker
        )
    assert len(calls) == 3
    assert not (tmp_path / "grade/result.json").exists()
    assert len(list((tmp_path / "grade").glob("worker-execution-*.json"))) == 3


def test_cancelled_job_does_not_manufacture_verdict(tmp_path):
    stop = threading.Event()
    stop.set()
    with pytest.raises(EvaluationIncomplete):
        grade_frozen(
            tmp_path / "sequence.seq",
            s1_case(),
            ROOT,
            tmp_path,
            1,
            lambda *args, **kwargs: pytest.fail("cancelled worker launched"),
            stop_when=stop,
        )
    assert not (tmp_path / "result.json").exists()


def test_run_interruption_preserves_submission_and_has_no_result(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "mriseqbench.harness.feedback.execute", lambda *a, **k: {"status": "error"}
    )
    catalog = Catalog(ROOT)
    experiment = catalog.get("experiments", "smoke").model_copy(deep=True)
    experiment.submission_mode = "submit"
    experiment.agent.command = [
        sys.executable,
        "-c",
        """
import pathlib, subprocess
pathlib.Path('sequence.seq').write_bytes(b'candidate')
subprocess.run(['./bin/submit'], check=True)
""",
    ]
    with pytest.raises(EvaluationIncomplete):
        run_one(catalog, experiment, s1_case(), tmp_path / "run", 1)
    attempt = tmp_path / "run/control/submissions/attempt-001"
    assert (attempt / "sequence.seq").read_bytes() == b"candidate"
    assert (attempt / "receipt.json").is_file()
    assert not (attempt / "result.json").exists()
    assert not (tmp_path / "run/evaluation/result.json").exists()


def test_accepted_grading_finishes_after_agent_exits(monkeypatch, tmp_path):
    started = threading.Event()
    requests = []
    threads = []

    def worker(command, directory, *args, **kwargs):
        started.set()
        time.sleep(0.2)
        assert not kwargs["stop_when"].is_set()
        (directory / "result.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "checks": [],
                    "metrics": {},
                }
            )
        )
        return {"status": "completed"}

    def agent(command, directory, *args, **kwargs):
        config = json.loads((directory / ".harness.json").read_text())

        def submit():
            req = urllib.request.Request(
                f"http://127.0.0.1:{config['port']}/submit",
                data=b"accepted candidate",
                headers={"X-MRISeqBench-Token": config["token"]},
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                requests.append(json.loads(response.read()))

        request = threading.Thread(target=submit)
        threads.append(request)
        request.start()
        assert started.wait(5)
        return {"status": "completed"}

    monkeypatch.setattr("mriseqbench.harness.feedback.execute", worker)
    monkeypatch.setattr("mriseqbench.harness.runner.execute", agent)
    catalog = Catalog(ROOT)
    experiment = catalog.get("experiments", "smoke").model_copy(deep=True)
    experiment.submission_mode = "submit"
    report = run_one(catalog, experiment, s1_case(), tmp_path / "run", 1)
    for request in threads:
        request.join(5)
    assert report["status"] == "passed"
    assert report["feedback"]["submissions_used"] == 1
    assert requests[0]["status"] == "submitted"


def test_retry_credit_counts_active_retry_and_is_retained_on_exception(
    monkeypatch, tmp_path
):
    now = [10.0]
    monkeypatch.setattr("mriseqbench.harness.feedback.time.monotonic", lambda: now[0])
    server = FeedbackServer(s1_case(), FeedbackConfig(), ROOT, tmp_path)
    try:
        assert server.retry_credit() == 0
        with pytest.raises(RuntimeError), server.numerical_retry():
            now[0] = 17.0
            assert server.retry_credit() == 7.0
            raise RuntimeError("retry worker failed")
        now[0] = 30.0
        assert server.retry_credit() == 7.0
    finally:
        server.httpd.server_close()


def test_process_deadline_applies_trusted_retry_credit(tmp_path):
    result = execute(
        [sys.executable, "-c", "import time; time.sleep(0.25)"],
        tmp_path,
        0.1,
        tmp_path / "credited",
        timeout_credit=lambda: 0.4,
    )
    assert result["status"] == "completed"


@pytest.mark.parametrize("recover_worker", [False, True])
def test_runner_allows_candidate_revision_after_slow_worker_recovery(
    monkeypatch, tmp_path, recover_worker
):
    calls, now = [], [10.0]
    credited = threading.Event()
    clock = SimpleNamespace(monotonic=lambda: now[0], time=time.time)
    monkeypatch.setattr("mriseqbench.harness.feedback.time", clock)
    monkeypatch.setattr("mriseqbench.process.time", clock)

    def agent(*args, timeout_credit, **kwargs):
        def observed_credit():
            credit = timeout_credit()
            if now[0] == 14.0:
                assert credit == 4.0
                credited.set()
            return credit

        return execute(*args, timeout_credit=observed_credit, **kwargs)

    def worker(command, directory, budget, *args, **kwargs):
        snapshot = Path(command[command.index("--submission") + 1])
        calls.append((snapshot, snapshot.read_bytes(), budget))
        if recover_worker and len(calls) == 1:
            return {"status": "error"}
        if snapshot.read_bytes() == b"first candidate":
            # Advance only trusted grading time, independent of process startup.
            # The real deadline loop must observe this active credit before the
            # evaluator releases its failure verdict for candidate revision.
            now[0] = 14.0
            assert credited.wait(5), "deadline loop did not apply active grading credit"
            status = "failed"
        else:
            status = "passed"
        (directory / "result.json").write_text(
            json.dumps({"status": status, "checks": [], "metrics": {}})
        )
        return {"status": "completed"}

    monkeypatch.setattr("mriseqbench.harness.runner.execute", agent)
    monkeypatch.setattr("mriseqbench.harness.feedback.execute", worker)
    catalog = Catalog(ROOT)
    experiment = catalog.get("experiments", "smoke").model_copy(deep=True)
    experiment.submission_mode = "submit"
    experiment.agent.timeout_s = 3
    experiment.feedback.timeout_s = 5
    script = """
import json, pathlib, subprocess
pathlib.Path('sequence.seq').write_bytes(b'first candidate')
reply = subprocess.run(['./bin/submit'], capture_output=True, text=True, check=False)
assert json.loads(reply.stdout)['status'] == 'retry_required', reply.stdout
pathlib.Path('sequence.seq').write_bytes(b'revised candidate')
subprocess.run(['./bin/submit'], check=True)
"""
    experiment.agent.command = [sys.executable, "-c", script]
    report = run_one(catalog, experiment, s1_case(), tmp_path / "run", 1)
    assert report["status"] == "passed"
    assert report["feedback"]["submissions_used"] == 2
    assert report["feedback"]["grading_wall_s"] == 4
    assert report["feedback"]["numerical_retries"] == 0
    if recover_worker:
        assert calls[0][:2] == calls[1][:2]
    assert [call[2] for call in calls] == [None] * (2 + recover_worker)
    assert calls[-1][1] == b"revised candidate"
    assert report["first_submission_status"] == "failed"
    assert report["passed_on_submission"] == 2
