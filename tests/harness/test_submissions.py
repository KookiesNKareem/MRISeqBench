import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.harness.feedback import FeedbackServer
from mriseqbench.harness.runner import run
from mriseqbench.models import Experiment, FeedbackConfig

ROOT = Path(__file__).resolve().parents[2]


def smoke_case():
    catalog = Catalog(ROOT)
    return catalog.cases(catalog.get("suites", "smoke"))[0]


def test_three_attempts_blind_first_and_failure_feedback(submission, tmp_path):
    # Even a legacy evaluation-feedback config exposes only basic checks.
    config = FeedbackConfig(mode="evaluation", max_checks=99, timeout_s=30)
    with FeedbackServer(smoke_case(), config, ROOT, tmp_path) as server:
        assert server.max_submissions == 3
        for attempt in (1, 2):
            for kind in ("lint", "check"):
                code, result = server.handle(kind, server.token, b"invalid sequence")
                assert code == 200
                assert result["status"] == "failed"  # The file is malformed.
                assert result["metrics"] == {}
                assert all(not c["name"].startswith("sar") for c in result["checks"])
                assert len(server.attempts) == attempt - 1
            code, result = server.handle("submit", server.token, b"invalid sequence")
            assert code == 200
            assert result["status"] == "retry_required"
            assert result["attempt"] == attempt
            assert result["submissions_remaining"] == 3 - attempt
            assert result["evaluation"]["status"] == "failed"
            assert any(not check["passed"] for check in result["evaluation"]["checks"])
            assert not server.submitted.is_set()
        code, result = server.handle("submit", server.token, submission.read_bytes())
        assert code == 200
        assert result["status"] == "submitted"
        assert server.submitted.is_set()
        assert server.handle("submit", server.token, b"fourth")[0] == 403
        assert server.handle("check", server.token, b"fourth")[0] == 403
        assert [a["blind"] for a in server.attempts] == [True, False, False]
        assert [a["evaluation"]["status"] for a in server.attempts] == [
            "failed",
            "failed",
            "smoke_passed",
        ]
        assert server.stats()["checks_used"] == 2
        assert server.stats()["max_checks"] is None
        for attempt in server.attempts:
            directory = Path(attempt["directory"])
            assert (
                hashlib.sha256((directory / "sequence.seq").read_bytes()).hexdigest()
                == attempt["sha256"]
            )
            assert (
                json.loads((directory / "result.json").read_text())
                == attempt["evaluation"]
            )
        assert (
            tmp_path / "submissions/attempt-001/sequence.seq"
        ).read_bytes() == b"invalid sequence"
        assert server.submission_path.read_bytes() == submission.read_bytes()


def test_third_failure_ends_run_without_fourth_attempt(tmp_path):
    with FeedbackServer(
        smoke_case(), FeedbackConfig(timeout_s=30), ROOT, tmp_path
    ) as server:
        statuses = [
            server.handle("submit", server.token, b"invalid")[1]["status"]
            for _ in range(3)
        ]
        assert statuses == ["retry_required", "retry_required", "submitted"]
        assert len(server.attempts) == 3
        assert all(a["evaluation"]["status"] == "failed" for a in server.attempts)
        assert server.handle("submit", server.token, b"fourth")[0] == 403


def test_lint_checks_validity_while_check_adds_hardware_limits(submission, tmp_path):
    case = smoke_case()
    case["hardware"]["max_b1_uT"] = 0.01
    # The new flow enables basic commands even with mode=none and max_checks=0.
    with FeedbackServer(case, FeedbackConfig(timeout_s=30), ROOT, tmp_path) as server:
        code, lint = server.handle("lint", server.token, submission.read_bytes())
        assert code == 200
        assert lint["status"] == "lint_passed"
        assert all(c["name"] != "peak_b1" for c in lint["checks"])
        code, check = server.handle("check", server.token, submission.read_bytes())
        assert code == 200
        assert check["status"] == "failed"
        assert any(c["name"] == "peak_b1" and not c["passed"] for c in check["checks"])
        assert {c["name"] for c in lint["checks"]} <= {
            c["name"] for c in check["checks"]
        }
        assert lint["metrics"] == check["metrics"] == {}
        assert server.attempts == []
        assert server.stats()["submissions_used"] == 0


def test_first_pass_ends_run_and_invalid_transport_does_not_consume_attempt(
    submission, tmp_path
):
    with FeedbackServer(
        smoke_case(), FeedbackConfig(timeout_s=30), ROOT, tmp_path
    ) as server:
        assert server.handle("submit", "wrong-token", submission.read_bytes())[0] == 403
        assert server.handle("submit", server.token, b"")[0] == 400
        assert server.attempts == []
        assert (
            server.handle("submit", server.token, submission.read_bytes())[1]["status"]
            == "submitted"
        )
        assert len(server.attempts) == 1
        assert server.attempts[0]["blind"]


def test_evaluator_error_is_not_reported_as_submission_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "mriseqbench.harness.feedback.execute",
        lambda *args, **kwargs: {"status": "timeout"},
    )
    with FeedbackServer(smoke_case(), FeedbackConfig(), ROOT, tmp_path) as server:
        result = server.handle("submit", server.token, b"candidate")[1]
        assert result["status"] == "submitted"
        assert server.submitted.is_set()
        assert server.attempts[0]["evaluation"]["status"] == "evaluator_error"


@pytest.mark.parametrize("failures", [1, 2])
def test_runner_continues_same_agent_and_records_each_attempt(
    submission, tmp_path, failures
):
    catalog = Catalog(ROOT)
    experiment = copy.deepcopy(catalog.get("experiments", "smoke"))
    experiment.submission_mode = "submit"
    experiment.agent.timeout_s = 45
    experiment.feedback.timeout_s = 30
    script = """import json,pathlib,shutil,subprocess,sys,time
for attempt in range(int(sys.argv[2])):
    pathlib.Path('sequence.seq').write_text('invalid sequence')
    result = subprocess.run(['./bin/submit'],capture_output=True,text=True,check=True)
    reply = json.loads(result.stdout)
    assert reply['status']=='retry_required',reply
    pathlib.Path(f'feedback-{attempt+1}.json').write_text(result.stdout)
shutil.copyfile(sys.argv[1],'sequence.seq')
subprocess.run(['./bin/submit'],check=True)
pathlib.Path('sequence.seq').write_text('changed after submission')
time.sleep(60)
"""
    experiment.agent.command = [
        sys.executable,
        "-c",
        script,
        str(submission),
        str(failures),
    ]
    _, reports = run(catalog, experiment, tmp_path / "run")
    result = reports[0]
    assert result["status"] == "smoke_passed"
    assert result["agent"]["status"] == "submitted"
    assert result["first_submission_status"] == "failed"
    assert result["passed_on_submission"] == failures + 1
    assert len(result["submissions"]) == failures + 1
    workspace = Path(result["workspace"])
    assert (
        workspace / "evaluation/sequence.seq"
    ).read_bytes() == submission.read_bytes()
    assert (workspace / "agent/feedback-1.json").exists()


def test_runner_retains_failed_attempt_if_agent_exits_early(tmp_path):
    catalog = Catalog(ROOT)
    experiment = copy.deepcopy(catalog.get("experiments", "smoke"))
    experiment.submission_mode = "submit"
    experiment.agent.command = [
        sys.executable,
        "-c",
        "from pathlib import Path; import subprocess; Path('sequence.seq').write_text('invalid'); subprocess.run(['./bin/submit'],check=True)",
    ]
    _, reports = run(catalog, experiment, tmp_path / "run")
    assert reports[0]["status"] == "failed"
    assert reports[0]["first_submission_status"] == "failed"
    assert reports[0]["passed_on_submission"] is None
    assert len(reports[0]["submissions"]) == 1


@pytest.mark.parametrize("count", [0, 4])
def test_submission_budget_is_bounded(count):
    document = Catalog(ROOT).get("experiments", "smoke").model_dump()
    document["max_submissions"] = count
    with pytest.raises(ValueError):
        Experiment.model_validate(document)
