"""Infrastructure failures interrupt trials; they are never benchmark outcomes."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from mriseqbench.harness import arm_runtime, feedback
from mriseqbench.harness.grading import EvaluationIncomplete


def test_any_trusted_grading_failure_latches_an_interruption(tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("disk full")

    server = lint_server(tmp_path, monkeypatch, {"status": "completed"})
    monkeypatch.setattr(feedback, "grade_frozen", broken)
    with pytest.raises(EvaluationIncomplete):
        server._submit(b"bytes")
    assert isinstance(server.evaluation_failure, EvaluationIncomplete)
    assert server.submitted.is_set() and not server.attempts
    kept = tmp_path / "control/submissions/attempt-001/sequence.seq"
    assert kept.read_bytes() == b"bytes"  # Frozen for recovery.


@pytest.mark.parametrize(
    ("returncode", "info", "running", "expected"),
    [
        (125, "ok", "true", "exit 125"),
        (1, None, "true", "daemon unavailable"),
        (1, "ok", "false", "stopped"),
        (1, "ok", None, "container lost"),
        (1, "ok", "true", None),
    ],
)
def test_docker_failures_are_infrastructure(
    monkeypatch, returncode, info, running, expected
):
    def docker_output(args, timeout=120):
        value = info if args[0] == "info" else running
        if value is None:
            raise ValueError("Cannot connect to the Docker daemon")
        # infrastructure_failure inspects the retained container's JSON state.
        return value if args[0] == "info" else f'{{"Running": {value}}}'

    monkeypatch.setattr(arm_runtime, "docker_output", docker_output)
    reason = arm_runtime.infrastructure_failure(
        {"container": "x", "persistent_container": True},
        {"status": "error", "returncode": returncode},
    )
    assert (reason is None) if expected is None else (expected in reason)


def test_smoke_pass_is_a_binary_outcome():
    from mriseqbench.evaluation import normalize_verdict

    case = {"case_id": "c", "evaluation": {"id": "pulseq_preflight"}}
    assert normalize_verdict({"status": "smoke_passed"}, case) == {
        "status": "passed",
        "scope": "smoke",
    }
    assert (
        "smoke_passed" not in Path(feedback.__file__).with_name("client.py").read_text()
    )


def lint_server(tmp_path, monkeypatch, execution, result=None):
    def run(_command, directory, *_args, **_kwargs):
        if result is not None:
            (directory / "result.json").write_text(__import__("json").dumps(result))
        return execution

    monkeypatch.setattr(feedback, "execute", run)
    config = SimpleNamespace(timeout_s=5, max_submission_bytes=1024)
    case = {"case_id": "c"}
    return feedback.FeedbackServer(case, config, tmp_path, tmp_path / "control")


def test_lint_worker_failures_latch_but_input_failures_do_not(tmp_path, monkeypatch):
    failed = {"status": "failed", "checks": [], "metrics": {}}
    server = lint_server(tmp_path / "a", monkeypatch, {"status": "completed"}, failed)
    assert server._handle("check", b"seq") == (200, failed)
    assert server.evaluation_failure is None and not server.submitted.is_set()
    server = lint_server(tmp_path / "b", monkeypatch, {"status": "timeout"})
    code, value = server._handle("lint", b"seq")
    assert code == 200 and value["status"] == "failed"
    assert server.evaluation_failure is None
    for status in ("launch_error", "error"):
        server = lint_server(tmp_path / status, monkeypatch, {"status": status})
        with pytest.raises(EvaluationIncomplete):
            server._handle("check", b"seq")
        assert server.submitted.is_set()
        assert isinstance(server.evaluation_failure, EvaluationIncomplete)
        assert (tmp_path / status / "control/infrastructure-failure.json").exists()


def test_backoff_shares_the_interval_union(tmp_path, monkeypatch):
    import time

    server = lint_server(tmp_path, monkeypatch, {"status": "completed"})
    with server.excluded("simulator_queue"):
        with server.backoff():
            time.sleep(0.05)
        time.sleep(0.05)
    credit = server.grading_credit()
    assert 0.1 <= credit < 0.15  # Overlap credited once, as waited.
    assert 0.05 <= server.stats()["provider_backoff_wall_s"] < 0.1
