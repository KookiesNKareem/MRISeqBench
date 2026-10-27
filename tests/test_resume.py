import json
from pathlib import Path

import pytest

from mriseqbench.catalog import Catalog, load_yaml
from mriseqbench.harness.grading import EvaluationIncomplete, grade_frozen
from mriseqbench.harness.runner import run
from mriseqbench.models import Experiment

ROOT = Path(__file__).resolve().parents[1]
PAID = {"route": "/chat/completions", "request_id": "x", "status": 200}


def grade(tmp_path, worker, evaluator="rf_excitation", gate="preflight_passed"):
    """Grade with a fake evaluation worker behind a fake preflight gate."""

    def run(command, directory, timeout, log, **kwargs):
        if command[command.index("--mode") + 1] != "preflight":
            return worker(command, directory, timeout, log, **kwargs)
        (directory / "result.json").write_text(json.dumps({"status": gate}))
        return {"status": "completed"}

    snapshot = tmp_path / "sequence.seq"
    snapshot.write_text("frozen submission")
    case = {"case_id": "case", "evaluation": {"id": evaluator}}
    return grade_frozen(snapshot, case, ROOT, tmp_path / "grading", 300, run)


@pytest.mark.parametrize("evaluator", ["rf_excitation", "s1", "s4"])
def test_adaptive_grading_has_no_time_limit(tmp_path, evaluator):
    def completed(command, directory, timeout, _log, **_kwargs):
        if "evaluation" in command:  # Only the parse gate is bounded.
            assert timeout is None and "--time-budget-s" not in command
        (directory / "result.json").write_text(json.dumps({"status": "passed"}))
        return {"status": "completed"}

    assert grade(tmp_path, completed, evaluator)["status"] == "passed"


def test_preflight_gate_decides_before_unbounded_evaluation(tmp_path):
    from mriseqbench.harness.grading import bounded_preflight

    def gate(status, result=None):
        directory = tmp_path / status / str(result)
        directory.mkdir(parents=True)

        def run(command, workdir, timeout, _log, **_kwargs):
            assert "preflight" in command and timeout == 300
            if result:
                (workdir / "result.json").write_text(json.dumps(result))
            return {"status": status}

        case = {"case_id": "case"}
        return bounded_preflight(case, None, None, ROOT, directory, 300, run, None)

    assert gate("timeout")["failure_kind"] == "preflight_time_limit"
    memory = {"status": "failed", "checks": [{"name": "preflight_memory_limit"}]}
    assert gate("completed", memory) == memory  # Never re-parsed unbounded.
    assert gate("completed", {"status": "preflight_passed"}) is None
    for status, result in (
        ("error", None),
        ("launch_error", None),
        ("completed", None),
    ):
        with pytest.raises(EvaluationIncomplete):
            gate(status, result)


@pytest.mark.parametrize("status", ["numerically_unresolved"])
def test_unresolved_verdicts_fail_conservatively(tmp_path, status):
    def unresolved(_command, directory, _timeout, _log, **_kwargs):
        (directory / "result.json").write_text(json.dumps({"status": status}))
        return {"status": "completed"}

    result = grade(tmp_path, unresolved, "s1")
    assert result["status"] == "failed"
    assert result["failure_kind"] == "unresolved_at_sample_cap"
    assert result["unresolved_status"] == status


def test_infrastructure_crashes_still_interrupt(tmp_path):
    def crashes(_command, _directory, _timeout, _log, **_kwargs):
        return {"status": "error"}

    with pytest.raises(EvaluationIncomplete):
        grade(tmp_path, crashes, "s4")


def test_resume_regrades_frozen_bytes_and_never_terminalises_paid_retries(
    tmp_path,
):
    catalog = Catalog(ROOT)
    experiment = Experiment.model_validate(load_yaml(ROOT / "experiments/smoke.yaml"))
    out = tmp_path / "campaign"
    assert run(catalog, experiment, out)[2]
    trial = next(out.glob("*--r0"))
    agent_log = (trial / "agent.execution.json").read_text()
    (trial / "result.json").unlink()
    for path in trial.glob("**/evaluation/result.json"):
        path.unlink()
    for path in trial.glob("control/submissions/*/result.json"):
        path.unlink()
    _, reports, complete = run(catalog, experiment, out)
    assert complete and reports[0]["resumed"] == "regraded_frozen_submission"
    assert reports[0]["status"] == "passed" and reports[0]["scope"] == "smoke"
    assert (trial / "agent.execution.json").read_text() == agent_log
    for name in ("--r1", "--r1.retry1"):  # Paid interruptions, never terminal.
        (out / (trial.name[:-4] + name)).mkdir()
        (out / (trial.name[:-4] + name) / "api.jsonl").write_text(json.dumps(PAID))
    _, reports, complete = run(catalog, experiment, out, repeats=2)
    assert complete and [r["status"] for r in reports] == ["passed", "passed"]
    assert reports[-1]["workspace"].endswith("--r1.retry2")


def test_recovery_selection_runs_only_exact_case_repeat_keys(tmp_path):
    catalog = Catalog(ROOT)
    experiment = Experiment.model_validate(load_yaml(ROOT / "experiments/smoke.yaml"))
    case = catalog.cases(catalog.get("suites", experiment.suite))[0]
    key = case["case_id"] + "--r1"
    out = tmp_path / "selected"
    _, reports, complete = run(catalog, experiment, out, repeats=3, selected_keys={key})
    assert complete and len(reports) == 1 and reports[0]["repeat"] == 1
    assert len(list(out.glob("*/result.json"))) == 1
    assert not list(out.glob("*--r0")) and not list(out.glob("*--r2"))


def test_recovery_selection_rejects_unknown_or_empty_keys_before_authoring(tmp_path):
    catalog = Catalog(ROOT)
    experiment = Experiment.model_validate(load_yaml(ROOT / "experiments/smoke.yaml"))
    for index, selection in enumerate([set(), {"not-a-case--r0"}]):
        out = tmp_path / str(index)
        with pytest.raises(ValueError, match="exact existing case/repeat"):
            run(catalog, experiment, out, selected_keys=selection)
        assert not list(out.glob("*/agent"))


def attempt(workspace, number, status=None):
    directory = workspace / "control/submissions" / f"attempt-{number:03d}"
    directory.mkdir(parents=True)
    (directory / "sequence.seq").write_text("frozen")
    (directory / "receipt.json").write_text(json.dumps({"attempt": number}))
    if status:
        (directory / "result.json").write_text(json.dumps({"status": status}))
    return directory


def test_recover_finalises_only_ended_sessions(tmp_path):
    from mriseqbench.harness.runner import recover

    def ended(name, status="timeout"):
        (tmp_path / name).mkdir(exist_ok=True)
        (tmp_path / name / "session.json").write_text(json.dumps({"ended": status}))

    # Mid-session interruptions void the trial: rerun fresh, never finalise
    # an intermediate failure or grade a non-final attempt.
    attempt(tmp_path / "grading", 1)
    assert recover(tmp_path / "grading", 3) is None
    attempt(tmp_path / "open", 1, "failed")
    assert recover(tmp_path / "open", 3) is None
    assert recover(tmp_path / "empty", 3) is None
    # The last accepted submission ended the session.
    passed = attempt(tmp_path / "passed", 1, "passed")
    assert recover(tmp_path / "passed", 3) == ("regrade", passed)
    attempt(tmp_path / "final", 1, "failed")
    final = attempt(tmp_path / "final", 2)
    assert recover(tmp_path / "final", 2) == ("regrade", final)
    # A recorded genuine session end completes from the files.
    last = attempt(tmp_path / "quit", 1, "failed")
    ended("quit", "completed")
    assert recover(tmp_path / "quit", 3) == ("regrade", last)
    ended("idle")
    assert recover(tmp_path / "idle", 3) == ("closed", {"ended": "timeout"})
    judge = tmp_path / "idle/evaluation"
    judge.mkdir()
    (judge / "sequence.seq").write_text("final file")
    assert recover(tmp_path / "idle", 3) == ("regrade", judge)


def test_genuine_session_ends_fail_with_a_kind():
    from mriseqbench.harness.runner import no_submission

    case = {"case_id": "c"}
    assert no_submission(case, "completed")["failure_kind"] == "no_submission"
    assert no_submission(case, "timeout") == {
        "case_id": "c",
        "status": "failed",
        "failure_kind": "agent_timeout",
        "checks": [],
        "metrics": {},
    }


def test_cached_verdicts_are_normalised_or_regraded(tmp_path, monkeypatch):
    from mriseqbench.harness import runner

    calls = []

    def grade_frozen(*_args, **_kwargs):
        calls.append(1)
        return {"status": "failed"}

    monkeypatch.setattr(runner, "grade_frozen", grade_frozen)
    experiment = Experiment.model_validate(load_yaml(ROOT / "experiments/smoke.yaml"))
    case = {"case_id": "case", "evaluation": {"id": "s1"}}
    directory = attempt(tmp_path / "a", 1, "numerically_unresolved")
    report = runner.regrade(
        Catalog(ROOT), experiment, case, tmp_path / "a", 0, directory
    )
    assert report["status"] == "failed" and not calls
    assert report["failure_kind"] == "unresolved_at_sample_cap"
    directory = attempt(tmp_path / "b", 1, "agent_crashed")
    report = runner.regrade(
        Catalog(ROOT), experiment, case, tmp_path / "b", 0, directory
    )
    assert report["status"] == "failed" and calls == [1]


def test_regrade_holds_a_grading_slot(tmp_path, monkeypatch):
    import threading

    from mriseqbench.harness import runner

    slots = threading.BoundedSemaphore(1)

    def grade_frozen(_snapshot, _case, _root, directory, *_args, **_kwargs):
        assert not slots.acquire(blocking=False)
        (directory / "result.json").write_text(json.dumps({"status": "passed"}))
        return {"status": "passed"}

    monkeypatch.setattr(runner, "grade_frozen", grade_frozen)
    experiment = Experiment.model_validate(load_yaml(ROOT / "experiments/smoke.yaml"))
    directory = attempt(tmp_path, 1)
    case = {"case_id": "case", "evaluation": {"id": "s1"}}
    report = runner.regrade(
        Catalog(ROOT), experiment, case, tmp_path, 0, directory, slots
    )
    assert report["status"] == "passed" and slots.acquire(blocking=False)
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "passed"


def test_campaign_lock_precedes_side_effects(tmp_path, monkeypatch):
    import fcntl

    from mriseqbench.harness import runner

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ran before the campaign lock")

    monkeypatch.setattr(runner, "source_identity", forbidden)
    monkeypatch.setattr(runner, "resolve_runtime", forbidden)
    monkeypatch.setattr(runner.arm_runtime, "remove_campaign_containers", forbidden)
    out = tmp_path / "campaign"
    out.mkdir()
    holder = (out / ".lock").open("a")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    experiment = Experiment.model_validate(load_yaml(ROOT / "experiments/smoke.yaml"))
    with pytest.raises(ValueError, match="another process"):
        run(Catalog(ROOT), experiment, out)
    holder.close()


def test_live_grading_capacity_preserves_campaign_identity_and_final_results(
    tmp_path, monkeypatch
):
    from mriseqbench.harness import runner
    from mriseqbench.harness.grading_pool import atomic_json

    monkeypatch.delenv("MRISEQBENCH_GRADING_GPU_CONFIG", raising=False)
    catalog = Catalog(ROOT)
    experiment = Experiment.model_validate(load_yaml(ROOT / "experiments/smoke.yaml"))
    out = tmp_path / "campaign"
    assert run(catalog, experiment, out)[2]
    identity = (out / "campaign.json").read_bytes()
    finals = {p: p.read_bytes() for p in out.glob("*/result.json")}
    atomic_json(out / "grading-slots.json", {"slots": 4})

    def forbidden(*_args, **_kwargs):
        raise AssertionError("resizing reran an already completed trial")

    monkeypatch.setattr(runner, "run_one", forbidden)
    assert run(catalog, experiment, out)[2]
    assert (out / "campaign.json").read_bytes() == identity
    assert all(p.read_bytes() == value for p, value in finals.items())
    accounting = json.loads((out / "accounting.json").read_text())
    assert accounting["grading_pool"]["configured_slots"] == 4


def test_launch_error_leaves_trial_resumable(tmp_path):
    raw = load_yaml(ROOT / "experiments/smoke.yaml")
    raw["agent"]["command"] = [str(tmp_path / "missing-agent")]
    experiment = Experiment.model_validate(raw)
    out = tmp_path / "campaign"
    _, reports, complete = run(Catalog(ROOT), experiment, out)
    assert not complete and not reports
    trial = next(out.glob("*--r0"))
    accounting = json.loads((trial / "accounting.json").read_text())
    assert accounting["status"] == "interrupted"
    assert not (trial / "result.json").exists()


def test_native_disagreements_are_flagged_for_review(tmp_path):
    raw = load_yaml(ROOT / "experiments/smoke.yaml")
    raw["agent"]["command"] = [str(tmp_path / "missing-agent")]
    experiment = Experiment.model_validate(raw)
    out = tmp_path / "campaign"
    run(Catalog(ROOT), experiment, out)
    trial = next(out.glob("*--r0"))
    detail = "KomaMRI: Native consistency: RF count differs"
    (trial / "result.json").write_text(json.dumps({"checks": [{"detail": detail}]}))
    run(Catalog(ROOT), experiment, out)
    accounting = json.loads((out / "accounting.json").read_text())
    assert accounting["needs_review"] == [trial.name]


@pytest.mark.parametrize(
    ("requests", "failed"),
    [
        ([], False),
        ([{"status": 200, "response_complete": True}], False),
        ([{"status": 200, "response_complete": False}], True),
        ([{"status": 200, "response_complete": False, "cancelled": True}], False),
        ([{"status": None}], True),
        ([{"status": 200, "provider_error": "x"}], True),
        ([{"status": 200, "response_complete": True}, {"status": 429}], True),
    ],
)
def test_model_api_failure(requests, failed):
    from mriseqbench.harness.runner import model_api_failure

    assert (model_api_failure(requests) is not None) == failed


def test_source_identity_covers_catalog_docs_and_paths(tmp_path):
    from mriseqbench.harness.runner import source_identity

    reference = tmp_path / "benchmark/references/case/target.json"
    reference.parent.mkdir(parents=True)
    reference.write_text("{}")
    for name in ("Project.toml", "LocalPreferences.toml"):
        before = source_identity(tmp_path)
        (tmp_path / name).write_text("[x]")
        assert source_identity(tmp_path) != before
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/S1_EVALUATION.md").write_text("rules")
    base = source_identity(tmp_path)
    reference.write_text('{"changed": true}')
    changed = source_identity(tmp_path)
    assert changed != base
    (tmp_path / "docs/S1_EVALUATION.md").write_text("new rules")
    assert source_identity(tmp_path) != changed
    moved = tmp_path / "benchmark/references/other/target.json"
    moved.parent.mkdir()
    before = source_identity(tmp_path)
    reference.rename(moved)
    assert source_identity(tmp_path) != before


def test_clean_agent_exit_survives_a_crash_before_the_session_marker(tmp_path):
    from mriseqbench.harness.runner import recover

    def trial(name, mode="submit", infra=False, api=None):
        workspace = tmp_path / name
        (workspace / "control").mkdir(parents=True)
        (workspace / "session-exit.json").write_text(
            json.dumps({"ended": "timeout", "submission_mode": mode})
        )
        if infra:
            (workspace / "control/infrastructure-failure.json").write_text("{}")
        if api:
            (workspace / "api.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in api)
            )
        return recover(workspace, 3)

    assert trial("clean") == ("closed", {"ended": "timeout"})
    assert trial("infra", infra=True) is None
    assert trial("file", mode="file") is None
    truncated = [
        {"event": "started", "request_id": "r", "route": "/responses"},
        {
            "event": "finished",
            "request_id": "r",
            "route": "/responses",
            "status": 200,
            "response_complete": False,
        },
    ]
    assert trial("truncated", api=truncated) is None


def test_agent_access_ends_before_launcher_teardown(tmp_path, monkeypatch):
    """A timed-out agent's surviving processes cannot submit during cleanup."""
    from contextlib import contextmanager

    from mriseqbench.harness import runner

    servers, late = [], []

    class Server(runner.FeedbackServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            servers.append((self, self.token))

    @contextmanager
    def launch(*_args, **_kwargs):
        yield ["agent"], {"backend": "test"}
        server, token = servers[-1]
        late.append(server.handle("submit", token, b"late sequence"))

    monkeypatch.setattr(runner, "FeedbackServer", Server)
    monkeypatch.setattr(runner, "launch", launch)
    monkeypatch.setattr(
        runner,
        "execute",
        lambda *a, **k: {"status": "timeout", "returncode": None},
    )
    experiment = Experiment.model_validate(load_yaml(ROOT / "experiments/smoke.yaml"))
    _, reports, _ = run(Catalog(ROOT), experiment, tmp_path / "c", repeats=1)
    assert late and late[0][0] == 403
    assert reports and all(r["status"] == "failed" for r in reports)


def test_campaign_abort_ends_a_queued_grading_wait():
    import threading

    from mriseqbench.harness.grading import grading_slot

    slots, abort = threading.BoundedSemaphore(1), threading.Event()
    slots.acquire()  # Another trial is grading.
    threading.Timer(0.2, abort.set).start()
    with pytest.raises(EvaluationIncomplete), grading_slot(slots, abort):
        pass


def test_regrade_passes_the_campaign_abort_to_grading(tmp_path, monkeypatch):
    import threading

    from mriseqbench.harness import runner

    seen = {}

    def grade_frozen(*_args, stop_when=None, **_kwargs):
        seen["stop_when"] = stop_when
        raise EvaluationIncomplete("aborted")

    monkeypatch.setattr(runner, "grade_frozen", grade_frozen)
    experiment = Experiment.model_validate(load_yaml(ROOT / "experiments/smoke.yaml"))
    abort = threading.Event()
    with pytest.raises(EvaluationIncomplete):
        runner.regrade(
            Catalog(ROOT),
            experiment,
            {"case_id": "case", "evaluation": {"id": "s1"}},
            tmp_path,
            0,
            attempt(tmp_path, 1),
            abort_event=abort,
        )
    assert seen["stop_when"] is abort


def test_campaign_abort_stops_feedback_grading_during_drain(tmp_path):
    import threading

    from mriseqbench.harness.feedback import FeedbackServer
    from mriseqbench.models import FeedbackConfig

    abort = threading.Event()
    server = FeedbackServer(
        {"case_id": "case", "evaluation": {"id": "s1"}},
        FeedbackConfig(),
        ROOT,
        tmp_path,
        abort=abort,
    )
    assert not server.stopping.is_set()
    abort.set()  # The agent has exited; only the drain is left.
    assert server.stopping.is_set() and not server.closed.is_set()
    assert server.handle("lint", server.token, b"seq")[0] == 403
