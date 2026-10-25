"""Infrastructure vs participant failures in the arm runtime (Docker mocked)."""

import json
import subprocess
import threading
from types import SimpleNamespace

import pytest

from mriseqbench.harness import arm_runtime
from mriseqbench.harness.arm_runtime import DockerUnavailable, Simulator, SimulatorSlots
from mriseqbench.harness.grading import EvaluationIncomplete


def fake_docker(monkeypatch, state=None, info_ok=True):
    def docker_output(args, timeout=120):
        if args[0] == "info":
            if not info_ok:
                raise DockerUnavailable("daemon down")
            return "24.0.2"
        if args[0] == "inspect":
            if state is None:
                raise ValueError("Error: No such object")
            return json.dumps(state)
        raise AssertionError(args)

    monkeypatch.setattr(arm_runtime, "docker_output", docker_output)


@pytest.mark.parametrize(
    ("persistent", "state", "status", "infra"),
    [
        (False, {"ExitCode": 137, "OOMKilled": False}, "error", True),
        (False, {"ExitCode": 137, "OOMKilled": True}, "error", False),
        (False, {"ExitCode": 1}, "error", False),
        (False, {"ExitCode": 247}, "error", False),  # agent's process got SIGKILL
        (False, {"Running": True}, "error", True),
        (False, None, "error", True),  # container gone
        (False, {"ExitCode": 137}, "timeout", False),  # harness ended it
        (True, {"Running": False, "ExitCode": 137}, "completed", True),
        (True, {"Running": False, "OOMKilled": True}, "error", False),
        (True, {"Running": True}, "completed", False),
    ],
)
def test_author_container_loss_rule(monkeypatch, persistent, state, status, infra):
    fake_docker(monkeypatch, state)
    isolation = {"container": "c", "persistent_container": persistent}
    reason = arm_runtime.infrastructure_failure(
        isolation, {"status": status, "returncode": 1}
    )
    assert (reason is not None) == infra


def test_daemon_loss_is_infrastructure(monkeypatch):
    fake_docker(monkeypatch, {"ExitCode": 0}, info_ok=False)
    assert "daemon" in arm_runtime.infrastructure_failure(
        {"container": "c"}, {"status": "error", "returncode": 1}
    )


def test_python_author_container_is_retained_for_inspection(tmp_path, monkeypatch):
    monkeypatch.setattr(arm_runtime.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(arm_runtime, "rootless", lambda: True)
    monkeypatch.setattr(
        arm_runtime.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", ""),
    )
    removed = []
    monkeypatch.setattr(arm_runtime, "ensure_removed", removed.append)
    (tmp_path / "transport").mkdir()
    runtime = {
        "arm": {"language": "python", "tool_interface": "cli"},
        "author_image": "sha256:" + "0" * 64,
        "author_memory_mb": 2048,
    }
    with arm_runtime.launch_arm(runtime, ["pi"], tmp_path, tmp_path, {}) as (
        args,
        metadata,
    ):
        assert "--rm" not in args and metadata["container"] in args
    assert removed == [metadata["container"]]


def test_docker_transport_errors_are_retryable(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("docker", 1)

    monkeypatch.setattr(arm_runtime.subprocess, "run", timeout)
    with pytest.raises(DockerUnavailable):
        arm_runtime.docker_output(["info"])
    for stderr, error in (
        ("Cannot connect to the Docker daemon at unix:///x.sock", DockerUnavailable),
        ("Error: No such image: x", ValueError),
    ):
        monkeypatch.setattr(
            arm_runtime.subprocess,
            "run",
            lambda *a, stderr=stderr, **k: subprocess.CompletedProcess(
                a, 1, "", stderr
            ),
        )
        with pytest.raises(error):
            arm_runtime.docker_output(["image", "inspect", "x"])


def test_cli_exits_retryably_when_docker_is_unavailable(monkeypatch, tmp_path):
    from mriseqbench import cli

    def run(*args, **kwargs):
        raise DockerUnavailable("daemon down")

    monkeypatch.setattr(cli, "run", run)
    root = arm_runtime.Path(__file__).resolve().parents[1]
    code = cli.main(
        [
            "--root",
            str(root),
            "run",
            "--experiment",
            str(root / "experiments/smoke.yaml"),
            "--out",
            str(tmp_path / "c"),
        ]
    )
    assert code == 4


def simulator(
    tmp_path, monkeypatch, execution, state, stderr="", removed=True, slots=None
):
    (tmp_path / "simulation_inputs").mkdir(exist_ok=True)
    monkeypatch.setattr(
        arm_runtime,
        "container_args",
        lambda image, name=None: (["docker", "--rm"], image),
    )
    monkeypatch.setattr(arm_runtime, "configure_simulator_args", lambda args: args)
    monkeypatch.setattr(arm_runtime, "remove_container", lambda name: removed)
    fake_docker(monkeypatch, state)

    def execute(args, cwd, timeout, prefix, **_kwargs):
        assert "--rm" not in args
        prefix.with_name(prefix.name + ".stdout.log").write_text("")
        prefix.with_name(prefix.name + ".stderr.log").write_text(stderr)
        return execution

    monkeypatch.setattr(arm_runtime, "execute", execute)
    slots, reasons = slots or SimulatorSlots(1, retries=1, interval=0), []
    sim = Simulator(
        {"simulator_image": "sim", "arm": {"simulator": "koma"}},
        SimpleNamespace(simulation_timeout_s=10),
        tmp_path,
        tmp_path,
        {},
        slots=slots,
        infrastructure=reasons.append,
    )
    status, body = sim(b"seq")
    return status, body, reasons, slots.free.acquire(blocking=False)


@pytest.mark.parametrize(
    ("execution", "state", "stderr", "infra"),
    [
        (
            {"status": "error", "returncode": 1},
            {"ExitCode": 1},
            "ValueError: nonuniform B1",
            False,
        ),
        ({"status": "timeout", "returncode": None}, {"ExitCode": 137}, "", False),
        (
            {"status": "error", "returncode": 137},
            {"ExitCode": 137, "OOMKilled": True},
            "",
            False,
        ),
        ({"status": "launch_error", "error": "x"}, None, "", True),
        ({"status": "error", "returncode": 125}, None, "", True),
        ({"status": "error", "returncode": 137}, {"ExitCode": 137}, "", True),
        (
            {"status": "error", "returncode": 1},
            {"ExitCode": 1},
            "CUDA is required; no CPU fallback",
            True,
        ),
        (
            {"status": "error", "returncode": 1},
            {"ExitCode": 1},
            "Simulator image was not prewarmed on a GPU",
            True,
        ),
        ({"status": "error", "returncode": 1}, None, "", True),
    ],
)
def test_simulator_failure_classification(
    tmp_path, monkeypatch, execution, state, stderr, infra
):
    status, _body, reasons, slot_free = simulator(
        tmp_path, monkeypatch, execution, state, stderr
    )
    assert status == (503 if infra else 422)
    assert bool(reasons) == infra and slot_free


def test_unremoved_simulator_container_keeps_its_slot_and_latches(
    tmp_path, monkeypatch
):
    status, _body, reasons, slot_free = simulator(
        tmp_path,
        monkeypatch,
        {"status": "timeout", "returncode": None},
        {"Running": True},
        removed=False,
    )
    assert status == 503 and reasons and not slot_free


def test_unremoved_container_slot_returns_once_cleanup_succeeds(tmp_path, monkeypatch):
    slots = SimulatorSlots(1, retries=3, interval=0)
    results = iter([False, False, True])  # first removal, one failed retry
    monkeypatch.setattr(arm_runtime, "remove_container", lambda name: next(results))
    slots.free.acquire()
    slots.reap("c")
    assert slots.free.acquire(timeout=2) and not slots.broken.is_set()


def test_lost_slots_interrupt_waiting_trials(tmp_path, monkeypatch):
    slots = SimulatorSlots(1, retries=1, interval=0)
    monkeypatch.setattr(arm_runtime, "remove_container", lambda name: False)
    slots.free.acquire()
    slots.reap("c")
    # Keep removal failing when the helper replaces the Docker mocks, so the
    # background reaper cannot accidentally recover the slot.
    status, _body, reasons, _ = simulator(
        tmp_path,
        monkeypatch,
        {"status": "completed"},
        None,
        removed=False,
        slots=slots,
    )
    assert slots.broken.is_set() and status == 503 and reasons


def test_queued_request_is_released_when_the_trial_ends(tmp_path, monkeypatch):
    slots = SimulatorSlots(1)
    slots.free.acquire()
    monkeypatch.setattr(
        arm_runtime, "container_args", lambda image, name=None: (["--rm"], image)
    )
    monkeypatch.setattr(arm_runtime, "configure_simulator_args", lambda args: args)
    (tmp_path / "simulation_inputs").mkdir()
    reasons = []
    sim = Simulator(
        {"simulator_image": "sim", "arm": {"simulator": "koma"}},
        SimpleNamespace(simulation_timeout_s=10),
        tmp_path,
        tmp_path,
        {},
        slots=slots,
        infrastructure=reasons.append,
    )
    threading.Timer(0.2, sim.closed.set).start()
    status, _body = sim(b"seq")
    assert status == 503 and not reasons


def test_simulator_setup_failure_latches(tmp_path, monkeypatch):
    def unavailable(image, name=None):
        raise DockerUnavailable("daemon down")

    monkeypatch.setattr(arm_runtime, "container_args", unavailable)
    (tmp_path / "simulation_inputs").mkdir()
    slots, reasons = SimulatorSlots(1), []
    sim = Simulator(
        {"simulator_image": "sim", "arm": {"simulator": "koma"}},
        SimpleNamespace(simulation_timeout_s=10),
        tmp_path,
        tmp_path,
        {},
        slots=slots,
        infrastructure=reasons.append,
    )
    status, _body = sim(b"seq")
    assert status == 503 and "daemon down" in reasons[0]
    assert slots.free.acquire(blocking=False)


@pytest.mark.parametrize("earlier_attempt", [False, True])
def test_simulator_latch_survives_a_recorded_session_exit(tmp_path, earlier_attempt):
    """Latch -> session-exit.json -> crash -> recovery must rerun, not score."""
    from mriseqbench.harness.feedback import FeedbackServer
    from mriseqbench.harness.runner import recover

    control = tmp_path / "control"
    control.mkdir()
    server = SimpleNamespace(
        directory=control,
        evaluation_failure=None,
        submitted=threading.Event(),
        retry_time_lock=threading.Lock(),
    )
    FeedbackServer.interrupt(server, EvaluationIncomplete("Simulator: GPU lost"))
    assert server.submitted.is_set()
    (tmp_path / "session-exit.json").write_text(
        json.dumps({"ended": "submitted", "submission_mode": "submit"})
    )
    if earlier_attempt:
        attempt = control / "submissions/attempt-0001"
        attempt.mkdir(parents=True)
        (attempt / "receipt.json").write_text("{}")
        (attempt / "result.json").write_text(json.dumps({"status": "failed"}))
    assert recover(tmp_path, 3) is None


def test_recorded_session_never_outranks_the_infrastructure_latch(tmp_path):
    from mriseqbench.harness.runner import recover

    (tmp_path / "control").mkdir()
    (tmp_path / "session.json").write_text(json.dumps({"ended": "timeout"}))
    assert recover(tmp_path, 3) == ("closed", {"ended": "timeout"})
    (tmp_path / "control/infrastructure-failure.json").write_text("{}")
    assert recover(tmp_path, 3) is None


def test_cancelled_trial_never_acquires_a_free_slot():
    slots = SimulatorSlots(1)
    assert not slots.acquire(lambda: True)
    assert slots.free.acquire(blocking=False)  # the slot was returned


def test_mailbox_close_cancels_a_running_simulation(tmp_path, monkeypatch):
    from mriseqbench.harness.arm_mailbox import Mailbox

    monkeypatch.setattr(
        arm_runtime, "container_args", lambda image, name=None: (["--rm"], image)
    )
    monkeypatch.setattr(arm_runtime, "configure_simulator_args", lambda args: args)
    monkeypatch.setattr(arm_runtime, "remove_container", lambda name: True)
    started = threading.Event()

    def execute(args, cwd, timeout, prefix, stop_when=None):
        started.set()
        assert stop_when.wait(5)
        for suffix in (".stdout.log", ".stderr.log"):
            prefix.with_name(prefix.name + suffix).write_text("")
        return {"status": "submitted", "returncode": None}

    monkeypatch.setattr(arm_runtime, "execute", execute)
    (tmp_path / "simulation_inputs").mkdir()
    slots, reasons, result = SimulatorSlots(1), [], {}
    sim = Simulator(
        {"simulator_image": "sim", "arm": {"simulator": "koma"}},
        SimpleNamespace(simulation_timeout_s=300),
        tmp_path,
        tmp_path,
        {},
        slots=slots,
        infrastructure=reasons.append,
    )
    worker = threading.Thread(target=lambda: result.update(r=sim(b"seq")))
    worker.start()
    assert started.wait(5)
    Mailbox(tmp_path / "transport", None, simulator=sim).close()
    worker.join(5)
    assert result["r"][0] == 503 and not reasons
    assert slots.free.acquire(blocking=False)


def test_unremovable_author_container_raises(monkeypatch):
    monkeypatch.setattr(arm_runtime, "remove_container", lambda name: False)
    with pytest.raises(DockerUnavailable):
        arm_runtime.ensure_removed("c", attempts=2, interval=0)


def test_mcp_diagnostic_failure_still_removes_the_container(tmp_path, monkeypatch):
    from mriseqbench.harness import arm_mcp

    removed = []

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("docker", 30)

    def down(*args, **kwargs):
        raise DockerUnavailable("down")

    monkeypatch.setattr(
        arm_runtime,
        "container_args",
        lambda image, name=None: (["docker", "--rm"], image),
    )
    monkeypatch.setattr(arm_runtime, "mount", lambda *a, **k: [])
    monkeypatch.setattr(arm_runtime, "docker_output", down)
    monkeypatch.setattr(arm_runtime, "ensure_removed", removed.append)
    monkeypatch.setattr(arm_mcp.subprocess, "run", timeout)
    runtime = {
        "arm": {"simulator": "koma", "pulse_designer": False},
        "author_image": "img",
        "author_memory_mb": 1024,
        "setup_timeout_s": 10,
        "julia_heap_size_hint_mb": None,
    }
    with (
        pytest.raises(DockerUnavailable),
        arm_mcp.launch_mcp(runtime, ["pi"], tmp_path, tmp_path, {}),
    ):
        pass
    assert len(removed) == 1


def test_campaign_waits_for_pending_simulator_cleanup(monkeypatch):
    slots = SimulatorSlots(1, retries=2, interval=0.1)
    monkeypatch.setattr(arm_runtime, "remove_container", lambda name: False)
    slots.free.acquire()
    slots.reap("c")
    slots.drain()
    assert slots.broken.is_set()


def test_recorded_cache_hit_is_never_finalised(tmp_path):
    from mriseqbench.harness.runner import recover

    (tmp_path / "control").mkdir()
    (tmp_path / "session-exit.json").write_text(
        json.dumps({"ended": "completed", "submission_mode": "submit"})
    )
    row = {
        "event": "finished",
        "request_id": "r",
        "route": "/responses",
        "status": 200,
        "response_complete": False,
        "cancelled": True,
        "response_cache_status": "HIT",
    }
    (tmp_path / "api.jsonl").write_text(
        json.dumps({"event": "started", "request_id": "r", "route": "/responses"})
        + "\n"
        + json.dumps(row)
        + "\n"
    )
    assert recover(tmp_path, 3) is None
    # Nor a terminal attempt (a pass, or the final allowance).
    for number, status in ((1, "passed"), (3, "failed")):
        attempt = tmp_path / f"control/submissions/attempt-{number:03d}"
        attempt.mkdir(parents=True)
        (attempt / "receipt.json").write_text("{}")
        (attempt / "result.json").write_text(json.dumps({"status": status}))
        assert recover(tmp_path, 3) is None
