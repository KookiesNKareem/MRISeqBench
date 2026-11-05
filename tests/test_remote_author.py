import base64
import json
import subprocess
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from mriseqbench.harness.arm_mailbox import Mailbox
from mriseqbench.harness.remote_author import Fleet, manifest


def test_remote_mailbox_streams_through_same_feedback_policy(tmp_path):
    config = tmp_path / "fleet.json"
    config.write_text(json.dumps({"hosts": [], "assignments": {}}))
    fleet = Fleet(config)
    directory = tmp_path / "transport"

    class Feedback:
        def handle(self, route, token, body):
            assert (route, token, body) == ("status", "trial-only", b"")
            return 200, {"remaining_s": 14400}

    try:
        with Mailbox(directory, Feedback()):
            fleet.tokens["secret"] = directory
            payload = {
                "target": "feedback",
                "path": "/status",
                "method": "GET",
                "token": "trial-only",
                "body": "",
            }
            req = urllib.request.Request(
                f"http://127.0.0.1:{fleet.server.server_port}/secret/" + "a" * 32,
                data=json.dumps(payload).encode(),
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                frames = [json.loads(line) for line in response]
            assert frames[0]["status"] == 200
            assert json.loads(base64.b64decode(frames[1]["data"])) == {
                "remaining_s": 14400
            }
            assert frames[-1]["done"] is True
            req = urllib.request.Request(
                f"http://127.0.0.1:{fleet.server.server_port}/wrong/" + "b" * 32,
                data=json.dumps(payload).encode(),
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(req, timeout=5)
            assert error.value.code == 403
    finally:
        fleet.close()


def test_author_manifest_detects_mutation_without_following_links(tmp_path):
    file = tmp_path / "sequence.seq"
    file.write_bytes(b"original")
    (tmp_path / "link").symlink_to("/etc/passwd")
    before = manifest(tmp_path)
    assert set(before) == {"sequence.seq"}
    file.write_bytes(b"mutated")
    assert before != manifest(tmp_path)


@pytest.mark.parametrize("disconnected", [False, True])
def test_remote_author_inspection_does_not_require_local_docker(
    monkeypatch, disconnected
):
    from mriseqbench.harness import arm_runtime, remote_author

    def local_docker(*args, **kwargs):
        pytest.fail("Remote author inspection must not use the local Docker daemon")

    def remote(*args, **kwargs):
        if disconnected:
            raise subprocess.CalledProcessError(255, "ssh")
        return SimpleNamespace(stdout=json.dumps({"ExitCode": 1, "OOMKilled": False}))

    monkeypatch.setattr(arm_runtime, "docker_output", local_docker)
    monkeypatch.setattr(remote_author, "remote", remote)
    reason = arm_runtime.infrastructure_failure(
        {"container": "author", "remote_host": {"endpoint": "example"}},
        {"status": "error", "returncode": 1},
    )
    assert (reason is not None) == disconnected


def test_failed_remote_transfer_revokes_token_and_releases_slot(tmp_path, monkeypatch):
    from mriseqbench.harness import arm_runtime, remote_author

    workspace = tmp_path / "trial" / "agent"
    control = workspace.parent / "control"
    workspace.mkdir(parents=True)
    control.mkdir()
    (workspace / "sequence.seq").write_text("returned content")
    host = {"endpoint": "example", "key": "/unused", "instance_id": "test"}
    fleet = Fleet.__new__(Fleet)
    fleet.config = {"assignments": {"trial": 0}, "state_dir": str(tmp_path)}
    fleet.hosts = [host]
    fleet.slots = [threading.Semaphore(1)]
    fleet.pending = [1]
    fleet.tokens = {}
    fleet.lock = threading.Lock()
    fleet.helper_root = "/unused"
    fleet.broker_port = 8766

    def remote(host, args, **kwargs):
        result = {"sequence.seq": "incorrect-hash"} if "python3" in args else {}
        return SimpleNamespace(stdout=json.dumps(result))

    monkeypatch.setattr(remote_author, "remote", remote)
    monkeypatch.setattr(remote_author, "sync", lambda *args: None)
    monkeypatch.setattr(
        arm_runtime,
        "container_args",
        lambda image, name: (["docker", "run", "--rm", "--user", "1000:1000"], image),
    )
    with (
        pytest.raises(RuntimeError, match="transfer failure"),
        fleet.launch(
            {"author_image": "test", "author_memory_mb": 1024},
            ["author"],
            workspace,
            control,
            {},
        ),
    ):
        assert fleet.tokens
    assert not fleet.tokens
    assert fleet.slots[0].acquire(blocking=False)
    assert not (tmp_path / "test-drain-request.json").exists()


@pytest.mark.parametrize("reported_simulator", ["none", "koma"])
def test_remote_mcp_checks_arm_before_admitting_pi(
    tmp_path, monkeypatch, reported_simulator
):
    from mriseqbench.harness import arm_mcp, arm_runtime, remote_author

    workspace = tmp_path / "trial" / "agent"
    control = workspace.parent / "control"
    workspace.mkdir(parents=True)
    control.mkdir()
    fleet = Fleet.__new__(Fleet)
    fleet.config = {"assignments": {"trial": 0}}
    fleet.hosts = [{"endpoint": "example", "key": "/unused", "instance_id": "test"}]
    fleet.slots = [threading.Semaphore(1)]
    fleet.pending = [1]
    fleet.tokens = {}
    fleet.lock = threading.Lock()
    fleet.helper_root = "/unused"
    fleet.broker_port = 8766
    calls = []

    def remote(host, args, **kwargs):
        calls.append(args)
        result = (
            {
                "mode": "persistent_mcp",
                "pulse_designer": True,
                "simulator": reported_simulator,
            }
            if "--ready" in args
            else {}
        )
        return SimpleNamespace(stdout=json.dumps(result))

    monkeypatch.setattr(remote_author, "remote", remote)
    monkeypatch.setattr(remote_author, "sync", lambda *args: None)
    monkeypatch.setattr(
        arm_runtime,
        "container_args",
        lambda image, name: (["docker", "run", "--rm", "--user", "1000:1000"], image),
    )
    runtime = {
        "author_image": "test",
        "author_memory_mb": 2048,
        "julia_heap_size_hint_mb": 100,
        "setup_timeout_s": 600,
        "arm": {
            "tool_interface": "julia_mcp",
            "pulse_designer": True,
            "simulator": "none",
        },
    }

    def launch():
        with fleet.launch(runtime, ["pi", "-p", "control"], workspace, control, {}) as (
            _,
            isolation,
        ):
            assert isolation["persistent_container"]
            assert isolation["active_tools"] == arm_mcp.TOOLS
            spec = json.loads((control / "remote-author-spec.json").read_text())
            assert spec["command"][:3] == ["sudo", "docker", "exec"]
            assert json.loads(spec["command"][-1]) == arm_mcp.pi_command(
                ["pi", "-p", "control"]
            )

    if reported_simulator == "none":
        launch()
    else:
        with pytest.raises(ValueError, match="not the selected arm"):
            launch()
        assert not (control / "remote-author-spec.json").exists()
    assert any("--serve" in args and "--pulse-designer" in args for args in calls)
    assert any(args[:4] == ["sudo", "docker", "rm", "--force"] for args in calls)
    assert (
        json.loads((control / "remote-author-transfer.json").read_text())["status"]
        == "verified"
    )
    assert not fleet.tokens
    assert fleet.slots[0].acquire(blocking=False)


@pytest.mark.parametrize("ready", [True, False])
def test_remote_cli_julia_requires_warm_session_and_preserves_pi_tools(
    tmp_path, monkeypatch, ready
):
    from mriseqbench.harness import arm_runtime, remote_author

    workspace = tmp_path / "trial" / "agent"
    control = workspace.parent / "control"
    workspace.mkdir(parents=True)
    control.mkdir()
    fleet = Fleet.__new__(Fleet)
    fleet.config = {"assignments": {"trial": 0}}
    fleet.hosts = [{"endpoint": "example", "key": "/unused", "instance_id": "test"}]
    fleet.slots = [threading.Semaphore(1)]
    fleet.pending = [1]
    fleet.tokens = {}
    fleet.lock = threading.Lock()
    fleet.helper_root = "/unused"
    fleet.broker_port = 8766
    calls = []

    def remote(host, args, **kwargs):
        calls.append(args)
        result = (
            {"mode": "persistent" if ready else "wrong", "revise_version": "3"}
            if "--ready" in args
            else {}
        )
        return SimpleNamespace(stdout=json.dumps(result))

    monkeypatch.setattr(remote_author, "remote", remote)
    monkeypatch.setattr(remote_author, "sync", lambda *args: None)
    monkeypatch.setattr(
        arm_runtime,
        "container_args",
        lambda image, name: (["docker", "run", "--rm", "--user", "1000:1000"], image),
    )
    runtime = {
        "author_image": "test",
        "author_memory_mb": 2048,
        "julia_heap_size_hint_mb": 100,
        "setup_timeout_s": 600,
        "arm": {
            "tool_interface": "cli",
            "language": "julia",
            "pulse_designer": True,
            "unitful": False,
            "simulator": "koma",
        },
    }
    command = ["pi", "-p", "control"]

    def launch():
        with fleet.launch(runtime, command, workspace, control, {}) as (_, isolation):
            assert isolation["persistent_container"]
            assert isolation["julia_session"]["mode"] == "persistent"
            spec = json.loads((control / "remote-author-spec.json").read_text())
            assert json.loads(spec["command"][-1]) == command

    if ready:
        launch()
    else:
        with pytest.raises(ValueError, match="warm readiness"):
            launch()
        assert not (control / "remote-author-spec.json").exists()
    start = next(args for args in calls if "--detach" in args)
    assert start[-3:] == ["/arm_julia_session.jl", "true", "false"]
    assert "--compiled-modules=existing" in start
    assert any(args[:4] == ["sudo", "docker", "rm", "--force"] for args in calls)
    assert not fleet.tokens
    assert fleet.slots[0].acquire(blocking=False)
