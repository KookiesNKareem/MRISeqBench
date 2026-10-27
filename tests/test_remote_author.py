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
def test_remote_author_inspection_does_not_require_local_docker(monkeypatch, disconnected):
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
        arm_runtime, "container_args",
        lambda image, name: (["docker", "run", "--rm", "--user", "1000:1000"], image),
    )
    with pytest.raises(RuntimeError, match="transfer failure"), fleet.launch(
        {"author_image": "test", "author_memory_mb": 1024},
        ["author"], workspace, control, {},
    ):
        assert fleet.tokens
    assert not fleet.tokens
    assert fleet.slots[0].acquire(blocking=False)
    assert not (tmp_path / "test-drain-request.json").exists()
