"""Remote transport integrity and disconnect controls, without renting a GPU."""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "remote_gpu_dispatch", ROOT / "ops/remote_gpu_dispatch.py"
)
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)


def test_transfer_rejects_corruption_and_unlisted_files(tmp_path):
    (tmp_path / "result.json").write_text('{"status":"passed"}')
    (tmp_path / "remote-transfer-manifest.json").write_text(
        json.dumps(remote.manifest(tmp_path))
    )
    remote.verify(tmp_path)
    (tmp_path / "result.json").write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match="SHA-256"):
        remote.verify(tmp_path)
    (tmp_path / "result.json").write_text('{"status":"passed"}')
    (tmp_path / "extra").write_text("unexpected")
    with pytest.raises(ValueError, match="SHA-256"):
        remote.verify(tmp_path)


def test_symlink_cannot_escape_transfer_manifest(tmp_path):
    (tmp_path / "escape").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlink"):
        remote.manifest(tmp_path)


def test_registry_requires_qualified_host_and_safe_endpoint(tmp_path):
    config = tmp_path / "pool.json"
    assert remote.remote_hosts(config) == {}
    host = {
        "endpoint": "root@host.example",
        "port": 22,
        "key": "/private/key",
        "root": "/frozen/source",
        "gpu_name": "H100",
        "qualified_at": "today",
    }
    registry = tmp_path / "remote-grading-hosts.json"
    registry.write_text(json.dumps({"GPU-h100": host}))
    assert remote.remote_hosts(config)["GPU-h100"] == host
    host["endpoint"] = "root@host; false"
    registry.write_text(json.dumps({"GPU-h100": host}))
    with pytest.raises(ValueError, match="endpoint"):
        remote.remote_hosts(config)


def test_supervisor_eof_kills_worker_and_descendant(tmp_path):
    root = tmp_path / "source"
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").symlink_to(sys.executable)
    package = root / "mriseqbench/evaluation"
    package.mkdir(parents=True)
    (package / "worker.py").write_text(
        "import subprocess, time, pathlib, os\n"
        "child = subprocess.Popen(['sleep', '300'])\n"
        "pathlib.Path(os.environ['PROBE']).write_text(str(child.pid))\n"
        "time.sleep(300)\n"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    probe = tmp_path / "pid"
    child = subprocess.Popen(
        [sys.executable, str(ROOT / "ops/remote_gpu_dispatch.py"), "--serve"],
        stdin=subprocess.PIPE,
        text=True,
    )
    request = {
        "root": str(root),
        "workspace": str(workspace),
        "argv": [],
        "env": {"PYTHONPATH": str(root), "PROBE": str(probe)},
    }
    child.stdin.write(json.dumps(request) + "\n")
    child.stdin.flush()
    try:
        deadline = time.monotonic() + 5
        while not probe.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        pid = int(probe.read_text())
        child.stdin.close()
        assert child.wait(timeout=10) == 125
        remote.verify(workspace)
        assert not (workspace / "result.json").exists()
        # A killed descendant can briefly remain a zombie until reaped by PID 1.
        stat = Path(f"/proc/{pid}/stat")
        assert not stat.exists() or stat.read_text().split()[2] == "Z"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


@pytest.mark.parametrize("corrupt", [False, True])
def test_dispatch_publishes_only_verified_results(tmp_path, monkeypatch, corrupt):
    import io

    root = tmp_path / "source with spaces"
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    case = workspace / "case.json"
    case.write_text("{}")
    submission = tmp_path / "sequence with spaces.seq"
    submission.write_text("test input")
    host = {
        "root": str(root),
        "endpoint": "root@host",
        "port": 22,
        "key": "/private/key",
    }
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        if command[0] == "rsync" and command[-2].startswith("root@host:"):
            stage = Path(command[-1])
            (stage / "result.json").write_text('{"status":"passed"}')
            (stage / "remote-transfer-manifest.json").write_text(
                json.dumps(remote.manifest(stage))
            )
            if corrupt:
                (stage / "result.json").write_text('{"status":"failed"}')
        return subprocess.CompletedProcess(command, 0)

    class Child:
        def __init__(self, command, **_kwargs):
            commands.append(command)
            self.stdin = io.StringIO()

        def wait(self):
            return 0

    monkeypatch.setattr(remote.subprocess, "run", run)
    monkeypatch.setattr(remote.subprocess, "Popen", Child)
    args = [
        "--mode",
        "evaluation",
        "--root",
        str(root),
        "--workspace",
        str(workspace),
        "--case",
        str(case),
        "--submission",
        str(submission),
    ]
    if corrupt:
        with pytest.raises(ValueError, match="SHA-256"):
            remote.dispatch(host, "GPU-test", args)
        assert not (workspace / "result.json").exists()
        assert not any("rm -rf" in c[-1] for c in commands)
    else:
        assert remote.dispatch(host, "GPU-test", args) == 0
        assert json.loads((workspace / "result.json").read_text())["status"] == "passed"
    # Remote shell commands quote paths; rsync handles names as protected argv.
    assert any("'" + str(workspace) + "'" in c[-1] for c in commands if c[0] == "ssh")
