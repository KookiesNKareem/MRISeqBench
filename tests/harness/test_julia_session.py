"""Low-memory lifecycle and client checks; no Docker or Julia is started."""

import json
import threading
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.harness import arm_julia_client, arm_runtime

ROOT = Path(__file__).resolve().parents[2]
JULIA_ARMS = [
    a for a in Catalog(ROOT).documents["arms"].values() if a.language == "julia"
]


@pytest.mark.parametrize("arm", JULIA_ARMS, ids=lambda arm: arm.id)
def test_session_warms_before_agent_launch_and_is_always_removed(
    monkeypatch, tmp_path, arm
):
    calls, removed = [], []
    ready = {"mode": "persistent", "pid": 7, "revise_version": "3.16.2"}

    def docker_output(args, timeout):
        calls.append(args)
        return json.dumps(ready) if args[0] == "exec" else "container-id"

    monkeypatch.setattr(arm_runtime.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(arm_runtime, "docker_output", docker_output)
    monkeypatch.setattr(arm_runtime, "remove_container", removed.append)
    monkeypatch.setattr(
        arm_runtime.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout="warmup log", stderr=""),
    )
    runtime = {
        "arm": arm.model_dump(mode="json"),
        "author_image": "sha256:" + "a" * 64,
        "simulator_image": None,
        "audit": {"ok": True},
        "julia_heap_size_hint_mb": 200,
    }
    control = tmp_path / "control"
    control.mkdir()
    with arm_runtime.launch_arm(runtime, ["pi"], tmp_path, control, {}) as (
        command,
        metadata,
    ):
        assert len(calls) == 2  # Readiness completes before the agent can execute.
        launch, probe = calls
        assert "--detach" in launch
        assert "--network=none" in launch
        assert "--read-only" in launch
        assert "/arm_julia_session.jl" in launch
        assert "--compiled-modules=existing" in launch
        assert "--heap-size-hint=200M" in launch
        assert launch[-2:] == [
            str(arm.pulse_designer).lower(),
            str(arm.unitful).lower(),
        ]
        assert probe[1] == command[2]  # Same container, not another Julia process.
        assert command[1] == "exec"
        assert metadata["julia_session"]["pid"] == 7
        assert metadata["julia_session"]["heap_size_hint_mb"] == 200
        assert metadata["julia_session"]["setup_elapsed_s"] >= 0
        assert (
            json.loads((control / "julia-session.json").read_text())
            == metadata["julia_session"]
        )
        assert not removed
    assert removed == [command[2]]
    assert (control / "julia-session.log").read_text() == "warmup log"


@pytest.mark.parametrize("failure", ["warmup", "logs"])
def test_failed_setup_or_log_collection_cannot_leak_container(
    monkeypatch, tmp_path, failure
):
    removed = []
    monkeypatch.setattr(arm_runtime.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(arm_runtime, "remove_container", removed.append)

    def docker_output(args, timeout):
        if args[0] == "exec":
            raise ValueError("warmup failed")
        return "container-id"

    def logs(*args, **kwargs):
        if failure == "logs":
            raise OSError("logs unavailable")
        return SimpleNamespace(returncode=1, stdout="", stderr="warmup failed")

    monkeypatch.setattr(arm_runtime, "docker_output", docker_output)
    monkeypatch.setattr(arm_runtime.subprocess, "run", logs)
    runtime = {
        "arm": JULIA_ARMS[0].model_dump(mode="json"),
        "author_image": "image",
        "simulator_image": None,
        "audit": {"ok": True},
    }
    with (
        pytest.raises((ValueError, OSError)),
        arm_runtime.launch_arm(runtime, ["pi"], tmp_path, tmp_path, {}),
    ):
        pytest.fail("agent started before warmup finished")
    assert len(removed) == 1


@pytest.mark.parametrize("mode", ["eval", "file"])
def test_client_preserves_code_paths_arguments_and_returns_output(
    monkeypatch, tmp_path, capsys, mode
):
    directory = tmp_path / "session"
    directory.mkdir()
    (directory / "ready.json").write_text('{"pid": 7}\n')
    monkeypatch.setattr(arm_julia_client, "DIRECTORY", directory)
    monkeypatch.chdir(tmp_path)
    code = 'println("unicode γ and $interpolation, quote \\" and newline\\n")'
    value = code if mode == "eval" else "a path with $ and γ.jl"
    observed, errors = [], []

    def respond():
        try:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                requests = list(directory.glob("request-*/request.toml"))
                if requests:
                    path = requests[0]
                    observed.append(tomllib.loads(path.read_text()))
                    (path.parent / "stdout").write_text("normal output γ\n")
                    (path.parent / "stderr").write_text("Julia error\n")
                    (path.parent / "done").write_text("1")
                    return
                time.sleep(0.01)
            raise AssertionError("client did not send a request")
        except (OSError, ValueError, AssertionError) as exc:
            errors.append(exc)

    thread = threading.Thread(target=respond)
    thread.start()
    arguments = (
        ["--timeout", "3", "-e", value]
        if mode == "eval"
        else ["--timeout", "3", value, "argument $γ", "--flag"]
    )
    assert arm_julia_client.main(arguments) == 1
    thread.join(timeout=4)
    assert not errors
    assert observed == [
        {
            "mode": mode,
            "value": value,
            "cwd": str(tmp_path),
            "args": [] if mode == "eval" else ["argument $γ", "--flag"],
        }
    ]
    output = capsys.readouterr()
    assert output.out == "normal output γ\n"
    assert output.err == "Julia error\n"
    assert not list(directory.glob("request-*"))


def test_client_ready_and_timeout(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(arm_julia_client, "DIRECTORY", tmp_path)
    assert arm_julia_client.main(["--ready", "--timeout", "0.01"]) == 1
    assert "did not respond" in capsys.readouterr().err
    (tmp_path / "ready.json").write_text('{"pid": 7}\n')
    assert arm_julia_client.main(["--ready"]) == 0
    assert json.loads(capsys.readouterr().out) == {"pid": 7}
