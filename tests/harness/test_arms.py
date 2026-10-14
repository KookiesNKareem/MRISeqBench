import base64
import copy
import importlib.util
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mriseqbench.catalog import Catalog
from mriseqbench.cli import main
from mriseqbench.harness import arm_gateway, arm_runtime
from mriseqbench.harness.arm_mailbox import Mailbox, read_request
from mriseqbench.harness.prepare import prepare
from mriseqbench.harness.runner import run, run_one
from mriseqbench.models import Arm, Experiment

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "sha256:" + "a" * 64


def configured(arm):
    catalog = Catalog(ROOT)
    experiment = copy.deepcopy(catalog.get("experiments", "pi_" + arm.id))
    experiment.agent.adapter = "command"
    experiment.agent.command = ["python3", "-c", "pass"]
    experiment.api_proxy = None
    return experiment


def runtime(arm):
    return {
        "arm": arm.model_dump(mode="json"),
        "author_image": IMAGE,
        "simulator_image": IMAGE if arm.simulator != "none" else None,
        "audit": {"ok": True},
    }


def test_nine_distinct_arms_and_matched_authoring_pairs():
    arms = list(Catalog(ROOT).documents["arms"].values())
    assert len(arms) == 9
    assert {a.language for a in arms} == {"python", "julia"}
    bases = [a for a in arms if a.simulator == "none"]
    assert len(bases) == 4

    def author(a):
        return a.language, a.authoring, a.pulse_designer, a.unitful

    assert len({author(a) for a in bases}) == 4
    for arm in arms:
        assert any(author(arm) == author(base) for base in bases)
        assert configured(arm).arm == f"{arm.id}@{arm.version}"


@pytest.mark.parametrize(
    "mutation",
    [
        {"language": "python", "authoring": "koma_base"},
        {"language": "julia", "authoring": "koma_base", "pulse_designer": True},
        {"language": "python", "authoring": "pypulseq", "unitful": True},
        {"language": "python", "authoring": "pypulseq", "simulator": "jemris"},
        {"language": "python", "authoring": "pypulseq", "matlab_runtime": "octave"},
        {"language": "matlab", "authoring": "pulseq_matlab"},
        {"language": "matlab", "authoring": "pulseq_matlab", "simulator": "jemris"},
    ],
)
def test_impossible_arm_combinations_rejected(mutation):
    with pytest.raises(ValidationError):
        Arm(schema_version=1, id="invalid", version=1, **mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        {"arm_runtime": None},
        {"agent": {"command": ["true"], "sandbox": {"backend": "none"}}},
        {"agent": {"command": ["true"], "sandbox": {"read_paths": ["/usr"]}}},
        {"agent": {"command": ["true"], "sandbox": {"network_ports": [443]}}},
        {
            "agent": {
                "command": ["true"],
                "environment": {"PYTHONPATH": "/host/packages"},
            }
        },
        {"agent": {"command": ["true"], "environment": {"JULIA_LOAD_PATH": "/host"}}},
    ],
)
def test_runtime_bypasses_rejected(mutation):
    arm = Catalog(ROOT).get("arms", "a2_pypulseq")
    data = configured(arm).model_dump(mode="json")
    data.update(mutation)
    with pytest.raises(ValidationError):
        Experiment.model_validate(data)


def test_mutated_experiment_revalidated_before_creating_run(tmp_path):
    arm = Catalog(ROOT).get("arms", "a2_pypulseq")
    experiment = configured(arm)
    experiment.agent.sandbox.backend = "none"
    with pytest.raises(ValidationError):
        run(Catalog(ROOT), experiment, tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "arm", list(Catalog(ROOT).documents["arms"].values()), ids=lambda a: a.id
)
def test_every_arm_prompt_reports_actual_access_and_never_claims_internet(
    arm, tmp_path
):
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "gre_t1w"),
        catalog.get("physics", "static_b0"),
    )
    prepare(
        tmp_path,
        case,
        configured(arm),
        SimpleNamespace(root=ROOT, port=1234, token="test"),
        arm=arm,
    )
    prompt = (tmp_path / "PROMPT.md").read_text()
    assert "Internet access is enabled" not in prompt
    assert "no direct Internet access" in prompt
    assert f"Arm: {arm.id}@1" in prompt
    assert (tmp_path / "bin/simulate").exists() == (arm.simulator != "none")
    if arm.language != "python":
        assert "PyPulseq are installed" not in prompt
    assert (
        json.loads((tmp_path / ".harness.json").read_text())["route_prefix"]
        == "/feedback"
    )
    assert "python3" in (tmp_path / "bin/submit").read_text()
    assert (tmp_path / "bin/julia").exists() == (arm.language == "julia")
    if arm.language == "julia":
        assert "persistent Julia session" in prompt
        assert "includet" in prompt
        assert "before your time budget starts" in prompt


def test_no_simulator_image_allowed_for_authoring_only(monkeypatch):
    arm = Catalog(ROOT).get("arms", "a2_pypulseq")
    experiment = configured(arm)
    experiment.arm_runtime.simulator_image = "unapproved"
    with pytest.raises(ValueError, match="exactly"):
        arm_runtime.resolve_runtime(experiment, arm)


def test_missing_simulator_image_fails_closed():
    arm = Catalog(ROOT).get("arms", "s3_pypulseq_komamripy")
    experiment = configured(arm)
    experiment.arm_runtime.simulator_image = None
    with pytest.raises(ValueError, match="exactly"):
        arm_runtime.resolve_runtime(experiment, arm)


def test_bad_author_audit_and_wrong_simulator_fail_closed(monkeypatch):
    monkeypatch.setattr(arm_runtime, "remove_container", lambda name: None)
    monkeypatch.setattr(arm_runtime, "pin_image", lambda image: IMAGE)
    arm = Catalog(ROOT).get("arms", "s3_pypulseq_komamripy")
    experiment = configured(arm)
    monkeypatch.setattr(
        arm_runtime,
        "docker_output",
        lambda *a: json.dumps(
            {"ok": False, "errors": ["PulseDesigner source present"]}
        ),
    )
    with pytest.raises(ValueError, match="violates"):
        arm_runtime.resolve_runtime(experiment, arm)
    monkeypatch.setattr(arm_runtime, "probe_author", lambda *a: {"ok": True})
    monkeypatch.setattr(
        arm_runtime,
        "docker_output",
        lambda *a: json.dumps(
            {"engine": "mrzero", "pulseq_input": True, "self_test": True}
        ),
    )
    with pytest.raises(ValueError, match="selected engine"):
        arm_runtime.resolve_runtime(experiment, arm)


def test_docker_launch_exposes_only_workspace_mailbox_and_trusted_gateway(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(arm_runtime.subprocess, "run", lambda *a, **kw: calls.append(a))
    arm = Catalog(ROOT).get("arms", "a2_pypulseq")
    control = tmp_path / "control"
    with arm_runtime.launch_arm(
        runtime(arm),
        ["pi"],
        tmp_path / "agent",
        control,
        {"HOME": str(tmp_path / "agent")},
    ) as (command, metadata):
        assert "--network=none" in command
        assert "--read-only" in command
        assert "--cap-drop=ALL" in command
        assert "--security-opt=no-new-privileges" in command
        assert "--privileged" not in command
        assert "/var/run/docker.sock" not in str(command)
        assert str(ROOT / ".venv") not in str(command)
        assert metadata["internet"] is False
        mounts = [
            command[i + 1] for i, value in enumerate(command) if value == "--mount"
        ]
        assert len(mounts) == 3
        assert not any(f"source={control}," in value for value in mounts)
    assert calls and "rm" in calls[-1][0]


def test_unknown_docker_image_id_rejected(monkeypatch):
    monkeypatch.setattr(arm_runtime, "docker_output", lambda *a: "mutable-tag")
    with pytest.raises(ValueError, match="immutable"):
        arm_runtime.pin_image("example:latest")


@pytest.mark.parametrize(
    "arm", list(Catalog(ROOT).documents["arms"].values()), ids=lambda a: a.id
)
def test_runner_enforces_every_arm_and_keeps_judge_separate(monkeypatch, tmp_path, arm):
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "gre_t1w"),
        catalog.get("physics", "static_b0"),
    )
    experiment = configured(arm)
    observed = []
    process_env = []

    @contextmanager
    def fake_launch(state, command, workspace, control, env):
        observed.append((state, env))
        yield command, {"backend": "docker", "isolated": True, "internet": False}

    monkeypatch.setattr("mriseqbench.harness.runner.launch_arm", fake_launch)
    monkeypatch.setattr(
        "mriseqbench.harness.runner.execute",
        lambda *a, **kw: (
            process_env.append(kw["env"]) or {"status": "completed", "returncode": 0}
        ),
    )
    report = run_one(
        catalog, experiment, case, tmp_path / arm.id, 0, runtime=runtime(arm)
    )
    assert report["arm"] == experiment.arm
    assert report["simulator"] == arm.simulator
    assert report["web_access"] is False
    assert report["prompt_mode"] == "detailed"
    assert observed[0][1]["PIP_NO_INDEX"] == "1"
    assert process_env == [None]  # The host Docker client retains its daemon context.
    assert str(ROOT / ".venv") not in observed[0][1]["PATH"]
    assert report["simulation_calls"] == []


def exchange(directory, request):
    name = os.urandom(16).hex()
    path = directory / (name + ".request")
    path.write_text(json.dumps(request))
    response = path.with_suffix(".response")
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if response.exists():
            lines = response.read_text().splitlines()
            if lines and json.loads(lines[-1]).get("done"):
                frames = [json.loads(line) for line in lines]
                data = b"".join(
                    base64.b64decode(f["data"]) for f in frames if "data" in f
                )
                return frames[0]["status"], json.loads(data)
        time.sleep(0.01)
    pytest.fail("mailbox did not reply")


@pytest.mark.parametrize(
    "payload",
    [
        {"target": "simulate", "path": "/simulate", "method": "POST"},
        {"target": "api", "path": "https://unapproved.invalid", "method": "GET"},
        {"target": "feedback", "path": "/evaluate", "method": "POST"},
        {"target": "host", "path": "/read-file", "method": "GET"},
    ],
)
def test_forged_requests_cannot_enable_simulation_or_unapproved_routes(
    tmp_path, payload
):
    with Mailbox(tmp_path, SimpleNamespace()):
        status, result = exchange(tmp_path, payload)
    assert status == 403
    assert "error" in result


def test_mailbox_cannot_read_symlinks(tmp_path):
    secret = tmp_path / "private"
    secret.write_text('{"secret": true}')
    link = tmp_path / "request"
    link.symlink_to(secret)
    with pytest.raises(OSError):
        read_request(link)


def test_mailbox_routes_only_exact_frozen_sequence_bytes(tmp_path):
    calls = []
    body = b"actual-sequence-bytes"

    def simulator(value):
        calls.append(value)
        return 200, {"engine": "koma", "artifacts": {}}

    with Mailbox(tmp_path, SimpleNamespace(), simulator=simulator):
        status, result = exchange(
            tmp_path,
            {
                "target": "simulate",
                "path": "/simulate",
                "method": "POST",
                "body": base64.b64encode(body).decode(),
            },
        )
    assert status == 200 and result["engine"] == "koma"
    assert calls == [body]


def test_failed_simulator_diagnostics_survive_temporary_files(monkeypatch, tmp_path):
    control = tmp_path / "control"
    (control / "simulation_inputs").mkdir(parents=True)
    monkeypatch.setattr(arm_runtime, "remove_container", lambda name: None)

    def fail(command, cwd, timeout, log):
        Path(str(log) + ".stderr.log").write_text(
            "x" * 20000 + "converter rejects changed gradient"
        )
        Path(str(log) + ".stdout.log").write_text("native converter diagnostic")
        return {"status": "error", "returncode": 1}

    monkeypatch.setattr(arm_runtime, "execute", fail)
    service = arm_runtime.Simulator(
        runtime(Catalog(ROOT).get("arms", "s2_pypulseq_mrzero")),
        SimpleNamespace(simulation_timeout_s=10),
        tmp_path / "agent",
        control,
        {},
    )
    status, result = service(b"sequence bytes")
    assert status == 422
    assert "changed gradient" in result["stderr"]
    assert len(result["stderr"]) == 4000
    saved = json.loads((control / "simulations.json").read_text())
    assert saved[0]["stderr"] == result["stderr"]
    assert saved[0]["stdout_tail"] == "native converter diagnostic"
    assert not list(control.glob("simulation-*"))


def test_simulation_client_displays_native_http_error(monkeypatch, tmp_path):
    from mriseqbench.harness import arm_simulation_client

    path = tmp_path / "sequence.seq"
    path.write_bytes(b"test")
    monkeypatch.setattr(arm_simulation_client.sys, "argv", ["simulate", str(path)])
    body = json.dumps(
        {"error": "simulator execution failed", "stderr": "unsupported gradient raster"}
    ).encode()

    def reject(*args, **kwargs):
        raise urllib.error.HTTPError(
            "http://localhost", 422, "failed", {}, io.BytesIO(body)
        )

    monkeypatch.setattr(arm_simulation_client.urllib.request, "urlopen", reject)
    with pytest.raises(ValueError, match="unsupported gradient raster"):
        arm_simulation_client.main()


def test_gateway_feedback_round_trip_without_external_network(monkeypatch, tmp_path):
    monkeypatch.setattr(arm_gateway, "DIRECTORY", tmp_path)
    feedback = SimpleNamespace(
        handle=lambda kind, token, body: (
            200,
            {
                "status": "lint_passed",
                "correct": kind == "lint"
                and token == "secret"
                and body == b"candidate",
            },
        )
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), arm_gateway.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with Mailbox(tmp_path, feedback):
            req = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/feedback/lint",
                data=b"candidate",
                headers={"X-MRISeqBench-Token": "secret"},
            )
            with urllib.request.urlopen(req, timeout=3) as response:
                assert json.load(response)["correct"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_strip_pulsedesigner_removes_includable_sources_and_exports(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "sanitize", ROOT / "runtimes/arms/sanitize_julia.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "src/sequences").mkdir(parents=True)
    (tmp_path / "src/sequences/PulseDesigner.jl").write_text(
        "module PulseDesigner; end"
    )
    (tmp_path / "src/KomaMRIBase.jl").write_text(
        'module KomaMRIBase\ninclude("sequences/PulseDesigner.jl")\nexport PulseDesigner\nend'
    )
    module.sanitize(tmp_path, False)
    assert "PulseDesigner" not in (tmp_path / "src/KomaMRIBase.jl").read_text()
    assert not (tmp_path / "src/sequences").exists()


@pytest.mark.parametrize("enabled", [False, True])
def test_julia_precompile_workloads_cannot_reintroduce_authoring_tools(
    tmp_path, enabled
):
    spec = importlib.util.spec_from_file_location(
        "sanitize", ROOT / "runtimes/arms/sanitize_julia.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("KomaMRIBase", "KomaMRIFiles"):
        src = tmp_path / name / "src"
        src.mkdir(parents=True)
        (src / (name + ".jl")).write_text(
            'include("precompile.jl")\n'
            + (
                'include("sequences/PulseDesigner.jl")\nexport PulseDesigner\n'
                if name == "KomaMRIBase"
                else ""
            )
        )
        (src / "precompile.jl").write_text("using KomaMRIBase.PulseDesigner")
    sequences = tmp_path / "KomaMRIBase/src/sequences"
    sequences.mkdir()
    (sequences / "PulseDesigner.jl").write_text("module PulseDesigner; end")
    module.sanitize(tmp_path / "KomaMRIBase", enabled)
    for name in ("KomaMRIBase", "KomaMRIFiles"):
        src = tmp_path / name / "src"
        assert not (src / "precompile.jl").exists()
        assert "precompile.jl" not in (src / (name + ".jl")).read_text()
    assert sequences.exists() == enabled


def test_dependency_cleanup_keeps_real_packages_and_removes_test_only_unitful(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "sanitize", ROOT / "runtimes/arms/sanitize_julia.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    test = tmp_path / "packages/StaticArrays/hash/test"
    test.mkdir(parents=True)
    (test / "unitful.jl").write_text("using Unitful")
    actual = tmp_path / "packages/Unitful/hash/src/Unitful.jl"
    actual.parent.mkdir(parents=True)
    actual.write_text("module Unitful; end")
    module.sanitize_depot(tmp_path)
    assert not test.exists()
    assert actual.exists()  # Cleanup cannot hide a genuinely installed tool.


def test_cli_lists_all_arms(capsys):
    assert main(["--root", str(ROOT), "arms"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 9


def test_model_stream_reaches_agent_before_completion(monkeypatch, tmp_path):
    from http.server import BaseHTTPRequestHandler

    from mriseqbench.harness.proxy import ApiProxy
    from mriseqbench.models import ApiProxyConfig

    release = threading.Event()
    seen_keys = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["model"] == "test/model"
            seen_keys.append(self.headers["Authorization"])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: first\n\n")
            self.wfile.flush()
            release.wait(5)
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    monkeypatch.setattr(arm_gateway, "DIRECTORY", tmp_path / "transport")
    server = ThreadingHTTPServer(("127.0.0.1", 0), arm_gateway.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    config = ApiProxyConfig.model_construct(
        upstream_url=f"http://127.0.0.1:{upstream.server_port}/v1"
    )
    try:
        with (
            ApiProxy(
                config,
                tmp_path / "api.jsonl",
                model="test/model",
                test_key="host-synthetic-key",
            ) as proxy,
            Mailbox(tmp_path / "transport", SimpleNamespace(), proxy),
        ):
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/api/chat/completions",
                data=json.dumps({"model": "test/model"}).encode(),
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                assert response.read(len(b"data: first\n\n")) == b"data: first\n\n"
                assert not release.is_set()
                release.set()
                assert response.read() == b"data: [DONE]\n\n"
        assert seen_keys == ["Bearer host-synthetic-key"]
        assert "host-synthetic-key" not in (tmp_path / "api.jsonl").read_text()
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        upstream.shutdown()
        upstream.server_close()


def test_author_audit_rejects_hidden_pulsedesigner_sources(monkeypatch):
    from mriseqbench.harness import arm_audit

    __import__("h5py")  # Load native imports before replacing the subprocess probe.
    arm = Catalog(ROOT).get("arms", "a3_koma_base")
    monkeypatch.setattr(
        arm_audit.importlib.util,
        "find_spec",
        lambda name: None if name == "pypulseq" else object(),
    )
    monkeypatch.setattr(arm_audit.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        arm_audit.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(
        arm_audit.os,
        "walk",
        lambda path: (
            [("/opt/hidden", [], ["PulseDesigner.jl"])] if path == "/opt" else []
        ),
    )
    result = arm_audit.audit(arm.model_dump(mode="json"))
    assert not result["ok"]
    assert any("PulseDesigner.jl" in error for error in result["errors"])


def test_cli_doctor_uses_arm_runtime_preflight(monkeypatch, capsys):
    observed = []

    def resolve(experiment, arm):
        observed.append((experiment, arm))
        return runtime(arm)

    monkeypatch.setattr(arm_runtime, "resolve_runtime", resolve)
    assert (
        main(
            [
                "--root",
                str(ROOT),
                "doctor",
                "--arm",
                "a2_pypulseq",
                "--author-image",
                "author:local",
            ]
        )
        == 0
    )
    assert observed[0][0].arm_runtime.author_image == "author:local"
    assert observed[0][1].authoring == "pypulseq"
    assert json.loads(capsys.readouterr().out)["author_image"] == IMAGE


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "test/model", "tools": [{"type": "code_interpreter"}]},
        {"model": "test/model", "tools": [{"type": "web_search"}]},
        {
            "model": "test/model",
            "tools": [{"type": "mcp", "server_url": "https://unapproved.invalid"}],
        },
        {"model": "test/model", "plugins": [{"id": "web"}]},
    ],
)
def test_remote_provider_tools_cannot_escape_arm_policy(tmp_path, payload):
    with Mailbox(tmp_path, SimpleNamespace(), proxy=SimpleNamespace()):
        status, result = exchange(
            tmp_path,
            {
                "target": "api",
                "path": "/chat/completions",
                "method": "POST",
                "body": base64.b64encode(json.dumps(payload).encode()).decode(),
            },
        )
    assert status == 403
    assert "remote built-in tools" in result["error"]
