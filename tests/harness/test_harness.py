import copy
import hashlib
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.harness.feedback import FeedbackServer
from mriseqbench.harness.prepare import prepare_inputs, prepare_pi
from mriseqbench.harness.proxy import ApiProxy
from mriseqbench.harness.runner import run
from mriseqbench.harness.sandbox import environment, launch
from mriseqbench.models import Agent, ApiProxyConfig, FeedbackConfig, SandboxConfig
from mriseqbench.process import execute, expand

ROOT = Path(__file__).resolve().parents[2]


def smoke_case():
    catalog = Catalog(ROOT)
    return catalog.cases(catalog.get("suites", "smoke"))[0]


def test_environment_never_inherits_host_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "host-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "another-secret")
    agent = Agent(
        command=[sys.executable], environment={"OPENROUTER_API_KEY": "{api_token}"}
    )
    env = environment(agent, ROOT, tmp_path, api_token="dummy-run-token")
    assert env["OPENROUTER_API_KEY"] == "dummy-run-token"
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert env["HOME"] == str(tmp_path.resolve())


def test_template_values_are_not_reexpanded(tmp_path):
    assert expand(
        ["{prompt_text}"],
        ROOT,
        tmp_path,
        prompt_text="literal {api_token}",
        api_token="dummy",
    ) == ["literal {api_token}"]


def test_no_silent_sandbox_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "unsupported-platform")
    with (
        pytest.raises(ValueError, match="isolation was not bypassed"),
        launch(
            SandboxConfig(),
            [sys.executable, "-c", "pass"],
            ROOT,
            tmp_path / "agent",
            tmp_path / "control",
        ),
    ):
        pytest.fail("must not launch")


def test_broad_read_grant_rejected(tmp_path):
    with (
        pytest.raises(ValueError, match="too broad"),
        launch(
            SandboxConfig(read_paths=[str(Path.home())]),
            [sys.executable],
            ROOT,
            tmp_path / "agent",
            tmp_path / "control",
        ),
    ):
        pytest.fail("must not launch")


def test_feedback_budget_refund_and_one_shot(monkeypatch, tmp_path):
    def worker(command, cwd, timeout, log_prefix, **kwargs):
        (Path(cwd) / "result.json").write_text(
            json.dumps({"status": "preflight_passed", "checks": [], "metrics": {}})
        )
        return {"status": "completed", "returncode": 0}

    monkeypatch.setattr("mriseqbench.harness.feedback.execute", worker)
    config = FeedbackConfig(mode="preflight", max_checks=1)
    with FeedbackServer(
        smoke_case(), config, ROOT, tmp_path, max_submissions=1
    ) as server:
        assert server.handle("check", "wrong", b"candidate")[0] == 403
        assert server.handle("lint", server.token, b"candidate")[0] == 200
        assert server.checks_used == 0
        assert server.handle("check", server.token, b"candidate")[0] == 200
        assert server.checks_used == 1
        assert (
            server.handle("check", server.token, b"candidate")[1]["status"]
            == "budget_exhausted"
        )
        assert (
            server.handle("submit", server.token, b"final candidate")[1]["status"]
            == "submitted"
        )
        assert server.handle("submit", server.token, b"replacement")[0] == 403
        assert server.submission_path.read_bytes() == b"final candidate"
        assert (tmp_path / "receipt.json").exists()


def test_infrastructure_failure_refunds_check(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "mriseqbench.harness.feedback.execute", lambda *a, **kw: {"status": "timeout"}
    )
    with FeedbackServer(
        smoke_case(),
        FeedbackConfig(mode="preflight", max_checks=1),
        ROOT,
        tmp_path,
        max_submissions=1,
    ) as server:
        _, result = server.handle("check", server.token, b"candidate")
        assert result["check_refunded"] is True
        assert server.checks_used == 0


def test_blind_feedback_and_body_limits(tmp_path):
    with FeedbackServer(
        smoke_case(),
        FeedbackConfig(max_submission_bytes=10),
        ROOT,
        tmp_path,
        max_submissions=1,
    ) as server:
        assert (
            server.handle("lint", server.token, b"abc")[1]["status"]
            == "feedback_disabled"
        )
        assert server.handle("submit", server.token, b"x" * 11)[0] == 400
        assert not server.submitted.is_set()
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.port}/submit",
            data=b"accepted",
            headers={"X-MRISeqBench-Token": server.token},
        )
        with urllib.request.urlopen(request) as response:
            assert json.loads(response.read())["status"] == "submitted"


def test_proxy_hides_key_streams_and_revokes(tmp_path):
    key = "synthetic-test-key"
    release = threading.Event()
    observed = []

    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self):
            observed.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: first\n\n")
            self.wfile.flush()
            release.wait(5)
            self.wfile.write(b"data: second\n\n")

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    # Production schemas require HTTPS; only this local test uses HTTP.
    config = ApiProxyConfig.model_construct(
        upstream_url=f"http://127.0.0.1:{upstream.server_port}/api/v1"
    )
    try:
        with ApiProxy(config, tmp_path / "api.jsonl", test_key=key) as proxy:
            request = urllib.request.Request(
                proxy.base_url + "/models",
                headers={"Authorization": f"Bearer {proxy.token}"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.read(len(b"data: first\n\n")) == b"data: first\n\n"
                assert (
                    not release.is_set()
                )  # first SSE chunk arrived before the upstream finished
                release.set()
                assert response.read() == b"data: second\n\n"
            assert observed == [f"Bearer {key}"]
            bad = urllib.request.Request(
                proxy.base_url + "/unapproved",
                headers={"Authorization": f"Bearer {proxy.token}"},
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(bad)
            assert error.value.code == 403
            proxy.revoke()
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            assert error.value.code == 401
        assert key not in (tmp_path / "api.jsonl").read_text()
    finally:
        release.set()
        upstream.shutdown()
        upstream.server_close()
        thread.join(2)


@pytest.mark.skipif(sys.platform != "darwin", reason="native macOS isolation probe")
def test_live_macos_blocks_private_reads_writes_and_other_ports(tmp_path, monkeypatch):
    box, control = tmp_path / "agent", tmp_path / "control"
    box.mkdir()
    control.mkdir()
    secret = control / "hidden.txt"
    secret.write_text("private")
    allowed = tmp_path.parent / f"public-{tmp_path.name}.txt"
    allowed.write_text("public")
    sockets = []
    import socket

    for _ in range(2):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        sockets.append(listener)
    script = """import json,os,pathlib,socket
out = {}
for key,path in [('secret',os.environ['SECRET_PATH']),('allowed',os.environ['ALLOWED_PATH'])]:
    try: out[key]=pathlib.Path(path).read_text()
    except PermissionError: out[key]='blocked'
try: pathlib.Path(os.environ['OUTSIDE_PATH']).write_text('escape');out['outside_write']='allowed'
except PermissionError: out['outside_write']='blocked'
for key,port in [('gateway',os.environ['GOOD_PORT']),('other',os.environ['BAD_PORT'])]:
    try:
        with socket.create_connection(('127.0.0.1',int(port)),timeout=.2): out[key]='allowed'
    except OSError: out[key]='blocked'
out['inherited_secret']=os.environ.get('HOST_SECRET','absent')
try:
    with socket.socket() as listener: listener.bind(('127.0.0.1',0)); out['listener']='allowed'
except PermissionError: out['listener']='blocked'
pathlib.Path('probe.json').write_text(json.dumps(out))
"""
    monkeypatch.setenv("HOST_SECRET", "do-not-inherit")
    agent = Agent(
        command=[sys.executable, "-c", script],
        environment={
            "SECRET_PATH": str(secret),
            "ALLOWED_PATH": str(allowed),
            "OUTSIDE_PATH": str(control / "escape.txt"),
            "GOOD_PORT": str(sockets[0].getsockname()[1]),
            "BAD_PORT": str(sockets[1].getsockname()[1]),
        },
    )
    config = SandboxConfig(backend="macos", read_paths=[str(allowed)])
    try:
        with launch(
            config, agent.command, ROOT, box, control, [sockets[0].getsockname()[1]]
        ) as (command, metadata):
            result = execute(
                command, box, 10, control / "probe", env=environment(agent, ROOT, box)
            )
        assert result["status"] == "completed", (
            control / "probe.stderr.log"
        ).read_text()
        assert metadata["isolated"]
        assert json.loads((box / "probe.json").read_text()) == {
            "secret": "blocked",
            "allowed": "public",
            "outside_write": "blocked",
            "gateway": "allowed",
            "other": "blocked",
            "inherited_secret": "absent",
            "listener": "blocked",
        }
    finally:
        for listener in sockets:
            listener.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="native macOS runner probe")
def test_submit_ends_sandboxed_run_and_grades_accepted_bytes(tmp_path):
    catalog = Catalog(ROOT)
    experiment = copy.deepcopy(catalog.get("experiments", "smoke"))
    experiment.agent.sandbox = SandboxConfig(
        backend="macos", read_paths=["{root}/tests/fixtures/make_smoke_submission.py"]
    )
    experiment.submission_mode = "submit"
    experiment.feedback = FeedbackConfig(mode="preflight", max_checks=1, timeout_s=30)
    experiment.max_submissions = 1  # Legacy budgeted-feedback mode.
    # Build a real Pulseq file, use real HTTP feedback, submit, then try to keep working.
    script = """import subprocess,sys,time
subprocess.run([sys.executable,sys.argv[1],'--case','case.json','--output','sequence.seq'],check=True)
subprocess.run(['./bin/lint'],check=True)
subprocess.run(['./bin/check'],check=True)
subprocess.run(['./bin/submit'])
time.sleep(60)
"""
    experiment.agent.command = [
        sys.executable,
        "-c",
        script,
        "{root}/tests/fixtures/make_smoke_submission.py",
    ]
    experiment.agent.timeout_s = 45
    output, results = run(catalog, experiment, tmp_path / "run")
    result = results[0]
    assert result["status"] == "smoke_passed"
    assert result["agent"]["status"] == "submitted"
    assert result["feedback"]["checks_used"] == 1
    workspace = Path(result["workspace"])
    assert (workspace / "evaluation/sequence.seq").read_bytes() == (
        workspace / "control/accepted.seq"
    ).read_bytes()
    assert (output / "results.jsonl").exists()


def test_submit_mode_does_not_accept_leftover_file(tmp_path):
    catalog = Catalog(ROOT)
    experiment = copy.deepcopy(catalog.get("experiments", "smoke"))
    experiment.submission_mode = "submit"
    experiment.agent.command = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('sequence.seq').write_text('leftover')",
    ]
    _, results = run(catalog, experiment, tmp_path / "run")
    assert results[0]["status"] == "agent_not_submitted"


def test_public_input_assets_are_copied_and_verified(tmp_path):
    root, workspace = tmp_path / "catalog", tmp_path / "agent"
    root.mkdir()
    workspace.mkdir()
    source = root / "labels.nii.gz"
    source.write_bytes(b"synthetic-asset-bytes")
    asset = {
        "path": source.name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "format": "nifti",
        "shape": [2, 2],
    }
    mapping = prepare_inputs(workspace, {"object": {"asset": asset}}, root)
    copy_path = workspace / mapping[source.name]
    assert copy_path.name.endswith(".nii.gz")
    assert copy_path.read_bytes() == source.read_bytes()
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset changed"):
        prepare_inputs(workspace, {"object": {"asset": asset}}, root)


def test_pi_adapter_config_uses_proxy_only(tmp_path):
    env = prepare_pi(tmp_path, "http://127.0.0.1:12345/v1")
    assert env["PI_OFFLINE"] == "1"
    models = json.loads((tmp_path / ".pi-agent/models.json").read_text())
    assert models["providers"]["openrouter"]["baseUrl"] == "http://127.0.0.1:12345/v1"
    assert not (tmp_path / ".pi-agent/auth.json").exists()
