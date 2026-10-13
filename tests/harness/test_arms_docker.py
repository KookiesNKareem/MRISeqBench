"""Opt-in real container/Pi smoke test; synthetic upstream, no paid model API."""

import copy
import hashlib
import importlib.util
import json
import os
import shlex
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mriseqbench.catalog import Catalog
from mriseqbench.harness import arm_runtime
from mriseqbench.harness.proxy import ApiProxy
from mriseqbench.harness.runner import run
from mriseqbench.io import write_json
from mriseqbench.process import execute

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    not os.environ.get("MRISEQBENCH_DOCKER_AUTHOR_IMAGE"),
    reason="set MRISEQBENCH_DOCKER_AUTHOR_IMAGE to a built Python author image",
)


MATRIX = [a.id for a in Catalog(ROOT).documents["arms"].values()]
PROBE_CACHE = {}
CONTROLS = os.environ.get("MRISEQBENCH_DOCKER_CONTROLS", "rf_adc").split(",")
if set(CONTROLS) - {"rf_adc", "gradient_echo", "spin_echo"}:
    raise ValueError("unknown MRISeqBench sequence control")


def container_materialize(case, root, output, image, timeout_s=600):
    """Run the real phantom loader under the same container cap as the tests."""
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "case.json", case)
    name = "mriseqbench-materialize-" + uuid.uuid4().hex
    args, image = arm_runtime.container_args(image, name=name)
    args += arm_runtime.mount(root, root)
    args += arm_runtime.mount(output, output, readonly=False)
    args += [
        image,
        "julia",
        "--startup-file=no",
        "--compiled-modules=existing",
        *(
            [
                "--heap-size-hint="
                + os.environ["MRISEQBENCH_DOCKER_JULIA_HEAP_HINT_MB"]
                + "M"
            ]
            if os.environ.get("MRISEQBENCH_DOCKER_JULIA_HEAP_HINT_MB")
            else []
        ),
        "--project=/opt/simulator",
        str(root / "src/mriseqbench/phantoms/loader.jl"),
        str(output / "case.json"),
        str(output),
    ]
    try:
        execution = execute(args, root, timeout_s, output / "julia")
        assert execution["status"] == "completed", execution
    finally:
        arm_runtime.remove_container(name)
    files = ("phantom.phantom", "fields.h5", "manifest.json")
    for filename in files:
        assert (output / filename).is_file() and not (output / filename).is_symlink()
    write_json(
        output / "artifacts.json",
        {
            "phantom": case["object"]["phantom"],
            "physics": f"{case['physics']['id']}@{case['physics']['version']}",
            "seed": case["seed"],
            "files": {
                filename: hashlib.sha256((output / filename).read_bytes()).hexdigest()
                for filename in files
            },
        },
    )
    return json.loads((output / "manifest.json").read_text())


@pytest.mark.parametrize(
    "arm_id", MATRIX if os.environ.get("MRISEQBENCH_DOCKER_MATRIX") else ["a2_pypulseq"]
)
@pytest.mark.parametrize("control_kind", CONTROLS)
def test_real_offline_pi_authors_and_submits(
    tmp_path, monkeypatch, arm_id, control_kind
):
    memory = os.environ.get("MRISEQBENCH_DOCKER_MEMORY")
    if memory:
        original_args = arm_runtime.container_args

        def limited_args(*args, **kwargs):
            command, image = original_args(*args, **kwargs)
            command += [
                "--memory",
                memory,
                "--memory-swap",
                os.environ.get("MRISEQBENCH_DOCKER_MEMORY_SWAP", memory),
                "--cpus",
                "1",
            ]
            if os.environ.get("MRISEQBENCH_DOCKER_NODE_HEAP_MB"):
                command += [
                    "--env",
                    "NODE_OPTIONS=--max-old-space-size="
                    + os.environ["MRISEQBENCH_DOCKER_NODE_HEAP_MB"],
                ]
            return command, image

        monkeypatch.setattr(arm_runtime, "container_args", limited_args)
    if os.environ.get("MRISEQBENCH_DOCKER_REUSE_PROBES"):
        original_probe = arm_runtime.image_probe

        def cached_probe(image, command, timeout, mounts=()):
            # Only reuse successful audits of identical immutable images and
            # mounted bytes. Every trial still starts its own native session.
            key = (
                image,
                tuple(command),
                timeout,
                memory,
                os.environ.get("MRISEQBENCH_DOCKER_MEMORY_SWAP", memory),
                tuple(
                    (
                        str(source),
                        target,
                        hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                    )
                    for source, target in mounts
                ),
            )
            if key not in PROBE_CACHE:
                PROBE_CACHE[key] = original_probe(image, command, timeout, mounts)
            return PROBE_CACHE[key]

        monkeypatch.setattr(arm_runtime, "image_probe", cached_probe)
    materialize_image = os.environ.get("MRISEQBENCH_DOCKER_MATERIALIZE_IMAGE")
    if materialize_image:
        image = arm_runtime.pin_image(materialize_image)
        monkeypatch.setattr(
            "mriseqbench.harness.runner.materialize",
            lambda case, root, output: container_materialize(case, root, output, image),
        )
    catalog = Catalog(ROOT)
    arm = catalog.get("arms", arm_id)
    secret = tmp_path / "host-secret.txt"
    secret.write_text("host-only")
    host_key = "synthetic-host-key-must-not-enter-container"
    key_hash = hashlib.sha256(host_key.encode()).hexdigest()
    checks = f"""
import hashlib, importlib.util, json, os, pathlib, socket, urllib.error, urllib.request
assert not pathlib.Path({str(secret)!r}).exists()
assert not pathlib.Path('/var/run/docker.sock').exists()
assert pathlib.Path('bin/simulate').exists() == {arm.simulator != "none"!r}
assert {key_hash!r} not in {{hashlib.sha256(v.encode()).hexdigest() for v in os.environ.values()}}
for package in ('MRzeroCore', 'komamripy', 'juliacall', *{("pypulseq",) if arm.language != "python" else ()!r}):
    assert importlib.util.find_spec(package) is None
try:
    pathlib.Path('/root-write-probe').write_text('blocked')
except OSError:
    pass
else:
    raise AssertionError('root filesystem was writable')
try:
    socket.create_connection(('1.1.1.1', 443), timeout=2)
except OSError:
    pass
else:
    raise AssertionError('external network was available')
for route, body in (
    *{(("simulate/simulate", b"forged sequence"),) if arm.simulator == "none" else ()!r},
    ('simulate/exec', b'forged simulator code'),
    ('feedback/evaluate', b'forged judge request'),
    ('api/chat/completions', json.dumps({{'model':'openai/gpt-4o-mini',
        'tools':[{{'type':'web_search'}}]}}).encode()),
    *[('api/chat/completions', json.dumps({{'model':'openai/gpt-4o-mini',
         'tools':[{{'type':kind}}]}}).encode()) for kind in ('code_interpreter','mcp')],
    *[('api/chat/completions', json.dumps({{'model':'openai/gpt-4o-mini',
         key:[]}}).encode()) for key in ('plugins','web_search_options','mcp_servers')],
):
    req = urllib.request.Request('http://127.0.0.1:8765/' + route,
        data=body, headers={{'Content-Type':'application/json'}})
    try:
        urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as exc:
        assert exc.code == 403, exc.code
    else:
        raise AssertionError('forbidden request accepted: ' + route)
pathlib.Path('isolation-verified.json').write_text(json.dumps({{'ok':True}}))
"""
    if memory:
        checks += (
            "assert int(pathlib.Path('/sys/fs/cgroup/memory.max').read_text()) > 0\n"
        )
        if os.environ.get("MRISEQBENCH_DOCKER_MEMORY_SWAP", memory) == memory:
            checks += "assert pathlib.Path('/sys/fs/cgroup/memory.swap.max').read_text().strip() == '0'\n"
    fixture_name = (
        "arm_smoke_author.py" if control_kind == "rf_adc" else "arm_readout_author.py"
    )
    fixture = (ROOT / "tests/fixtures" / fixture_name).read_text()
    simulation = ""
    if arm.simulator != "none":
        simulation = (
            "pathlib.Path('inputs/phantom.phantom').write_bytes(b'tampered agent copy')\n"
            "subprocess.run(['./bin/simulate'], check=True)\n"
            "import h5py, numpy as np\n"
            "with h5py.File('simulation/signal.h5') as f:\n"
            "    signal=f['signal_real'][()]+1j*f['signal_imag'][()]\n"
            "    assert signal.size == 32 and np.isfinite(signal).all()\n"
            "    assert np.abs(signal).max() > 0\n"
        )
    script = (
        checks
        + "\nimport sys, subprocess\n"
        + "sys.argv=['fixture','--case','case.json','--output','sequence.seq']\n"
        + f"control_kind={control_kind!r}\n"
        + f"exec({fixture!r})\n"
        + simulation
        + "pathlib.Path('resource-usage.json').write_text(json.dumps({key: pathlib.Path('/sys/fs/cgroup/' + key).read_text().strip() for key in ('memory.max', 'memory.swap.max', 'memory.peak', 'memory.events')}))\n"
        + "subprocess.run(['./bin/submit'], check=True)\n"
    )
    if arm.language == "julia":
        session_checks = """
import subprocess, json, pathlib, time
def julia(code, check=True):
    result = subprocess.run(['./bin/julia', '-e', code], text=True, capture_output=True)
    if check:
        assert result.returncode == 0, (code, result.stdout, result.stderr)
    return result
first = julia('session_value = 41; println(getpid())')
second = julia('println(session_value + 1); println(getpid())')
assert second.stdout.splitlines() == ['42', first.stdout.strip()]
pathlib.Path('session-definitions.jl').write_text('session_function() = 1\\n')
assert julia('includet("session-definitions.jl"); println(session_function())').stdout.strip() == '1'
pathlib.Path('session-definitions.jl').write_text('session_function() = 2\\n')
for _ in range(20):
    updated = julia('println(session_function())')
    if updated.stdout.strip() == '2':
        break
    time.sleep(0.05)
assert updated.stdout.strip() == '2', updated
assert julia('error("expected recovery test")', check=False).returncode != 0
assert julia('println(session_value)').stdout.strip() == '41'
pathlib.Path('session-args.jl').write_text('println(join(ARGS, "|"))\\n')
assert subprocess.check_output(['./bin/julia', 'session-args.jl', 'a', '--b'], text=True).strip() == 'a|--b'
pathlib.Path('session-verified.json').write_text(json.dumps({'pid': int(first.stdout.strip()), 'revise': True, 'recovered': True}))
"""
        script = session_checks + script
    tool_command = shlex.join(["python3", "-c", script])
    observed, errors = [], []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                observed.append(body)
                assert self.headers["Authorization"] == "Bearer " + host_key
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                delta = {"role": "assistant", "content": "Done."}
                if len(observed) == 1:
                    delta = {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_smoke",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": json.dumps({"command": tool_command}),
                                },
                            }
                        ],
                    }
                for value, finish in (
                    (delta, None),
                    ({}, "tool_calls" if len(observed) == 1 else "stop"),
                ):
                    chunk = {
                        "id": "chatcmpl-local",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": body["model"],
                        "choices": [
                            {"index": 0, "delta": value, "finish_reason": finish}
                        ],
                    }
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception as exc:
                errors.append(exc)
                raise

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    def local_proxy(config, path, **kwargs):
        # Only this test factory substitutes a local HTTP upstream. Production
        # experiment validation continues to require HTTPS.
        local = config.model_copy(
            update={"upstream_url": f"http://127.0.0.1:{upstream.server_port}/v1"}
        )
        return ApiProxy(local, path, test_key=host_key, **kwargs)

    monkeypatch.setattr("mriseqbench.harness.runner.ApiProxy", local_proxy)
    try:
        experiment = copy.deepcopy(catalog.get("experiments", "pi_" + arm_id))
        experiment.suite = "smoke@1"
        experiment.agent.model = "openai/gpt-4o-mini"
        experiment.agent.timeout_s = 300
        if os.environ.get("MRISEQBENCH_DOCKER_SETUP_TIMEOUT_S"):
            experiment.arm_runtime.probe_timeout_s = int(
                os.environ["MRISEQBENCH_DOCKER_SETUP_TIMEOUT_S"]
            )
        if arm.language == "julia" and os.environ.get(
            "MRISEQBENCH_DOCKER_JULIA_HEAP_HINT_MB"
        ):
            experiment.arm_runtime.julia_heap_size_hint_mb = int(
                os.environ["MRISEQBENCH_DOCKER_JULIA_HEAP_HINT_MB"]
            )
        if arm.language == "python":
            experiment.arm_runtime.author_image = os.environ[
                "MRISEQBENCH_DOCKER_AUTHOR_IMAGE"
            ]
        elif arm.language == "julia" and os.environ.get(
            "MRISEQBENCH_DOCKER_JULIA_IMAGE_TAG"
        ):
            tag = os.environ["MRISEQBENCH_DOCKER_JULIA_IMAGE_TAG"]
            experiment.arm_runtime.author_image = (
                experiment.arm_runtime.author_image.rsplit(":", 1)[0] + ":" + tag
            )
        if arm.simulator != "none" and os.environ.get(
            "MRISEQBENCH_DOCKER_SIMULATOR_IMAGE"
        ):
            experiment.arm_runtime.simulator_image = os.environ[
                "MRISEQBENCH_DOCKER_SIMULATOR_IMAGE"
            ]
        output = Path(os.environ.get("MRISEQBENCH_DOCKER_OUTPUT", tmp_path / "run"))
        if os.environ.get("MRISEQBENCH_DOCKER_MATRIX"):
            output = (
                output / arm_id
                if control_kind == "rf_adc"
                else output / control_kind / arm_id
            )
        _, reports = run(catalog, experiment, output)
        (output / "test-settings.json").write_text(
            json.dumps(
                {
                    "control": control_kind,
                    "memory": memory,
                    "memory_swap": os.environ.get(
                        "MRISEQBENCH_DOCKER_MEMORY_SWAP", memory
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        assert not errors
        report = reports[0]
        workspace = Path(report["workspace"])
        stderr = (workspace / "agent.stderr.log").read_text()
        assert report["status"] == "smoke_passed", (report, stderr)
        assert report["agent"]["status"] == "submitted"
        if memory:
            resources = json.loads(
                (workspace / "agent/resource-usage.json").read_text()
            )
            events = dict(
                line.split() for line in resources["memory.events"].splitlines()
            )
            assert int(events["oom_kill"]) == 0, resources
        assert report["isolation"]["internet"] is False
        assert report["web_access"] is False
        if arm.language == "julia":
            session = report["isolation"]["julia_session"]
            assert session["setup_elapsed_s"] >= 0
            assert session["revise_version"]
            verified = json.loads(
                (workspace / "agent/session-verified.json").read_text()
            )
            assert verified["pid"] == session["pid"]
            assert verified["revise"] and verified["recovered"]
        assert len(report["simulation_calls"]) == (arm.simulator != "none")
        if arm.simulator != "none":
            receipt = json.loads((workspace / "control/receipt.json").read_text())
            assert report["simulation_calls"][0]["sha256"] == receipt["sha256"]
        assert (workspace / "agent/isolation-verified.json").is_file()
        assert (workspace / "control/receipt.json").is_file()
        assert observed, "Pi did not reach the synthetic model through the mailbox"
        if control_kind != "rf_adc":
            spec = importlib.util.spec_from_file_location(
                "readout_checks", ROOT / "tests/fixtures/arm_readout_checks.py"
            )
            checks = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(checks)
            metrics = checks.validate_readout(
                workspace / "control/submissions/attempt-001/sequence.seq", control_kind
            )
            (workspace / "control/readout-validation.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
    finally:
        upstream.shutdown()
        upstream.server_close()


@pytest.mark.skipif(
    not os.environ.get("MRISEQBENCH_DOCKER_MRZERO_IMAGE"),
    reason="set MRISEQBENCH_DOCKER_MRZERO_IMAGE to a built MRzero service image",
)
def test_real_mrzero_service_simulates_frozen_public_inputs(tmp_path):
    fixture = (ROOT / "tests/fixtures/make_smoke_submission.py").read_text()
    script = (
        "import sys, subprocess, h5py, numpy as np\n"
        "sys.argv=['fixture','--case','case.json','--output','sequence.seq']\n"
        f"exec({fixture!r})\n"
        "from pathlib import Path\n"
        "Path('inputs/phantom.phantom').write_bytes(b'tampered agent copy')\n"
        "subprocess.run(['./bin/simulate'], check=True)\n"
        "with h5py.File('simulation/signal.h5') as f:\n"
        "    signal=f['signal_real'][()]+1j*f['signal_imag'][()]\n"
        "    assert signal.size == 32 and np.isfinite(signal).all()\n"
        "    assert np.abs(signal).max() > 0\n"
        "subprocess.run(['./bin/submit'], check=True)\n"
    )
    catalog = Catalog(ROOT)
    experiment = copy.deepcopy(catalog.get("experiments", "pi_s2_pypulseq_mrzero"))
    experiment.suite = "smoke@1"
    experiment.agent.adapter = "command"
    experiment.agent.command = ["python3", "-c", script]
    experiment.agent.timeout_s = 180
    experiment.api_proxy = None
    experiment.arm_runtime.author_image = os.environ["MRISEQBENCH_DOCKER_AUTHOR_IMAGE"]
    experiment.arm_runtime.simulator_image = os.environ[
        "MRISEQBENCH_DOCKER_MRZERO_IMAGE"
    ]
    destination = os.environ.get("MRISEQBENCH_DOCKER_OUTPUT")
    output = Path(destination + "-mrzero") if destination else tmp_path / "run"
    _, reports = run(catalog, experiment, output)
    report = reports[0]
    workspace = Path(report["workspace"])
    stderr = (workspace / "agent.stderr.log").read_text()
    assert report["status"] == "smoke_passed", (report, stderr)
    assert report["simulator"] == "mrzero"
    assert len(report["simulation_calls"]) == 1
    assert (workspace / "control/simulations.json").is_file()
    assert (workspace / "control/receipt.json").is_file()
    receipt = json.loads((workspace / "control/receipt.json").read_text())
    assert report["simulation_calls"][0]["sha256"] == receipt["sha256"]
    assert (
        workspace / "agent/inputs/phantom.phantom"
    ).read_bytes() == b"tampered agent copy"
