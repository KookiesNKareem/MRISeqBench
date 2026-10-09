import copy
import json
import shlex
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from mriseqbench.catalog import Catalog
from mriseqbench.harness.feedback import FeedbackServer
from mriseqbench.harness.prepare import prepare, prompt
from mriseqbench.harness.proxy import ApiProxy
from mriseqbench.harness.runner import run
from mriseqbench.models import ApiProxyConfig, FeedbackConfig, SandboxConfig

ROOT = Path(__file__).resolve().parents[2]


def test_feedback_client_reports_failed_checks(tmp_path, monkeypatch):
    def worker(command, cwd, *args, **kwargs):
        (Path(cwd) / "result.json").write_text(json.dumps({"status": "failed"}))
        return {"status": "completed"}

    monkeypatch.setattr("mriseqbench.harness.feedback.execute", worker)
    catalog = Catalog(ROOT)
    case = catalog.cases(catalog.get("suites", "smoke"))[0]
    experiment = catalog.get("experiments", "sandbox_smoke")
    workspace = tmp_path / "agent"
    with FeedbackServer(
        case, FeedbackConfig(mode="preflight"), ROOT, tmp_path / "control"
    ) as server:
        prepare(workspace, case, experiment, server)
        (workspace / "sequence.seq").write_bytes(b"invalid submission")
        result = subprocess.run(
            [str(workspace / "bin/lint")], capture_output=True, text=True, check=False
        )
        assert result.returncode == 1
        assert json.loads(result.stdout)["status"] == "failed"


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("pi") is None,
    reason="requires installed Pi and macOS Seatbelt",
)
def test_real_pi_receives_task_and_submits_without_paid_api(tmp_path, monkeypatch):
    observed = []
    errors = []
    script = (
        "import json,pathlib,subprocess,sys,h5py,yaml; "
        "c=json.loads(pathlib.Path('case.json').read_text()); "
        "t=yaml.safe_load(pathlib.Path('task.yaml').read_text()); "
        "assert t['objective']==c['task']['objective']; "
        "assert c['case_id']=='pulseq_smoke-v1--ideal-v1'; "
        "assert c['task']['objective'] in pathlib.Path('PROMPT.md').read_text(); "
        "subprocess.run([sys.executable,sys.argv[1],'--case','case.json',"
        "'--output','sequence.seq'],check=True); "
        "subprocess.run(['./bin/submit'],check=True)"
    )
    command = shlex.join(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT / "tests/fixtures/make_smoke_submission.py"),
        ]
    )

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                observed.append(body)
                assert self.path == "/v1/chat/completions"
                assert self.headers["Authorization"] == "Bearer local-synthetic-key"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                delta = {"role": "assistant", "content": "Done."}
                if len(observed) == 1:
                    delta = {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_probe",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": json.dumps({"command": command}),
                                },
                            }
                        ],
                    }
                for value, finish in (
                    (delta, None),
                    ({}, "tool_calls" if len(observed) == 1 else "stop"),
                ):
                    chunk = {
                        "id": "chatcmpl-probe",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": body["model"],
                        "choices": [
                            {"index": 0, "delta": value, "finish_reason": finish}
                        ],
                    }
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception as exc:
                errors.append(exc)
                raise

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    monkeypatch.setattr(
        "mriseqbench.harness.runner.ApiProxy",
        lambda config, path, **kwargs: ApiProxy(
            config, path, test_key="local-synthetic-key", **kwargs
        ),
    )
    try:
        catalog = Catalog(ROOT)
        experiment = copy.deepcopy(catalog.get("experiments", "sandbox_smoke"))
        experiment.agent.adapter = "pi"
        experiment.agent.model = "openai/gpt-4o-mini"
        experiment.agent.command = [
            shutil.which("pi"),
            "--provider",
            "openrouter",
            "--model",
            "{model}",
            "--thinking",
            "off",
            "--mode",
            "json",
            "--session-dir",
            "{workspace}/transcript",
            "-p",
            "{prompt_text}",
        ]
        experiment.agent.timeout_s = 60
        experiment.agent.sandbox = SandboxConfig(
            backend="macos",
            read_paths=[
                str(Path(shutil.which("pi")).parent.parent),
                "{root}/tests/fixtures/make_smoke_submission.py",
            ],
        )
        experiment.submission_mode = "submit"
        experiment.api_proxy = ApiProxyConfig.model_construct(
            upstream_url=f"http://127.0.0.1:{upstream.server_port}/v1"
        )
        _, reports = run(catalog, experiment, tmp_path / "run")
        assert not errors
        report = reports[0]
        assert report["status"] == "smoke_passed", report
        assert report["agent"]["status"] == "submitted"
        workspace = Path(report["workspace"])
        prompt_text = (workspace / "agent/PROMPT.md").read_text()
        delivered_text = []
        for message in observed[0]["messages"]:
            content = message.get("content", "")
            delivered_text.append(
                content
                if isinstance(content, str)
                else "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            )
        assert any(prompt_text.strip() in text for text in delivered_text)
        assert (workspace / "control/receipt.json").is_file()
        assert any((workspace / "agent/transcript").iterdir())
    finally:
        upstream.shutdown()
        upstream.server_close()


def test_proxy_rejects_other_models_and_fallback_routing(tmp_path):
    import urllib.error
    import urllib.request

    config = ApiProxyConfig(upstream_url="https://unused.invalid/v1")
    with ApiProxy(
        config, tmp_path / "api.jsonl", model="selected/model", test_key="synthetic-key"
    ) as proxy:
        for payload in (
            {"model": "other/model"},
            {"model": "selected/model", "models": ["other/model"]},
            {},
        ):
            request = urllib.request.Request(
                proxy.base_url + "/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {proxy.token}"},
            )
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(request, timeout=5)
            assert exc.value.code == 403
        assert proxy.requests == 0


def test_prompt_uses_resolved_contract():
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "gre_t1w"), catalog.get("physics", "b0_smooth")
    )
    text = prompt(case)
    assert case["task"]["objective"] in text
    requirements = yaml.safe_load(text.split("```yaml\n", 1)[1].split("```", 1)[0])
    assert requirements["physics"]["effects"] == case["physics"]["effects"]
    assert requirements["sequence"]["duration_s"] == 6
    assert requirements["sequence"]["te_ms"] == {"target": 5, "tolerance": 1}
    assert "null" not in text
    assert "schema_version" not in text
    assert "physical scoring is pending" in text


def test_hardware_provenance_stays_out_of_agent_context(tmp_path):
    from types import SimpleNamespace

    catalog = Catalog(ROOT)
    case = catalog.resolve(
        catalog.get("tasks", "gre_t1w"),
        catalog.get("physics", "ideal"),
        hardware_ref="low_field@1",
    )
    original = copy.deepcopy(case)
    prepare(
        tmp_path,
        case,
        catalog.get("experiments", "smoke"),
        SimpleNamespace(root=ROOT, port=12345, token="test-token"),
    )
    for filename in ("PROMPT.md", "task.yaml", "case.json"):
        text = (tmp_path / filename).read_text()
        assert "provenance" not in text
        assert "source_url" not in text
        assert "Siemens" not in text
    public = json.loads((tmp_path / "case.json").read_text())
    assert public["hardware"] == {
        k: v for k, v in case["hardware"].items() if k != "provenance"
    }
    assert case == original
