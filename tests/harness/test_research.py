import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mriseqbench.catalog import Catalog, load_yaml
from mriseqbench.harness.prepare import prepare
from mriseqbench.harness.sandbox import environment, launch, seatbelt_profile
from mriseqbench.models import Agent, SandboxConfig, Task
from mriseqbench.process import execute

ROOT = Path(__file__).resolve().parents[2]


def test_research_workspace_has_only_high_level_sequence_brief(tmp_path):
    catalog = Catalog(ROOT)
    case = catalog.resolve(
        Task.model_validate(
            load_yaml(
                ROOT / "tests/fixtures/legacy_research_tasks/t1_mapping_research.yaml"
            )
        ),
        catalog.get("physics", "ideal"),
    )
    prepare(
        tmp_path,
        case,
        catalog.get("experiments", "smoke"),
        SimpleNamespace(root=ROOT, port=12345, token="test"),
    )
    text = (tmp_path / "PROMPT.md").read_text()
    assert "normal tools" in text
    assert "bin/web" not in text
    for filename in ("case.json", "task.yaml"):
        data = yaml.safe_load((tmp_path / filename).read_text())
        assert set(data["sequence"]) == {"matrix", "fov_mm", "duration_s"}
        assert "identity_requirements" not in data
        assert "capabilities" not in data["sequence"]
        assert data["tissue_properties"] == case["object"]["tissue_properties"]
        assert set(data["tissue_properties"]) == {"fat", "myocardium", "blood"}
        assert set(data["tissue_properties"]["fat"]) == {
            "proton_density",
            "T1_s",
            "T2_s",
            "chemical_shift_hz",
        }
        assert data["tissue_properties"]["fat"]["chemical_shift_hz"] < 0
    assert not (tmp_path / "bin/web").exists()
    assert case["task"]["identity_requirements"]  # Trusted case is not mutated.


def test_native_internet_policy_preserves_filesystem_rules(tmp_path):
    restricted = seatbelt_profile(tmp_path)
    research = seatbelt_profile(tmp_path, internet=True)
    assert "(deny network-outbound)" in restricted
    assert "(deny network-outbound)" not in research
    assert "(deny file-read-data)" in research
    assert "(deny file-write*)" in research


@pytest.mark.skipif(sys.platform != "darwin", reason="native macOS network probe")
def test_native_research_network_keeps_private_files_hidden(tmp_path):
    box, control = tmp_path / "agent", tmp_path / "control"
    box.mkdir()
    control.mkdir()
    secret = control / "secret"
    secret.write_text("synthetic-private-data")
    # DNS is tested without relying on a remote site's availability or TLS stack.
    script = """import json,pathlib,socket,sys
out={}
try: socket.getaddrinfo('example.com',443);out['dns']=True
except OSError: out['dns']=False
try: pathlib.Path(sys.argv[1]).read_text();out['private_read']='allowed'
except PermissionError: out['private_read']='blocked'
pathlib.Path('probe.json').write_text(json.dumps(out))
"""
    agent = Agent(command=[sys.executable, "-c", script, str(secret)])
    with launch(
        SandboxConfig(backend="macos"), agent.command, ROOT, box, control, internet=True
    ) as (command, metadata):
        result = execute(
            command, box, 15, control / "probe", env=environment(agent, ROOT, box)
        )
    assert result["status"] == "completed"
    assert metadata["internet"] is True
    assert json.loads((box / "probe.json").read_text()) == {
        "dns": True,
        "private_read": "blocked",
    }


@pytest.mark.parametrize(
    "task_id,enabled", [("gre_t1w", False), ("gre_t1w_research", True)]
)
def test_runner_enables_native_network_only_for_research(
    monkeypatch, tmp_path, task_id, enabled
):
    from contextlib import contextmanager

    from mriseqbench.harness.runner import run_one

    catalog = Catalog(ROOT)
    case = catalog.resolve(
        (
            Task.model_validate(
                load_yaml(ROOT / f"tests/fixtures/legacy_research_tasks/{task_id}.yaml")
            )
            if task_id.endswith("_research")
            else catalog.get("tasks", task_id)
        ),
        catalog.get("physics", "ideal"),
    )
    observed = []

    @contextmanager
    def fake_launch(
        config, command, root, workspace, control_dir, ports=(), *, internet=False
    ):
        observed.append(internet)
        yield command, {"backend": "test", "isolated": True, "internet": internet}

    monkeypatch.setattr("mriseqbench.harness.runner.launch", fake_launch)
    monkeypatch.setattr(
        "mriseqbench.harness.runner.execute",
        lambda *args, **kwargs: {"status": "completed", "returncode": 0},
    )
    report = run_one(
        catalog, catalog.get("experiments", "smoke"), case, tmp_path / task_id, 0
    )
    assert observed == [enabled]
    assert report["web_access"] is enabled
    assert report["task_family"] == "gre_t1w"


def test_research_pi_uses_normal_online_mode(tmp_path):
    from mriseqbench.harness.prepare import prepare_pi

    env = prepare_pi(tmp_path, "http://127.0.0.1:12345/v1", offline=False)
    assert "PI_OFFLINE" not in env
    assert "PI_CODING_AGENT_DIR" in env
