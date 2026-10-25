"""Julia MCP arm plumbing that runs without Docker, Julia or the mcp package."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from mriseqbench.catalog import Catalog, load_yaml
from mriseqbench.harness import arm_mcp, arm_runtime
from mriseqbench.harness.arm_mcp_bridge import ALLOWED, truncate, validate, warmup_code
from mriseqbench.harness.prepare import prepare
from mriseqbench.models import Arm, Experiment

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "src/mriseqbench/harness"


def test_bridge_fixes_project_and_bounds_arguments():
    assert validate({"name": "julia_eval", "arguments": {"code": "1+1"}}) == (
        "julia_eval",
        {"code": "1+1", "timeout": 0, "env_path": "/opt/author"},
    )
    assert validate({"name": "julia_restart"}) == (
        "julia_restart",
        {"env_path": "/opt/author"},
    )
    assert validate({"name": "julia_list_sessions", "arguments": {}})[1] == {}
    for row in (
        {"name": "bash", "arguments": {}},
        {"name": "julia_eval", "arguments": {"code": "1", "env_path": "/"}},
        {"name": "julia_eval", "arguments": {"code": "1", "julia_cmd": "sh"}},
        {"name": "julia_eval", "arguments": {"code": "1", "timeout": 1201}},
        {"name": "julia_eval", "arguments": {"code": "1", "timeout": True}},
        {"name": "julia_eval", "arguments": {"code": 1}},
        {"name": "julia_eval", "arguments": {"code": "x" * (1024**2 + 1)}},
        {"name": "julia_restart", "arguments": {"env_path": "/tmp"}},
    ):
        with pytest.raises(ValueError):
            validate(row)


def test_bridge_truncates_long_output():
    response = truncate({"content": [{"type": "text", "text": "x" * 300_000}]})
    assert response["content"][0]["text"].endswith("[Tool output truncated]")
    assert len(response["content"][0]["text"]) < 270_000


@pytest.mark.parametrize("pulse_designer", [False, True])
def test_warmup_reuses_cli_session_without_its_mailbox(pulse_designer):
    source = (HARNESS / "arm_julia_session.jl").read_text()
    code = warmup_code(source, Path("/w"), pulse_designer)
    assert "MRISeqBenchJuliaSession.serve()" not in code
    assert f"warmup({str(pulse_designer).lower()},false)" in code
    assert 'cd("/w")' in code and "MRISEQBENCH_MCP_WARM_READY" in code


def test_tool_surface_is_consistent():
    extension = (HARNESS / "arm_mcp_extension.ts").read_text()
    tools = json.loads(re.search(r"const TOOLS = (\[.*?\]);", extension)[1])
    assert tools == arm_mcp.TOOLS and ALLOWED < set(tools)
    assert "bash" not in tools
    assert arm_mcp.pi_command(["pi", "--mode", "json"]) == [
        "pi",
        "--no-extensions",
        "--extension",
        "/arm_mcp_extension.ts",
        "--tools",
        ",".join(arm_mcp.TOOLS),
        "--mode",
        "json",
    ]
    vendored = ROOT / "vendor/julia-mcp"
    assert "MIT License" in (vendored / "LICENSE").read_text()
    assert "async def julia_eval(" in (vendored / "server.py").read_text()


def test_mcp_interface_requires_julia_without_unitful():
    base = {"schema_version": 1, "id": "x", "version": 1, "tool_interface": "julia_mcp"}
    Arm.model_validate({**base, "language": "julia", "authoring": "koma_base"})
    with pytest.raises(ValueError):
        Arm.model_validate({**base, "language": "python", "authoring": "pypulseq"})
    with pytest.raises(ValueError):
        Arm.model_validate(
            {
                **base,
                "language": "julia",
                "authoring": "pulseq_julia",
                "pulse_designer": True,
                "unitful": True,
            }
        )


@pytest.mark.parametrize(
    ("arm_id", "simulator_text"),
    [
        ("j6_koma_base_julia_mcp", "Agent-side simulation is disabled."),
        ("j7_pulseq_julia_mcp_koma", "Simulator: koma, through `./bin/simulate"),
    ],
)
def test_prompt_and_workspace_for_mcp_arms(tmp_path, arm_id, simulator_text):
    catalog = Catalog(ROOT)
    arm = catalog.get("arms", f"{arm_id}@1")
    experiment = Experiment.model_validate(
        load_yaml(ROOT / f"experiments/pi_{arm_id}.yaml")
    )
    case = catalog.resolve(catalog.get("tasks", "pulseq_smoke"))
    server = SimpleNamespace(root=ROOT, token="t", port=1)
    prompt_path, _, _ = prepare(tmp_path, case, experiment, server, arm=arm)
    text = prompt_path.read_text()
    assert "Execute all Julia code through the `julia_eval` MCP tool" in text
    assert "./bin/julia script.jl" not in text and simulator_text in text
    assert "julia_eval` MCP session" in (tmp_path / "ARM_API.md").read_text()
    assert "julia_eval" in (tmp_path / "bin/julia").read_text()


def test_launch_dispatches_to_warm_mcp_container(tmp_path, monkeypatch):
    calls = []

    def docker_output(args, timeout=120):
        calls.append(args)
        if "--ready" in args:
            return json.dumps(
                {"mode": "persistent_mcp", "pulse_designer": True, "simulator": "koma"}
            )
        return "container"

    monkeypatch.setattr(arm_runtime, "docker_output", docker_output)
    monkeypatch.setattr(arm_runtime, "ensure_removed", calls.append)
    monkeypatch.setattr(
        arm_runtime,
        "container_args",
        lambda image, name: (["docker", "run", "--rm"], image),
    )
    monkeypatch.setattr(
        arm_mcp.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="log", stderr=""),
    )
    arm = Catalog(ROOT).get("arms", "j7_pulseq_julia_mcp_koma@1")
    runtime = {
        "arm": arm.model_dump(mode="json"),
        "author_image": "sha256:" + "0" * 64,
        "author_memory_mb": 2048,
        "setup_timeout_s": 5,
        "julia_heap_size_hint_mb": None,
    }
    (tmp_path / "control/transport").mkdir(parents=True)
    with arm_runtime.launch_arm(
        runtime, ["pi", "-p", "x"], tmp_path, tmp_path / "control", {"A": "1"}
    ) as (command, metadata):
        assert command[:5] == [
            "docker",
            "exec",
            command[2],
            "python3",
            "/mriseqbench-gateway.py",
        ]
        assert json.loads(command[5])[:2] == ["pi", "--no-extensions"]
        assert metadata["active_tools"] == arm_mcp.TOOLS
    serve = calls[0]
    assert "--rm" not in serve and "--pulse-designer" in serve
    assert "MRISEQBENCH_AGENT_SIMULATOR=koma" in serve and "A=1" in serve
    assert serve[serve.index("--heap") + 1] == "100"
    assert (
        json.loads((tmp_path / "control/julia-mcp-session.json").read_text())[
            "upstream_commit"
        ]
        == arm_mcp.UPSTREAM_COMMIT
    )
    assert calls[-1] == command[2]  # Container removed.


class FakeJulia:
    """A julia-mcp session whose state is lost by restarts or crashes."""

    def __init__(self, warm):
        self.warm, self.calls = warm, []

    async def call_tool(self, name, arguments):
        code = arguments.get("code", "")
        self.calls.append("warmup" if "WARM_READY" in code else name)
        if name == "julia_restart":
            self.warm = False
        elif "MRISEQBENCH_MCP_WARM_READY" in code:
            self.warm = True
        output = (
            "MRISEQBENCH_MCP_WARM_READY"
            if "WARM_READY" in code
            else ("WARM" if self.warm else "COLD")
            if "isdefined(Main" in code
            else "ok"
        )
        return SimpleNamespace(
            isError=False, content=[SimpleNamespace(type="text", text=output)]
        )


@pytest.mark.parametrize(
    ("warm", "name", "rewarmed", "calls"),
    [
        (True, "julia_eval", None, ["julia_eval", "julia_eval"]),
        (False, "julia_eval", "lost", ["julia_eval", "warmup", "julia_eval"]),
        (True, "julia_restart", "restart", ["julia_restart", "warmup"]),
        (False, "julia_list_sessions", None, ["julia_list_sessions"]),
    ],
)
def test_lost_sessions_are_rewarmed_in_the_workspace(warm, name, rewarmed, calls):
    import asyncio

    from mriseqbench.harness.arm_mcp_bridge import dispatch

    session = FakeJulia(warm)
    code = warmup_code("module M end\nMRISeqBenchJuliaSession.serve()\n", "/w", True)
    _, flag = asyncio.run(dispatch(session, name, {}, "/w", code))
    assert flag == rewarmed and session.calls == calls
    assert session.warm or name == "julia_list_sessions"
