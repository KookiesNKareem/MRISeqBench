"""Julia MCP arms: Julia runs only through julia_eval (pinned aplavin/julia-mcp).

One persistent, pre-warmed Julia session per trial lives in the author
container behind `arm_mcp_bridge.py`; Pi reaches it through the mounted
`arm_mcp_extension.ts`, which also restricts the agent to file tools.
"""

import json
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from ..io import write_json

TOOLS = ["read", "write", "edit", "julia_eval", "julia_restart", "julia_list_sessions"]
UPSTREAM_COMMIT = "06e83b1569a736f16a3c58f4b39e44fc33ad28bb"  # vendor/julia-mcp
MOUNTS = {
    "arm_gateway.py": "/mriseqbench-gateway.py",
    "arm_julia_session.jl": "/arm_julia_session.jl",
    "arm_mcp_bridge.py": "/arm_mcp_bridge.py",
    "arm_mcp_extension.ts": "/arm_mcp_extension.ts",
}
DISABLED_JULIA = (
    '#!/bin/sh\necho "Use the julia_eval MCP tool in this arm" >&2\nexit 2\n'
)


def instructions(arm):
    """PROMPT.md text replacing the `./bin/julia` CLI instructions."""
    if arm.simulator == "none":
        return (
            "Execute all Julia code through the `julia_eval` MCP tool, provided by the pinned aplavin/julia-mcp server.\n"
            "One persistent Julia session belongs to this trial only. KomaMRIBase, KomaMRIFiles and Revise are preloaded.\n"
            "The same fixed generic authoring/write-read/Revise warmup finishes before your time budget starts; subsequent code compilation/execution counts.\n"
            "Use `julia_eval` with `code` and an optional timeout in seconds. The default 0 leaves the harness in charge of the 1200-second charged-time deadline, allowing excluded native grading to finish. Set a positive timeout only for short authoring calls. State persists; use println/display to see output.\n"
            'Read/write/edit tools operate on files. For a script, call `include("design.jl")` through julia_eval.\n'
            'For edits, use `includet("design.jl")` then edit function definitions and call your function again; Revise runs before every evaluation.\n'
            "Use Julia's `run` through MCP for benchmark controls, e.g. `run(`./bin/status`)` and `run(`./bin/submit`)`.\n"
            "The same lint/check/submit/feedback interfaces and three-submission limit apply. Foreground grading remains excluded from charged time.\n"
            "No Bash tool or direct Julia CLI tool is available in this arm. `julia_list_sessions` inspects your session.\n"
            "`julia_restart` clears your own state; use only as a last resort. Its restart/loading/compilation time counts toward the same remaining budget.\n"
            "No Julia process or variable state is shared with another trial. "
        )
    return (
        "Execute all Julia code through the `julia_eval` MCP tool and the pinned aplavin/julia-mcp server.\n"
        "One persistent Julia/PulseDesigner session belongs to this trial. KomaMRIBase, KomaMRIFiles and Revise are preloaded; PulseDesigner is enabled and Unitful is unavailable.\n"
        "Import PulseDesigner helpers explicitly, as documented in ARM_API.md. Generic authoring/write-read/Revise warmup completes before agent timing.\n"
        "Use `julia_eval` with code and optional timeout in seconds; default 0 leaves the harness in charge of the original 1200-second charged deadline.\n"
        'Use include("design.jl") or includet("design.jl") through MCP. State persists and Revise observes edits before each evaluation.\n'
        "Run controls through Julia: run(`./bin/status`), run(`./bin/lint`), run(`./bin/check`), run(`./bin/submit`), run(`./bin/feedback`).\n"
        "Use run(`./bin/simulate sequence.seq`) for the external KomaMRI simulator. Simulation time counts against the same charged agent allowance; waiting for a free simulator slot does not.\n"
        "The external simulator returns public diagnostics only. It exposes no judge scores, hidden reference targets or evaluator code.\n"
        "No Bash tool or direct Julia CLI bypass is available. julia_restart affects only this trial and its time counts; julia_list_sessions inspects the session.\n"
    )


def pi_command(command):
    """Load only the MCP extension and the allowed tools into Pi."""
    command = list(command)
    if Path(command[0]).name == "pi":
        command[1:1] = [
            "--no-extensions",
            "--extension",
            "/arm_mcp_extension.ts",
            "--tools",
            ",".join(TOOLS),
        ]
    return command


@contextmanager
def launch_mcp(runtime, command, workspace, control, env):
    from . import arm_runtime as base

    arm = runtime["arm"]
    name = "mriseqbench-mcp-" + uuid.uuid4().hex
    args, image = base.container_args(runtime["author_image"], name=name)
    args = base.with_memory(args, runtime["author_memory_mb"])
    args.remove("--rm")  # Keep a failed server for diagnostics until removal.
    args += base.mount(workspace, workspace, readonly=False)
    args += base.mount(control / "transport", "/run/mriseqbench", readonly=False)
    for filename, target in MOUNTS.items():
        args += base.mount(Path(__file__).with_name(filename).resolve(), target)
    args += ["--workdir", str(workspace)]
    for key, value in {**env, "MRISEQBENCH_AGENT_SIMULATOR": arm["simulator"]}.items():
        args += ["--env", f"{key}={value}"]
    try:
        started = time.monotonic()
        timeout = runtime["setup_timeout_s"]
        base.docker_output(
            args[1:]
            + [
                "--detach",
                image,
                "/opt/mcp-venv/bin/python",
                "/arm_mcp_bridge.py",
                "--serve",
                "--workspace",
                str(workspace),
                "--heap",
                str(runtime["julia_heap_size_hint_mb"] or 100),
                "--agent-simulator",
                arm["simulator"],
                *(["--pulse-designer"] if arm["pulse_designer"] else []),
            ],
            timeout=30,
        )
        ready = json.loads(
            base.docker_output(
                [
                    "exec",
                    name,
                    "/opt/mcp-venv/bin/python",
                    "/arm_mcp_bridge.py",
                    "--ready",
                    "--timeout",
                    str(timeout),
                ],
                timeout=timeout + 10,
            )
        )
        if (
            ready.get("mode") != "persistent_mcp"
            or ready.get("pulse_designer") != arm["pulse_designer"]
            or ready.get("simulator") != arm["simulator"]
        ):
            raise ValueError(f"Julia MCP session is not the selected arm: {ready}")
        ready.update(
            setup_elapsed_total_s=time.monotonic() - started,
            upstream_commit=UPSTREAM_COMMIT,
        )
        write_json(control / "julia-mcp-session.json", ready)
        yield (
            [
                args[0],
                "exec",
                name,
                "python3",
                "/mriseqbench-gateway.py",
                json.dumps(pi_command(command)),
            ],
            {
                "backend": "docker",
                "isolated": True,
                "internet": False,
                "container": name,
                "persistent_container": True,
                "julia_session": ready,
                "tool_interface": "aplavin/julia-mcp stdio through pinned Pi extension",
                "active_tools": TOOLS,
                "upstream_commit": UPSTREAM_COMMIT,
            },
        )
    finally:
        try:
            for label, query in (
                ("container-state", ["inspect", "--format", "{{json .State}}", name]),
                ("server-log", ["logs", "--tail=100", name]),
            ):
                result = subprocess.run(
                    [args[0], *query],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                (control / f"julia-mcp-{label}.log").write_text(
                    result.stdout + result.stderr
                )
        except (OSError, subprocess.SubprocessError):
            pass  # Diagnostics are best effort; removal below is not.
        finally:
            base.ensure_removed(name)
