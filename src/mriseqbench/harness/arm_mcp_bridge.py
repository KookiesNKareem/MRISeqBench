"""Container-local bridge to the pinned, unmodified aplavin/julia-mcp stdio server.

`--serve` starts one warmed author session per trial and listens on a Unix
socket; each tool call (`arm_mcp_bridge.py '<json>'`) is validated, forwarded
and traced to `mcp-tool-trace.jsonl`. No credentials or evaluator paths are
supplied; the agent can select neither the project nor the Julia executable.
"""

import argparse
import asyncio
import json
import os
import socket
import sys
import time
from pathlib import Path

SOCKET = "/tmp/mriseqbench-julia-mcp.sock"
READY = Path("/tmp/mriseqbench-julia-mcp-ready.json")
ALLOWED = {"julia_eval", "julia_restart", "julia_list_sessions"}
PROJECT = "/opt/author"
MAX_CODE = 1024**2
MAX_TEXT = 256 * 1024


def validate(row):
    """Map an agent tool call onto fixed server arguments."""
    name, args = row.get("name"), row.get("arguments", {})
    if name not in ALLOWED:
        raise ValueError("Unknown MCP tool")
    if not isinstance(args, dict):
        raise TypeError("Arguments must be an object")
    permitted = {"code", "timeout"} if name == "julia_eval" else set()
    if set(args) - permitted:
        raise ValueError(
            "Only the fixed author project and Julia executable are available"
        )
    if name == "julia_eval":
        code = args.get("code")
        if not isinstance(code, str) or len(code.encode()) > MAX_CODE:
            raise ValueError("Invalid code payload")
        timeout = args.get("timeout", 0)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 <= timeout <= 1200
        ):
            raise ValueError(
                "Timeout must be in [0,1200]; zero uses the harness charged-time deadline"
            )
        return name, {"code": code, "timeout": timeout, "env_path": PROJECT}
    return name, ({"env_path": PROJECT} if name == "julia_restart" else {})


def truncate(response):
    for item in response.get("content", []):
        if item.get("type") == "text" and len(item["text"]) > MAX_TEXT:
            item["text"] = item["text"][:MAX_TEXT] + "\n[Tool output truncated]"
    return response


def warmup_code(session_source, workspace, pulse_designer):
    """The CLI arms' generic warmup, loaded without their mailbox service."""
    ending = "MRISeqBenchJuliaSession.serve()\n"
    if not session_source.endswith(ending):
        raise ValueError("unexpected Julia session source")
    return (
        session_source[: -len(ending)]
        + "\nmkpath(MRISeqBenchJuliaSession.directory); using Revise,KomaMRIBase,KomaMRIFiles; "
        + f"MRISeqBenchJuliaSession.warmup({str(pulse_designer).lower()},false); "
        + "MRISeqBenchJuliaSession.warm_main(); MRISeqBenchJuliaSession.release_memory(); "
        + f"cd({json.dumps(str(workspace))}); "
        + 'println("MRISEQBENCH_MCP_WARM_READY"); println(getpid())\n'
    )


def text(result):
    return "\n".join(
        item.text for item in result.content if getattr(item, "type", None) == "text"
    )


REWARM_NOTES = {
    "restart": "Note: the Julia session was restarted at your request; the workspace "
    "directory and authoring packages were restored (this time counts as agent time).",
    "lost": "Note: the previous Julia session had ended (for example a crash, "
    "out-of-memory, exit() or a timeout) and all session variables were lost; a new "
    "session was started with the workspace directory and authoring packages restored "
    "before running this call (this time counts as agent time).",
}


async def warm(session, code):
    result = await session.call_tool(
        "julia_eval", arguments={"code": code, "env_path": PROJECT, "timeout": 600}
    )
    output = text(result)
    if result.isError or "MRISEQBENCH_MCP_WARM_READY" not in output:
        raise RuntimeError("MCP warmup failed: " + output[-3000:])
    return output


async def is_warm(session, workspace):
    probe = (
        "print(isdefined(Main, :MRISeqBenchJuliaSession) && "
        f'pwd() == {json.dumps(str(workspace))} ? "WARM" : "COLD")'
    )
    result = await session.call_tool(
        "julia_eval", arguments={"code": probe, "env_path": PROJECT, "timeout": 60}
    )
    return not result.isError and "WARM" in text(result) and "COLD" not in text(result)


async def dispatch(session, name, args, workspace, code):
    """Call one tool; a restarted or killed session loses the workspace and
    authoring imports, so restore them first (charged to the agent)."""
    if name == "julia_list_sessions":
        return await session.call_tool(name, arguments=args), None
    if name == "julia_restart":
        result = await session.call_tool(name, arguments=args)
        await warm(session, code)
        return result, "restart"
    rewarmed = None if await is_warm(session, workspace) else "lost"
    if rewarmed:
        await warm(session, code)
    return await session.call_tool(name, arguments=args), rewarmed


async def serve(workspace, heap, pulse_designer, agent_simulator):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server = StdioServerParameters(
        command="/opt/mcp-venv/bin/python",
        args=[
            "/opt/julia-mcp/server.py",
            "--threads=1",
            "--startup-file=no",
            "--compiled-modules=existing",
            f"--heap-size-hint={heap}M",
        ],
        env=dict(os.environ),
    )
    # Opened once at startup, before the event loop serves any tool call.
    with open("/tmp/mriseqbench-julia-mcp-stderr.log", "a") as errors:  # noqa: ASYNC230
        async with (
            stdio_client(server, errlog=errors) as (reader, writer),
            ClientSession(reader, writer) as session,
        ):
            identity = await session.initialize()
            listing = await session.list_tools()
            if {tool.name for tool in listing.tools} != ALLOWED:
                raise RuntimeError("unexpected julia-mcp tool surface")
            code = warmup_code(
                Path("/arm_julia_session.jl").read_text(), workspace, pulse_designer
            )
            started = time.monotonic()
            output = await warm(session, code)
            READY.write_text(
                json.dumps(
                    {
                        "mode": "persistent_mcp",
                        "protocol": identity.protocolVersion,
                        "server": identity.serverInfo.model_dump(),
                        "setup_elapsed_s": time.monotonic() - started,
                        "project": PROJECT,
                        "heap_size_hint_mb": heap,
                        "threads": 1,
                        "simulator": agent_simulator,
                        "pulse_designer": pulse_designer,
                        "tools": sorted(ALLOWED),
                        "warmup_output": output,
                    }
                )
            )
            lock = asyncio.Lock()

            async def handle(reader, writer):
                started = time.monotonic()
                try:
                    raw = await reader.readline()
                    name, args = validate(json.loads(raw))
                    async with lock:
                        result, rewarmed = await dispatch(
                            session, name, args, workspace, code
                        )
                    response = result.model_dump(mode="json", exclude_none=True)
                    response.pop("structuredContent", None)
                    if rewarmed:
                        response.setdefault("content", []).append(
                            {"type": "text", "text": REWARM_NOTES[rewarmed]}
                        )
                    truncate(response)
                    row = {
                        "name": name,
                        "arguments": args,
                        "elapsed_s": time.monotonic() - started,
                        "session_rewarmed": rewarmed,
                        "response": response,
                        "at": time.time(),
                    }
                    with (workspace / "mcp-tool-trace.jsonl").open("a") as trace:
                        trace.write(json.dumps(row) + "\n")
                except Exception as exc:  # noqa: BLE001 - returned to the agent
                    response = {
                        "isError": True,
                        "content": [
                            {"type": "text", "text": f"{type(exc).__name__}: {exc}"}
                        ],
                    }
                writer.write(json.dumps(response).encode() + b"\n")
                await writer.drain()
                writer.close()
                await writer.wait_closed()

            Path(SOCKET).unlink(missing_ok=True)
            listener = await asyncio.start_unix_server(
                handle, path=SOCKET, limit=2 * MAX_CODE
            )
            os.chmod(SOCKET, 0o600)
            async with listener:
                await listener.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--ready", action="store_true")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--heap", type=int, default=100)
    parser.add_argument("--pulse-designer", action="store_true")
    parser.add_argument("--agent-simulator", choices=["none", "koma"], default="none")
    parser.add_argument("--timeout", type=float, default=650)
    parser.add_argument("request", nargs="?")
    args = parser.parse_args()
    if args.serve:
        return asyncio.run(
            serve(args.workspace, args.heap, args.pulse_designer, args.agent_simulator)
        )
    if args.ready:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if READY.exists() and Path(SOCKET).exists():
                print(READY.read_text())
                return
            time.sleep(0.05)
        raise TimeoutError("MCP author warmup did not become ready")
    raw = args.request.encode()
    validate(json.loads(raw))
    with socket.socket(socket.AF_UNIX) as conn:
        conn.settimeout(None)
        conn.connect(SOCKET)
        conn.sendall(raw + b"\n")
        with conn.makefile("rb") as stream:
            response = stream.readline(2 * MAX_CODE)
    if not response.endswith(b"\n"):
        raise RuntimeError("Incomplete MCP tool response")
    sys.stdout.buffer.write(response)


if __name__ == "__main__":
    main()
