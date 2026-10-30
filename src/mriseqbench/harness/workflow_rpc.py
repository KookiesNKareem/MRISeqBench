"""Standalone Pi RPC driver copied into the sandbox. One process, real user turns."""

import base64
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


def rpc_command(command):
    """Convert the pinned print-mode invocation, retaining provider/model/session flags."""
    result = []
    i = 0
    while i < len(command):
        token = command[i]
        if token == "--mode":
            i += 2
        elif token in {"-p", "--print"} or token == "{prompt_text}":
            i += 1
        else:
            result.append(token)
            i += 1
    return [*result, "--mode", "rpc"]


def request(config, kind, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{config['port']}{config.get('route_prefix', '')}/{kind}",
        data=data,
        headers={"X-MRISeqBench-Token": config["token"]},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=config["timeout_s"]) as response:
        return json.loads(response.read())


def drive(command, root):
    config = json.loads((root / ".harness.json").read_text())
    child = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )

    def send(message, index):
        child.stdin.write(
            json.dumps(
                {
                    "id": f"stage-{index}",
                    "type": "prompt",
                    "message": message,
                    "streamingBehavior": "followUp",
                }
            )
            + "\n"
        )
        child.stdin.flush()

    try:
        send((root / "PROMPT.md").read_text(), 0)
        last_error = False
        for line in child.stdout:
            print(line, end="", flush=True)
            event = json.loads(line)
            if event.get("type") == "response" and event.get("success") is False:
                raise RuntimeError(f"Pi RPC command rejected: {event.get('error')}")
            if (
                event.get("type") == "message_end"
                and event.get("message", {}).get("role") == "assistant"
            ):
                last_error = event["message"].get("stopReason") == "error"
            if event.get("type") != "agent_settled":
                continue
            # Pinned Pi 0.87 emits agent_settled only after automatic retries and
            # compaction have ended. agent_end alone is not a turn boundary.
            if last_error:
                raise RuntimeError("Pi ended a stage with an inference error")
            status = request(config, "status")
            if not status.get("awaiting_next_turn"):
                return 0
            response = request(config, "next", {"after_stage": status["stage_index"]})
            if response.get("status") != "next_stage":
                raise RuntimeError(
                    f"workflow transition failed: {response.get('status')}"
                )
            for name, content in response["files"].items():
                if name not in {
                    "case.json",
                    "task.yaml",
                    "EVALUATION.md",
                    "CURRENT_STAGE.md",
                }:
                    raise RuntimeError("unexpected public workflow file")
                path = root / name
                if path.is_symlink():
                    path.unlink()
                path.write_text(content)
            if "target_npz_base64" in response:
                target = root / "target.npz"
                if target.is_symlink():
                    target.unlink()
                target.write_bytes(
                    base64.b64decode(response["target_npz_base64"], validate=True)
                )
            send(response["message"], response["stage_index"])
        raise RuntimeError(f"Pi RPC stream ended unexpectedly ({child.wait()})")
    finally:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def main():
    root = Path(__file__).resolve().parent
    try:
        return drive(sys.argv[1:], root)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(
            json.dumps({"type": "workflow_driver_error", "error": str(exc)}), flush=True
        )
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
