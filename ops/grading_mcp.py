"""Native MCP client for the campaign's existing per-trial feedback API."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlsplit

TOOLS = ("lint", "check", "submit", "status", "feedback", "simulate")
TOOL_LOCK = threading.Lock()
WRITE_LOCK = threading.Lock()
ACTIVE_LOCK = threading.Lock()
ACTIVE = {}
TIMING_LOCK = threading.Lock()


def timing(stage, **details):
    control = Path.cwd().parent / "control"
    if control.is_dir():
        with TIMING_LOCK, (control / "grading-mcp-timing.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"unix_s": time.time(), "stage": stage, **details}) + "\n"
            )


def read_context():
    base = Path.cwd()
    config = json.loads((base / ".harness.json").read_text())
    endpoint = os.environ.get("MRISEQBENCH_FEEDBACK_URL", "")
    token = os.environ.get("MRISEQBENCH_FEEDBACK_TOKEN", "")
    parts = urlsplit(endpoint)
    if (
        parts.scheme != "http"
        or parts.hostname != "127.0.0.1"
        or parts.port is None
        or parts.path
        or parts.query
        or parts.fragment
        or not token
    ):
        raise RuntimeError("campaign feedback endpoint is not configured for loopback")
    return base, config, endpoint, token


def atomic_json(path: Path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def present(result, receipt: str):
    evaluation = result.get("evaluation", result)
    outcomes = [
        {**check, "check_group": group}
        for group in ("checks", "application_checks", "image_checks")
        for check in evaluation.get(group, [])
        if isinstance(check, dict) and "passed" in check
    ]
    summary = {
        "status": result.get("status", "error"),
        "failed_checks": [check for check in outcomes if check["passed"] is False],
        "unresolved_checks": [check for check in outcomes if check["passed"] is None],
        "full_receipt": receipt,
    }
    for key in ("attempt", "submissions_remaining", "reason", "error", "runtime"):
        if key in result:
            summary[key] = result[key]
    if "evaluation" in result:
        summary["evaluation_status"] = evaluation.get("status")
    # Keep the complete public response in the tool result as well as on disk.
    summary.update(result)
    return summary


def feedback_request(kind):
    base, config, endpoint, token = read_context()
    query = kind in {"status", "feedback"}
    route = "receipt" if kind == "feedback" else kind
    body = None
    if not query:
        path = base / "sequence.seq"
        if not path.is_file():
            raise ValueError("sequence.seq is missing")
        if path.stat().st_size > config["max_submission_bytes"]:
            raise ValueError("sequence.seq exceeds the submission size limit")
        body = path.read_bytes()
    request = urllib.request.Request(
        f"{endpoint}/{route}",
        data=body,
        method="GET" if query else "POST",
        headers={
            "X-MRISeqBench-Token": token,
            "Content-Type": "application/octet-stream",
        },
    )
    timeout = None if kind == "submit" else config["timeout_s"] + 5
    timing("http_start", tool=kind)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        result = json.loads(response.read())
        status_code = response.status
    timing("http_finished", tool=kind, status=status_code)
    if not isinstance(result, dict):
        raise TypeError("feedback response must be an object")
    if kind == "status":
        return result, False

    directory = base / "feedback"
    directory.mkdir(exist_ok=True)
    receipt_path = directory / f"{kind}-{uuid.uuid4().hex}.json"
    atomic_json(receipt_path, result)
    atomic_json(directory / f"latest-{kind}.json", result)
    presented = present(result, str(receipt_path.relative_to(base)))
    presented["http_status"] = status_code
    return presented, status_code >= 500


def simulation_response():
    """Return the current candidate's public check and actual preview pixels."""
    base = Path.cwd()
    candidate = base / "sequence.seq"
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    checks, check_error = feedback_request("check")
    timing("simulation_start", tool="simulate")
    process = subprocess.run(
        [str(base / "bin/simulate"), "sequence.seq"],
        cwd=base,
        capture_output=True,
        text=True,
        check=False,
    )
    timing("simulation_finished", tool="simulate", returncode=process.returncode)
    simulation = (
        json.loads(process.stdout)
        if process.returncode == 0
        else {
            "error": process.stderr[-4000:],
            "returncode": process.returncode,
        }
    )
    current = hashlib.sha256(candidate.read_bytes()).hexdigest()
    same_candidate = current == digest and simulation.get("sha256", digest) == digest
    summary = {
        "sha256": digest,
        "candidate_unchanged": same_candidate,
        "check": {
            "status": checks["status"],
            "passed_checks": [
                check["name"]
                for check in checks.get("checks", [])
                if check.get("passed") is True
            ],
            "failed_checks": checks["failed_checks"],
            "unresolved_checks": checks["unresolved_checks"],
            "full_receipt": checks["full_receipt"],
        },
        "simulation": simulation,
        "scope": "Public file/timing/hardware checks and diagnostic images. No SAR, target comparison or image pass verdict.",
    }
    output = (base / "simulation").resolve()
    output.mkdir(exist_ok=True)
    atomic_json(output / "feedback.json", summary)
    content = [{"type": "text", "text": json.dumps(summary, indent=2)}]
    reconstruction = simulation.get("reconstruction", {})
    for image in reconstruction.get("images", []):
        if not image.get("magnitude_preview"):
            continue
        path = (output / image["magnitude_preview"]).resolve()
        if not path.is_relative_to(output) or path.suffix != ".png":
            raise ValueError("Invalid simulation preview path")
        content.append(
            {
                "type": "image",
                "mimeType": "image/png",
                "data": base64.b64encode(path.read_bytes()).decode(),
            }
        )
    is_error = (
        check_error
        or process.returncode != 0
        or not same_candidate
        or reconstruction.get("status") == "failed"
    )
    return content, is_error


def tool_definitions():
    descriptions = {
        "simulate": "Simulate the current sequence.seq and return its reconstructed images together with public file/timing/hardware checks. Uses no graded submission. Inspect the returned pixels before revising or submitting.",
        "lint": "Run the existing public lint feedback on the current sequence.seq and save the full receipt.",
        "check": "Run the existing public check feedback on the current sequence.seq and save the full receipt.",
        "submit": "Submit the current sequence.seq for grading. This call blocks until grading completes; the full receipt is saved before the complete public result is returned.",
        "status": "Read current charged time, remaining time, and submission allowance.",
        "feedback": "Retrieve the last disclosed feedback without grading or consuming a submission, and save a full receipt.",
    }
    return [
        {
            "name": name,
            "description": descriptions[name],
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
        for name in TOOLS
        if name != "simulate" or (Path.cwd() / "bin/simulate").is_file()
    ]


def write_message(value):
    data = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with WRITE_LOCK:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()


def rpc_error(request_id, code, message):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def run_tool(request_id, params, cancelled):
    key = json.dumps(request_id, sort_keys=True)
    try:
        name = params.get("name")
        if name not in TOOLS:
            raise ValueError(f"unknown grading tool: {name}")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict) or arguments:
            raise ValueError(
                "grading tools take no arguments; they use the current sequence.seq"
            )
        with TOOL_LOCK:
            timing("lock_acquired", tool=name, request_id=request_id)
            if cancelled.is_set():
                return
            if name == "simulate":
                content, is_error = simulation_response()
            else:
                value, is_error = feedback_request(name)
                content = [
                    {
                        "type": "text",
                        "text": json.dumps(value, indent=2, allow_nan=False),
                    }
                ]
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": content,
                "isError": is_error,
            },
        }
    except Exception as exc:  # noqa: BLE001 - return protocol errors to the agent
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": f"Feedback request failed: {type(exc).__name__}: {exc}",
                    }
                ],
                "isError": True,
            },
        }
    finally:
        if cancelled.is_set():
            with ACTIVE_LOCK:
                ACTIVE.pop(key, None)
    if not cancelled.is_set():
        write_message(response)
        timing("response_sent", tool=name, request_id=request_id)
        with ACTIVE_LOCK:
            ACTIVE.pop(key, None)


def serve():
    for line in sys.stdin.buffer:
        try:
            message = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            write_message(rpc_error(None, -32700, "invalid JSON"))
            continue
        if not isinstance(message, dict):
            write_message(rpc_error(None, -32600, "invalid request"))
            continue
        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") or {}
        if method == "notifications/initialized":
            continue
        if method == "notifications/cancelled":
            cancelled_id = params.get("requestId") if isinstance(params, dict) else None
            key = json.dumps(cancelled_id, sort_keys=True)
            with ACTIVE_LOCK:
                event = ACTIVE.get(key)
            if event is not None:
                event.set()
                write_message(rpc_error(cancelled_id, -32800, "request cancelled"))
            continue
        if request_id is None:
            continue
        if method == "initialize":
            requested = (
                params.get("protocolVersion") if isinstance(params, dict) else None
            )
            version = requested if isinstance(requested, str) else "2024-11-05"
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": version,
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {
                            "name": "mriseqbench-grading",
                            "version": "1.0.0",
                        },
                    },
                }
            )
        elif method == "ping":
            write_message({"jsonrpc": "2.0", "id": request_id, "result": {}})
        elif method == "tools/list":
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": tool_definitions()},
                }
            )
        elif method == "tools/call":
            timing("rpc_received", tool=params.get("name"), request_id=request_id)
            key = json.dumps(request_id, sort_keys=True)
            cancelled = threading.Event()
            with ACTIVE_LOCK:
                ACTIVE[key] = cancelled
            threading.Thread(
                target=run_tool,
                args=(request_id, params, cancelled),
                daemon=True,
            ).start()
        else:
            write_message(rpc_error(request_id, -32601, "method not found"))


if __name__ == "__main__":
    serve()
