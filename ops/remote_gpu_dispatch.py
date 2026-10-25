#!/usr/bin/env python3
"""SSH transport for qualified remote GPUs; the scientific worker is unchanged.

Only evaluation inputs/checkpoints go to the scratch host. A fresh SHA-256
manifest must verify on Ultron before the result is published. Losing the SSH
input channel kills the remote process group, including Julia/BART children.
"""

import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path


def remote_hosts(config):
    path = Path(config).with_name("remote-grading-hosts.json")
    if not path.exists():
        return {}
    hosts = json.loads(path.read_text())
    if not isinstance(hosts, dict):
        raise TypeError("remote grading hosts must be an object")
    for gpu, host in hosts.items():
        if not isinstance(gpu, str) or not isinstance(host, dict):
            raise TypeError("invalid remote GPU entry")
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+@[a-zA-Z0-9.-]+", host["endpoint"]):
            raise ValueError("invalid SSH endpoint")
        if type(host["port"]) is not int or not 1 <= host["port"] <= 65535:
            raise ValueError("invalid SSH port")
        for key in ("key", "root"):
            if not Path(host[key]).is_absolute():
                raise ValueError(f"remote {key} must be absolute")
        if not host.get("qualified_at") or not host.get("gpu_name"):
            raise ValueError("remote GPU needs qualification metadata")
    return hosts


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(directory):
    result = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            raise ValueError("remote workspace contains a symlink")
        if path.is_file() and path != Path(directory) / "remote-transfer-manifest.json":
            result[str(path.relative_to(directory))] = digest(path)
    return result


def verify(directory):
    expected = json.loads(
        (Path(directory) / "remote-transfer-manifest.json").read_text()
    )
    if expected != manifest(directory):
        raise ValueError("remote grading transfer failed SHA-256 verification")


def serve():
    request = json.loads(sys.stdin.readline())
    root, workspace = Path(request["root"]), Path(request["workspace"])
    worker = subprocess.Popen(
        [
            str(root / ".venv/bin/python"),
            "-m",
            "mriseqbench.evaluation.worker",
            *request["argv"],
        ],
        cwd=root,
        stdin=subprocess.DEVNULL,
        env={**os.environ, **request["env"]},
        start_new_session=True,
    )
    cancelled = threading.Event()
    terminated = threading.Event()

    def terminate(*_):
        cancelled.set()
        try:
            os.killpg(worker.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # The Python parent may exit before a Julia/BART descendant does.
            # Reap/kill the entire group even when the direct child has exited.
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            terminated.set()

    def channel():
        # The client retains this pipe for the full lifetime of the worker.
        sys.stdin.read()
        if worker.poll() is None:
            terminate()

    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, terminate)
    threading.Thread(target=channel, daemon=True).start()
    code = worker.wait()
    if cancelled.is_set():
        terminated.wait(timeout=6)
        (workspace / "result.json").unlink(missing_ok=True)
        code = 125
    (workspace / "remote-transfer-manifest.json").write_text(
        json.dumps(manifest(workspace), sort_keys=True) + "\n"
    )
    return code


def dispatch(host, gpu, argv):
    options = dict(zip(argv[::2], argv[1::2]))
    root, workspace = Path(options["--root"]), Path(options["--workspace"])
    if str(root) != host["root"]:
        raise ValueError("remote host is qualified for a different frozen source")
    ssh = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-p",
        str(host["port"]),
        "-i",
        host["key"],
    ]
    target = host["endpoint"]
    transport = shlex.join(ssh)
    identity = uuid.uuid4().hex
    staging = Path(tempfile.mkdtemp(prefix=".remote-fetch-", dir=workspace.parent))

    def remote(command, **kwargs):
        return subprocess.run(
            ssh + [target, shlex.join(command)], check=True, timeout=120, **kwargs
        )

    def sync(source, destination, delete=False):
        subprocess.run(
            [
                "rsync",
                "-a",
                "--protect-args",
                "--safe-links",
                "--exclude=/worker-*.stdout.log",
                "--exclude=/worker-*.stderr.log",
                *(["--delete"] if delete else []),
                "-e",
                transport,
                "--",
                source,
                destination,
            ],
            check=True,
            timeout=300,
        )

    receipt = workspace / f"remote-dispatch-{identity}.json"
    state = {
        "gpu_uuid": gpu,
        "host": target,
        "port": host["port"],
        "transport": "ssh-rsync-sha256",
        "status": "uploading",
    }
    receipt.write_text(json.dumps(state, indent=2) + "\n")
    try:
        inputs = [
            Path(options[key])
            for key in ("--case", "--submission", "--backend")
            if key in options
        ]
        directories = sorted({str(p.parent) for p in inputs} | {str(workspace)})
        remote(["mkdir", "-p", "--", *directories])
        # Case can be inside the workspace; copying it again is harmless.
        sync(str(workspace) + "/", f"{target}:{workspace}/", delete=True)
        for source in inputs:
            sync(str(source), f"{target}:{source}")
        remote(
            [
                "rm",
                "-f",
                "--",
                str(workspace / "result.json"),
                str(workspace / "remote-transfer-manifest.json"),
            ]
        )
        environment = {
            "HOME": "/home/kfareed",
            "PATH": "/home/kfareed/.juliaup/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
            "JULIA_DEPOT_PATH": "/home/kfareed/.julia",
            "JULIA_PKG_PRECOMPILE_AUTO": "0",
            "JULIA_CUDA_HARD_MEMORY_LIMIT": "8GiB",
            "MRISEQBENCH_BART": "/home/kfareed/.local/share/mriseqbench-bart-v1.0.00/bart",
            "MRISEQBENCH_KOMA_DEVICE": "cuda",
            "CUDA_VISIBLE_DEVICES": gpu,
            "OPENBLAS_NUM_THREADS": "1",
        }
        command = ["python3", str(root / "ops/remote_gpu_dispatch.py"), "--serve"]
        child = subprocess.Popen(
            ssh + [target, shlex.join(command)], stdin=subprocess.PIPE, text=True
        )
        stopped = False

        def cancel(*_):
            nonlocal stopped
            stopped = True
            child.stdin.close()  # EOF makes the remote supervisor kill its process group.

        prior = {
            s: signal.signal(s, cancel)
            for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        }
        try:
            child.stdin.write(
                json.dumps(
                    {
                        "argv": argv,
                        "root": str(root),
                        "workspace": str(workspace),
                        "env": environment,
                    }
                )
                + "\n"
            )
            child.stdin.flush()
            state["status"] = "running"
            receipt.write_text(json.dumps(state, indent=2) + "\n")
            code = child.wait()
        finally:
            child.stdin.close()
            for number, handler in prior.items():
                signal.signal(number, handler)
        sync(f"{target}:{workspace}/", str(staging) + "/")
        verify(staging)
        # Publish result last. A failed/incomplete transfer can never supply a verdict.
        for source in sorted(staging.rglob("*")):
            if source.is_file() and source != staging / "result.json":
                destination = workspace / source.relative_to(staging)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
        result = staging / "result.json"
        if code == 0 and not stopped and result.exists():
            os.replace(result, workspace / "result.json")
        else:
            (workspace / "result.json").unlink(missing_ok=True)
        state.update(
            status="verified" if code == 0 and not stopped else "interrupted",
            exit_code=code,
        )
        receipt.write_text(json.dumps(state, indent=2) + "\n")
        # Cleanup failure does not invalidate an already verified scientific verdict.
        try:
            remote(["rm", "-rf", "--", str(workspace)])
            outside = [
                str(path) for path in inputs if not path.is_relative_to(workspace)
            ]
            if outside:
                remote(["rm", "-f", "--", *outside])
            state["scratch_cleanup"] = "complete"
        except (OSError, subprocess.SubprocessError) as exc:
            state["scratch_cleanup"] = "pending"
            state["cleanup_error"] = str(exc)
            print(
                "Remote scratch cleanup pending; verified outputs are on Ultron",
                file=sys.stderr,
            )
        receipt.write_text(json.dumps(state, indent=2) + "\n")
        return 125 if stopped else code
    finally:
        shutil.rmtree(staging)


if __name__ == "__main__":
    if sys.argv[1:] == ["--serve"]:
        raise SystemExit(serve())
    raise SystemExit("Use through local_gpu_worker.py")
