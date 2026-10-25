#!/usr/bin/env python3
"""Dispatch trusted evaluations to a bounded pool of local and qualified SSH CUDA devices.

Set MRISEQBENCH_EVALUATION_WORKER to this file and
MRISEQBENCH_GRADING_GPU_CONFIG to a JSON file containing a list of
{"uuid": "GPU-...", "slots": 1}. Waiting dispatchers reload this file; running
workers finish on their assigned GPU. Preflight stays on the normal worker.
"""

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from remote_gpu_dispatch import dispatch, remote_hosts


def main(argv):
    options = dict(zip(argv[::2], argv[1::2]))
    root, workspace = Path(options["--root"]), Path(options["--workspace"])
    # The executable's shebang uses system Python; load this campaign's stdlib-
    # only pool parser from the same source tree as its evaluation worker.
    sys.path.insert(0, str(root / "src"))
    from mriseqbench.harness.grading_pool import read_gpu_pool

    config = Path(os.environ["MRISEQBENCH_GRADING_GPU_CONFIG"])
    locks = config.parent / "grading-gpu-locks"
    locks.mkdir(exist_ok=True)
    local_names, last_error = {}, None
    while True:
        try:
            devices = read_gpu_pool(config)
            hosts = remote_hosts(config)
            if any(
                d["slots"] and d["uuid"] not in local_names and d["uuid"] not in hosts
                for d in devices
            ):
                inventory = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
                    text=True,
                    timeout=10,
                )
                local_names = dict(
                    line.strip().split(", ", 1) for line in inventory.splitlines()
                )
            names = dict(local_names)
            names.update({gpu: host["gpu_name"] for gpu, host in hosts.items()})
            if any(d["slots"] and d["uuid"] not in names for d in devices):
                raise ValueError("configured grading GPU is absent from nvidia-smi")
            last_error = None
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
        ) as exc:
            # A bad live update must not crash a waiting submission. The runner
            # also pauses admission and exposes this error in pool status.
            if str(exc) != last_error:
                print(
                    f"grading pool unavailable; waiting for repair: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                last_error = str(exc)
            time.sleep(0.2)
            continue
        candidates = [
            (d, slot)
            for slot in range(max((d["slots"] for d in devices), default=0))
            for d in devices
            if slot < d["slots"]
        ]
        for device, slot in candidates:
            fd = os.open(
                locks / f"{device['uuid']}-{slot}.lock", os.O_CREAT | os.O_RDWR, 0o600
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            try:
                try:
                    current = read_gpu_pool(config)
                except (OSError, ValueError, TypeError):
                    break
                if current != devices or remote_hosts(config) != hosts:
                    # Re-plan if the pool changed while acquiring the lock.
                    break
                environment = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=device["uuid"],
                    MRISEQBENCH_KOMA_DEVICE="cuda",
                )
                workspace.mkdir(parents=True, exist_ok=True)
                receipt = {
                    "gpu_uuid": device["uuid"],
                    "gpu_name": names[device["uuid"]],
                    "slot": slot,
                    "wrapper_pid": os.getpid(),
                    "pool_config": devices,
                    "started_at": datetime.now(UTC).isoformat(),
                }
                # Each resumed worker keeps its own device assignment receipt.
                path = workspace / f"grading-gpu-{os.getpid()}.json"
                path.write_text(json.dumps(receipt, indent=2) + "\n")
                print(f"local_gpu_worker: {device['uuid']} slot {slot}", flush=True)
                if device["uuid"] in hosts:
                    receipt["remote_host"] = hosts[device["uuid"]]["endpoint"]
                    path.write_text(json.dumps(receipt, indent=2) + "\n")
                    return dispatch(hosts[device["uuid"]], device["uuid"], argv)
                return subprocess.call(
                    [
                        str(root / ".venv/bin/python"),
                        "-m",
                        "mriseqbench.evaluation.worker",
                        *argv,
                    ],
                    env=environment,
                )
            finally:
                os.close(fd)
        time.sleep(0.2)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
