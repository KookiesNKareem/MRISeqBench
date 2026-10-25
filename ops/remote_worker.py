#!/usr/bin/env python3
"""Run one evaluation on a remote GPU host at identical absolute paths.

Drop-in for ``python -m mriseqbench.evaluation.worker`` (set
``MRISEQBENCH_EVALUATION_WORKER`` to this file). The campaign host keeps the
agents, preflight and bookkeeping; only the Koma/BART evaluation runs remotely.
Hosts come from ``MRISEQBENCH_GRADING_HOSTS`` (comma-separated ``user@host[:port]``)
and/or ``MRISEQBENCH_GRADING_HOSTS_FILE`` (one per line, re-read every time),
each with ``MRISEQBENCH_GRADING_SLOTS`` concurrent evaluations, default 8); the
least-loaded host wins, tracked with flock files under /tmp. Every host must
hold the same source tree, venv, Julia project and BART at the same paths as
this machine (``ops/gpu_host_bootstrap.sh``).
"""

import fcntl
import os
import signal
import subprocess
import sys
from pathlib import Path

SSH = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=4",
]
KEY = os.environ.get("MRISEQBENCH_GRADING_SSH_KEY")
LOCKS = Path("/tmp/mriseqbench-grading")
REMOTE_ENV = {
    # The exact Julia the project was instantiated with; juliaup's launcher
    # would otherwise pick a default from the SSH user's own HOME.
    "HOME": "/home/kfareed",
    "PATH": "/home/kfareed/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
    "MRISEQBENCH_BART": "/home/kfareed/.local/share/mriseqbench-bart-v1.0.00/bart",
    "MRISEQBENCH_KOMA_DEVICE": "cuda",
    "JULIA_CUDA_HARD_MEMORY_LIMIT": os.environ.get(
        "JULIA_CUDA_HARD_MEMORY_LIMIT", "8GiB"
    ),
    "JULIA_PKG_PRECOMPILE_AUTO": "0",
    "JULIA_DEPOT_PATH": "/home/kfareed/.julia",
    "OPENBLAS_NUM_THREADS": "1",
}


def parse_host(spec):
    user_host, _, port = spec.partition(":")
    return user_host, port or "22"


def acquire_slot(hosts):
    """Block until a slot is free on some host; return (host, locked fd)."""
    import time

    while True:
        for spec, slots in hosts:
            d = LOCKS / spec.replace("@", "_").replace(":", "_")
            d.mkdir(parents=True, exist_ok=True)
            for i in range(slots):
                fd = os.open(d / f"slot-{i}", os.O_CREAT | os.O_WRONLY, 0o644)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    os.close(fd)
                    continue
                return spec, fd
        time.sleep(2)


def main(argv):
    args = dict(zip(argv[::2], argv[1::2]))
    case, submission = Path(args["--case"]), Path(args["--submission"])
    root, workspace = Path(args["--root"]), Path(args["--workspace"])
    slots = int(os.environ.get("MRISEQBENCH_GRADING_SLOTS", "8"))
    hosts = [
        (h, slots)
        for h in os.environ.get("MRISEQBENCH_GRADING_HOSTS", "").split(",")
        if h
    ]
    hosts_file = os.environ.get("MRISEQBENCH_GRADING_HOSTS_FILE")
    if hosts_file:  # "user@host[:port] [slots]" per line, re-read per evaluation.
        for line in Path(hosts_file).read_text().splitlines():
            fields = line.split("#")[0].split()
            if fields:
                hosts.append((fields[0], int(fields[1]) if len(fields) > 1 else slots))
    if not hosts:
        sys.exit(
            "MRISEQBENCH_GRADING_HOSTS / MRISEQBENCH_GRADING_HOSTS_FILE name no host"
        )
    spec, lock = acquire_slot(hosts)
    user_host, port = parse_host(spec)
    ssh = SSH + ["-p", port] + (["-i", KEY] if KEY else [])
    rsync_ssh = " ".join(ssh)
    dirs = sorted({str(p.parent) for p in (case, submission)} | {str(workspace)})
    print(f"remote_worker: {spec} {workspace}", flush=True)
    subprocess.run(ssh + [user_host, "mkdir -p " + " ".join(dirs)], check=True)
    for src in (case, submission):
        subprocess.run(
            ["rsync", "-a", "-e", rsync_ssh, str(src), f"{user_host}:{src}"], check=True
        )
    workspace.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--delete",
            "-e",
            rsync_ssh,
            f"{workspace}/",
            f"{user_host}:{workspace}/",
        ],
        check=True,
    )
    env = " ".join(f"{k}={v}" for k, v in REMOTE_ENV.items())
    remote = (
        f"cd {root} && exec env {env} {root}/.venv/bin/python -m mriseqbench.evaluation.worker "
        + " ".join(f"{k} {v}" for k, v in args.items())
    )
    child = subprocess.Popen(ssh + ["-tt", user_host, remote])

    def forward(signum, _frame):
        child.send_signal(signum)

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, forward)
    code = child.wait()
    subprocess.run(
        ["rsync", "-a", "-e", rsync_ssh, f"{user_host}:{workspace}/", f"{workspace}/"],
        check=False,
    )
    subprocess.run(ssh + [user_host, f"rm -rf {workspace}"], check=False)
    os.close(lock)
    sys.exit(code)


if __name__ == "__main__":
    main(sys.argv[1:])
