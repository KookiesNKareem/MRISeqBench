"""Argument-list subprocesses with deadlines and process-group cleanup."""

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def expand(command, root, workspace, **paths):
    values = {
        "python": sys.executable,
        "root": str(root),
        "workspace": str(workspace),
        **{key: str(value) for key, value in paths.items()},
    }
    return [
        re.sub(r"\{([a-z_]+)\}", lambda match: values.get(match[1], match[0]), arg)
        for arg in command
    ]


def execute(command, cwd, timeout_s, log_prefix, env=None, stop_when=None):
    prefix = Path(log_prefix)
    with (
        prefix.with_suffix(".stdout.log").open("w") as out,
        prefix.with_suffix(".stderr.log").open("w") as err,
    ):
        try:
            proc = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=out,
                stderr=err,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
        except OSError as exc:
            return {"status": "launch_error", "error": str(exc)}
        timed_out = False
        submitted = False
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                if stop_when is not None and stop_when.is_set():
                    submitted, code = True, None
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out, code = True, None
                    break
                try:
                    code = proc.wait(timeout=min(0.1, remaining))
                    submitted = stop_when is not None and stop_when.is_set()
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # Clean up the entire process group, including background children.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    return {
        "status": "submitted"
        if submitted
        else "timeout"
        if timed_out
        else "completed"
        if code == 0
        else "error",
        "returncode": code,
    }
