"""Argument-list subprocesses with deadlines and process-group cleanup."""

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
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


def execute(
    command, cwd, timeout_s, log_prefix, env=None, stop_when=None, timeout_credit=None
):
    prefix = Path(log_prefix)
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    credit_start = timeout_credit() if timeout_credit else 0.0

    def finish(result):
        elapsed = time.monotonic() - started
        credit = max(0.0, (timeout_credit() if timeout_credit else 0.0) - credit_start)
        result.update(
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
            wall_s=elapsed,
            excluded_grading_wall_s=min(credit, elapsed),
            charged_wall_s=max(0.0, elapsed - credit),
        )
        prefix.with_suffix(".execution.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        return result

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
            return finish({"status": "launch_error", "error": str(exc)})
        timed_out = False
        submitted = False
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        try:
            while True:
                if stop_when is not None and stop_when.is_set():
                    submitted, code = True, None
                    break
                remaining = (
                    (
                        deadline
                        + (timeout_credit() if timeout_credit else 0.0)
                        - time.monotonic()
                    )
                    if deadline is not None
                    else 0.1
                )
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
    return finish(
        {
            "status": "submitted"
            if submitted
            else "timeout"
            if timed_out
            else "completed"
            if code == 0
            else "error",
            "returncode": code,
        }
    )
