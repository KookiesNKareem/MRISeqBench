"""Small native isolation probe using only synthetic private data."""

import json
import sys
import tempfile
from pathlib import Path

from ..models import Agent, SandboxConfig
from ..process import execute
from .sandbox import environment, launch


def doctor(root, backend="auto"):
    with tempfile.TemporaryDirectory(prefix="mriseqbench-doctor-") as temporary:
        base = Path(temporary)
        workspace, control = base / "agent", base / "control"
        workspace.mkdir()
        control.mkdir()
        (control / "private.txt").write_text("synthetic-private-data")
        script = """import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); result={}
try: p.read_text(); result['private_read_blocked']=False
except PermissionError: result['private_read_blocked']=True
try: p.write_text('changed'); result['outside_write_blocked']=False
except PermissionError: result['outside_write_blocked']=True
pathlib.Path('probe.json').write_text(json.dumps(result))
"""
        command = [sys.executable, "-c", script, str(control / "private.txt")]
        agent = Agent(command=command)
        try:
            with launch(
                SandboxConfig(backend=backend), command, root, workspace, control
            ) as (args, isolation):
                execution = execute(
                    args,
                    workspace,
                    15,
                    control / "probe",
                    env=environment(agent, root, workspace),
                )
            if execution["status"] != "completed":
                return {
                    "ok": False,
                    "isolation": isolation,
                    "execution": execution,
                    "stderr": (control / "probe.stderr.log").read_text(),
                }
            probe = json.loads((workspace / "probe.json").read_text())
            return {
                "ok": isolation["isolated"] and all(probe.values()),
                "backend": isolation["backend"],
                **probe,
            }
        except (ValueError, OSError) as exc:
            return {"ok": False, "reason": str(exc)}
