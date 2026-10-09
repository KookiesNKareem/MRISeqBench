"""Materialize fixed, versioned KomaMRI phantoms through the Julia registry."""

import hashlib
import json
import shutil
from pathlib import Path

from ..io import write_json
from ..process import execute


def materialize(case, root, output, timeout_s=600):
    executable = shutil.which("julia")
    if executable is None:
        raise ValueError("Julia is required to materialize KomaMRI phantoms")
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "case.json", case)
    execution = execute(
        [
            executable,
            "--startup-file=no",
            f"--project={root}",
            str(root / "src/mriseqbench/phantoms/loader.jl"),
            str(output / "case.json"),
            str(output),
        ],
        root,
        timeout_s,
        output / "julia",
    )
    if execution["status"] != "completed":
        raise ValueError(
            f"phantom materialization {execution['status']}; see {output / 'julia.stderr.log'}"
        )
    for filename in ("phantom.phantom", "fields.h5", "manifest.json"):
        path = output / filename
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"phantom materialization omitted {filename}")
    manifest = json.loads((output / "manifest.json").read_text())
    write_json(
        output / "artifacts.json",
        {
            "phantom": case["object"]["phantom"],
            "physics": f"{case['physics']['id']}@{case['physics']['version']}",
            "seed": case["seed"],
            "files": {
                filename: hashlib.sha256((output / filename).read_bytes()).hexdigest()
                for filename in ("phantom.phantom", "fields.h5", "manifest.json")
            },
        },
    )
    return manifest
