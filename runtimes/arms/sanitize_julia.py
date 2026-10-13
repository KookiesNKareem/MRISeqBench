"""Build-time removal of PulseDesigner, including directly includable sources."""

import re
import shutil
import sys
from pathlib import Path


def sanitize_depot(root):
    for package in (Path(root) / "packages").glob("*/*"):
        for name in ("test", "docs", "examples", "benchmark", "benchmarks", ".git"):
            path = package / name
            if path.is_dir():
                shutil.rmtree(path)


def sanitize(root, enabled):
    root = Path(root)
    for package in (root, root.parent / "KomaMRIFiles"):
        if not package.exists():
            continue
        for name in ("test", "examples", "docs", ".git"):
            path = package / name
            if path.exists():
                shutil.rmtree(path)
        project = package / "Project.toml"
        if project.exists():
            project.write_text(
                re.sub(r"(?ms)^\[workspace\]\n.*?(?=^\[|\Z)", "", project.read_text())
            )
        # Upstream example precompile workloads import PulseDesigner in both
        # packages. Remove them uniformly before compiling any arm: otherwise
        # the base-only image can retain or try to load the excluded API.
        module = package / "src" / (package.name + ".jl")
        if module.exists():
            module.write_text(
                module.read_text().replace('include("precompile.jl")', "")
            )
        (package / "src/precompile.jl").unlink(missing_ok=True)
    if enabled:
        return
    source = root / "src/KomaMRIBase.jl"
    text = source.read_text()
    include = 'include("sequences/PulseDesigner.jl")'
    export = "export PulseDesigner"
    if text.count(include) != 1 or text.count(export) != 1:
        raise ValueError(
            "unrecognized KomaMRIBase source; cannot strip PulseDesigner safely"
        )
    source.write_text(text.replace(include, "").replace(export, ""))
    shutil.rmtree(root / "src/sequences")


if __name__ == "__main__":
    if sys.argv[1] == "--depot":
        sanitize_depot(sys.argv[2])
    else:
        if sys.argv[2] not in {"true", "false"}:
            raise ValueError("expected true or false")
        sanitize(sys.argv[1], sys.argv[2] == "true")
