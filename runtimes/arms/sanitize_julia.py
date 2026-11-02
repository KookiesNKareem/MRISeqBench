"""Prepare authoring sources: scanner-aware blocks, API isolation and ADC help."""

import re
import shutil
import subprocess
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
    # Apply before package compilation/API stripping. Already-patched sources
    # are accepted; a partial or incompatible source revision fails the build.
    sequence = (root / "src/datatypes/Sequence.jl").read_text()
    macro = (root / "src/datatypes/sequence/AddBlockMacro.jl").read_text()
    if not (
        "_block_duration(dur(seq, sys), events...)" in sequence
        and "def=seq.DEF" in macro
    ):
        subprocess.run(
            [
                "patch",
                "--batch",
                "--forward",
                "-p1",
                "-i",
                str(Path(__file__).with_name("koma-block-duration.patch").resolve()),
            ],
            cwd=root,
            check=True,
        )
    # Clarify the pinned package's existing timing convention for @doc ADC.
    # This changes documentation only and is idempotent across image layers.
    adc = root / "src/datatypes/sequence/ADC.jl"
    adc.write_text(
        adc.read_text()
        .replace(
            "- `T`: (`::Float64`, [`s`]) duration to acquire the samples",
            "- `T`: (`::Float64`, [`s`]) time between first and last sample centers "
            "for `N > 1`: `T = (N - 1) * dwell`. For `N == 1`, `T` is the dwell time. "
            "The Pulseq sampling window is `N * dwell`.",
        )
        .replace(
            "- `delay`: (`::Float64`, [`s`]) delay time to start the acquisition",
            "- `delay`: (`::Float64`, [`s`]) time to the first sample center for "
            "`N > 1`. Pulseq export uses the leading dwell edge: "
            "`delay(adc, sys) = adc.delay - dwell(adc, sys) / 2`.",
        )
    )
    timing = root / "src/datatypes/sequence/TimingChecks.jl"
    timing.write_text(
        timing.read_text().replace(
            'error("Block $block_id $label exceeds the block duration.")',
            'error("Block $block_id $label requires $(event_end) s, but the allocated '
            "block duration is $(block_duration) s (short by "
            '$(event_end - block_duration) s).")',
        )
    )
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
