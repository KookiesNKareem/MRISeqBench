"""Standalone image audit, injected read-only by the host (never image supplied)."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys


def audit(arm, julia_heap_size_hint_mb=None):
    errors = []
    versions = {}
    native_versions = {}
    python_author = arm["authoring"] == "pypulseq"
    for module, required in {
        "numpy": True,
        "h5py": True,
        "pypulseq": python_author,
    }.items():
        if (importlib.util.find_spec(module) is not None) != required:
            errors.append(f"unexpected availability: {module} (required={required})")
        elif required:
            versions[module] = getattr(__import__(module), "__version__", "unknown")
    forbidden = {
        "mrzerocore",
        "mrzero",
        "komamripy",
        "komamri",
        "komamricore",
        "pulseqzero",
        "jemris",
        "juliacall",
        "pyjulia",
    }
    for executable in ("jemris", "pjemris"):
        if shutil.which(executable):
            errors.append(f"forbidden executable: {executable}")
    if not python_author:
        forbidden |= {"pypulseq"}
    forbidden |= {"pulseq_matlab", "+mr"}
    for executable in ("matlab", "octave", "octave-cli"):
        if shutil.which(executable):
            errors.append(f"forbidden executable: {executable}")
    if arm["language"] != "julia":
        forbidden |= {"komamribase", "komamrifiles", "unitful"}
        if shutil.which("julia"):
            errors.append("Julia executable is available in a non-Julia arm")
    if not arm["pulse_designer"]:
        forbidden |= {"pulsedesigner"}
    if not arm["unitful"]:
        forbidden |= {"unitful"}
    # Scan sources and precompiled-cache names as well as import paths. Hidden
    # source copies are usable with include()/sys.path even when not exported.
    for root in ("/opt", "/usr", "/root", "/home", "/srv", "/app"):
        for directory, dirs, files in os.walk(root):
            for name in [*dirs, *files]:
                # KomaMRIFiles' JEMRIS.jl only reads/writes phantom HDF5 data;
                # it is part of the allowed Files API, not the simulator.
                if (
                    f"{directory}/{name}"
                    == "/opt/sources/KomaMRIFiles/src/Phantom/JEMRIS.jl"
                ):
                    continue
                normalized = name.lower().split(".")[0].replace("-", "_")
                if normalized in forbidden:
                    errors.append(f"forbidden package/source: {directory}/{name}")
    if arm["language"] == "julia":
        code = """using Revise, KomaMRIBase, KomaMRIFiles
@assert isdefined(KomaMRIBase, :PulseDesigner) == PD
@assert isdefined(KomaMRIFiles, :write_seq)
@assert (Base.find_package("Unitful") !== nothing) == UNITS
@assert Base.find_package("KomaMRICore") === nothing
@assert Base.find_package("KomaMRI") === nothing
""".replace("PD", str(arm["pulse_designer"]).lower()).replace(
            "UNITS", str(arm["unitful"]).lower()
        )
        if arm["unitful"]:
            code += '\nusing Unitful; @assert 1u"ms" == 0.001u"s"\n'
        code += '\nprintln("{\\"julia\\":\\"", VERSION, "\\",\\"Revise\\":\\"", pkgversion(Revise), "\\",\\"KomaMRIBase\\":\\"", pkgversion(KomaMRIBase), "\\",\\"KomaMRIFiles\\":\\"", pkgversion(KomaMRIFiles), "\\"}")\n'
        result = subprocess.run(
            [
                "julia",
                "--startup-file=no",
                "--compiled-modules=existing",
                *(
                    [f"--heap-size-hint={julia_heap_size_hint_mb}M"]
                    if julia_heap_size_hint_mb is not None
                    else []
                ),
                "--project=/opt/author",
                "-e",
                code,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
            env={
                **os.environ,
                "JULIA_DEPOT_PATH": "/opt/julia-depot",
                "JULIA_LOAD_PATH": "/opt/author:@stdlib",
            },
        )
        if result.returncode:
            errors.append("Julia authoring probe failed: " + result.stderr[-2000:])
        elif getattr(result, "stdout", "").strip():
            native_versions = json.loads(result.stdout.splitlines()[-1])
    return {
        "ok": not errors,
        "errors": errors,
        "language": arm["language"],
        "authoring": arm["authoring"],
        "pulse_designer": arm["pulse_designer"],
        "unitful": arm["unitful"],
        "python_packages": versions,
        "native_packages": native_versions,
    }


if __name__ == "__main__":
    try:
        result = audit(
            json.loads(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else None
        )
    except (OSError, ValueError, ImportError, subprocess.SubprocessError) as exc:
        result = {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(result))
