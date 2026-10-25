"""Finish a Koma/komamripy simulator image by compiling SimWarm on a CUDA GPU.

    MRISEQBENCH_SIMULATOR_GPU=<uuid> uv run python runtimes/arms/prewarm_simulator.py \
        mriseqbench/sim-koma:cold mriseqbench/sim-koma:<tag>

The GPU-compiled pkgimage is copied into a new layer of the cold image (no
`docker commit`, so no runtime state or environment leaks into the image).
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from mriseqbench.harness.arm_runtime import mount, rootless, simulator_gpu_args

# The image build compiled SimWarm without a GPU (WARM=false): drop those caches
# so SimWarm and its KomaMRI extension are compiled on the GPU into /warm.
WARMUP = """
for d in readdir(joinpath(DEPOT_PATH[2], "compiled"); join=true), n in readdir(d; join=true)
    startswith(basename(n), "SimWarm") && rm(n; recursive=true)
end
Base.Precompilation.precompilepkgs(); using SimWarm
SimWarm.WARM || error("SimWarm compiled without CUDA")
"""


def main(cold, tag):
    with tempfile.TemporaryDirectory() as tmp:
        depot = Path(tmp) / "depot"
        depot.mkdir()
        user = "0:0" if rootless() else f"{os.getuid()}:{os.getgid()}"
        subprocess.run(
            ["docker", "run", "--rm", "--pull=never", "--network=none"]
            + ["--user", user, "--memory=8g", "--entrypoint", ""]
            + simulator_gpu_args()
            + ["--env", "JULIA_DEPOT_PATH=/warm:/opt/julia-depot"]
            + mount(depot, "/warm", readonly=False)
            + [cold, "julia", "--startup-file=no", "--project=/opt/simulator", "-e"]
            + [WARMUP],
            check=True,
        )
        (Path(tmp) / "Dockerfile").write_text(
            f"FROM {cold}\n"
            "RUN find /opt/julia-depot/compiled -maxdepth 2 -name 'SimWarm*' -exec rm -rf {} +\n"
            "COPY depot/compiled /opt/julia-depot/compiled\n"
        )
        subprocess.run(["docker", "build", "-q", "-t", tag, tmp], check=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
