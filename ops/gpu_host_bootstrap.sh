#!/bin/bash
# Prepare a rented GPU host (Lambda VM as ubuntu, or RunPod pod as root) as a
# remote grading host for ops/remote_worker.py. The campaign host pushes
# ~/mriseqbench/src here first (see ops/push_gpu_host.sh); paths mirror Ultron.
set -euxo pipefail
SUDO=$(command -v sudo || true)
$SUDO mkdir -p /home/kfareed/mriseqbench /ssd-pool/kfareed/MRISeqBench-results/campaigns /home/kfareed/.local/share
$SUDO chown -R "$(id -u):$(id -g)" /home/kfareed /ssd-pool
export HOME=/home/kfareed
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq --no-install-recommends rsync git make gcc g++ libfftw3-dev libopenblas-dev liblapacke-dev curl ca-certificates >/dev/null
# uv (managed Python 3.11 for the locked venv) and Julia 1.12.6 via juliaup.
[ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh
[ -x "$HOME/.juliaup/bin/julia" ] || curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel 1.12.6 --path "$HOME/.juliaup"
export PATH="$HOME/.juliaup/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
cd /home/kfareed/mriseqbench/src
uv sync --locked --python 3.11
export JULIA_DEPOT_PATH=/home/kfareed/.julia JULIA_PKG_PRECOMPILE_AUTO=0 OPENBLAS_NUM_THREADS=1
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. -e 'using CUDA; @assert CUDA.functional(); println(CUDA.device())'
# CUDA BART (sm of this GPU); Ubuntu's cuda layout or /usr/local/cuda. BART
# needs GCC 12+ (Ubuntu 22.04 ships 11).
if [ "$(gcc -dumpversion | cut -d. -f1)" -lt 12 ]; then
  $SUDO apt-get install -y -qq --no-install-recommends gcc-12 g++-12 >/dev/null
  export CC=gcc-12 CXX=g++-12
fi
if [ -d /usr/local/cuda/lib64 ]; then export MRISEQBENCH_CUDA_BASE=/usr/local/cuda MRISEQBENCH_CUDA_LIB=lib64; fi
MRISEQBENCH_BART_INSTALL=/home/kfareed/.local/share/mriseqbench-bart-v1.0.00 bash runtimes/bart/install.sh
.venv/bin/python -c 'import mriseqbench, numpy; print("venv ok")'
echo "BOOTSTRAP DONE $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
