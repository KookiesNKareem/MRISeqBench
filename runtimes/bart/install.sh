#!/usr/bin/env bash
# Pinned BART. Production inverse NUFFT requires the Linux CUDA build.
# macOS builds support local format/FFT controls, not production NUFFT grading.
set -euo pipefail

bart_commit=37f3c5f0feaf0a19a7fca85c926ad74101f4a034
bart_destination="${MRISEQBENCH_BART_INSTALL:-${HOME}/.local/share/mriseqbench-bart-v1.0.00}"
if [[ ! -d "${bart_destination}/.git" ]]; then
    git clone --depth 1 --branch v1.0.00 https://github.com/mrirecon/bart.git "${bart_destination}"
fi
if [[ "$(git -C "${bart_destination}" rev-parse HEAD)" != "${bart_commit}" ]]; then
    printf '%s\n' 'Existing BART source does not match the pinned commit.' >&2
    exit 1
fi

if [[ "$(uname -s)" == Darwin ]]; then
    bart_brew_prefix="$(brew --prefix)"
    bart_gcc="$(find "${bart_brew_prefix}/bin" -maxdepth 1 -name 'gcc-[0-9]*' | sort -V | tail -1)"
    [[ -n "${bart_gcc}" ]] || { printf '%s\n' 'Install Homebrew gcc first.' >&2; exit 1; }
    bart_sdk="$(xcrun --show-sdk-path)"
    gmake -C "${bart_destination}" -j8 bart \
        "CC=${bart_gcc} -isysroot ${bart_sdk}" OMP=0 MACPORTS=0 PNG=0 \
        "FFTW_BASE=$(brew --prefix fftw)" "BLAS_BASE=$(brew --prefix openblas)"
else
    command -v nvcc >/dev/null || { printf '%s\n' 'CUDA nvcc is required; no CPU reconstruction fallback.' >&2; exit 1; }
    make -C "${bart_destination}" -j4 bart OMP=0 PNG=0 OPENBLAS=1 CUDA=1 \
        "CUDA_BASE=${MRISEQBENCH_CUDA_BASE:-/usr}" \
        "CUDA_LIB=${MRISEQBENCH_CUDA_LIB:-lib/x86_64-linux-gnu}"
fi
"${bart_destination}/bart" version
printf 'BART executable: %s/bart\n' "${bart_destination}"
