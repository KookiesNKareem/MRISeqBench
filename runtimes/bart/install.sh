#!/usr/bin/env bash
# Pinned CPU BART. macOS prerequisites: brew install gcc make fftw openblas
# Debian/Ubuntu: apt install git make gcc libfftw3-dev libopenblas-dev liblapacke-dev
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
    make -C "${bart_destination}" -j8 bart OMP=0 PNG=0 OPENBLAS=1
fi
"${bart_destination}/bart" version
printf 'BART executable: %s/bart\n' "${bart_destination}"
