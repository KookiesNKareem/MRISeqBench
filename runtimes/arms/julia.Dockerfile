# Build context must contain ONLY KomaMRIBase/ and KomaMRIFiles/ from a pinned source tree.
ARG BASE=julia:1.11
ARG NODE_BASE=node:22.19.0-bookworm-slim
FROM ${NODE_BASE} AS agent
ARG PI_PACKAGE=@earendil-works/pi-coding-agent
ARG PI_VERSION
RUN test -n "$PI_VERSION" && npm install -g ${PI_PACKAGE}@${PI_VERSION} && npm cache clean --force
FROM ${BASE}
COPY --from=agent /usr/local /usr/local
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir 'numpy>=2,<3' 'h5py>=3.12,<4'
ENV JULIA_DEPOT_PATH=/opt/julia-depot JULIA_LOAD_PATH=/opt/author:@stdlib JULIA_NUM_PRECOMPILE_TASKS=1 JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0
COPY KomaMRIBase /opt/sources/KomaMRIBase
COPY KomaMRIFiles /opt/sources/KomaMRIFiles
COPY sanitize_julia.py /tmp/sanitize_julia.py
# Resolve dependencies once without compiling any MRI package. Variant source
# removal precedes all MRI caches, while common dependency layers stay shared.
RUN python3 /tmp/sanitize_julia.py /opt/sources/KomaMRIBase true \
    && mkdir /opt/author && julia --startup-file=no --compiled-modules=no -O0 --heap-size-hint=700M --project=/opt/author -e 'using Pkg; Pkg.develop([PackageSpec(path="/opt/sources/KomaMRIBase"), PackageSpec(path="/opt/sources/KomaMRIFiles")]); Pkg.instantiate(;allow_autoprecomp=false)'
ARG PULSE_DESIGNER=false
RUN python3 /tmp/sanitize_julia.py /opt/sources/KomaMRIBase "$PULSE_DESIGNER"
ARG UNITFUL=false
ARG REVISE_VERSION=3.16.2
RUN if [ "$UNITFUL" = true ]; then julia --startup-file=no --compiled-modules=no -O0 --heap-size-hint=700M --project=/opt/author -e 'using Pkg; Pkg.add("Unitful")'; fi \
    && julia --startup-file=no --compiled-modules=no -O0 --heap-size-hint=700M --project=/opt/author -e 'using Pkg; Pkg.add(PackageSpec(name="Revise", version=ARGS[1]))' "$REVISE_VERSION" \
    && julia --startup-file=no --project=/opt/author -e 'using Revise, KomaMRIBase, KomaMRIFiles; ARGS[1] == "true" && @eval using Unitful' "$UNITFUL" \
    && python3 /tmp/sanitize_julia.py --depot /opt/julia-depot \
    && rm -rf /opt/julia-depot/registries /opt/julia-depot/logs /tmp/sanitize_julia.py
ENV JULIA_PKG_OFFLINE=true PATH=/opt/venv/bin:/usr/local/julia/bin:/usr/local/bin:/usr/bin:/bin
