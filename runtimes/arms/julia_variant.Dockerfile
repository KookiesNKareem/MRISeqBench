# Incremental build after julia.Dockerfile: BASE must be a restricted Julia
# author image from the same pinned source revision, with Unitful still absent.
# Context: clean KomaMRIBase/, KomaMRIFiles/, sanitize_julia.py.
ARG BASE=mriseqbench/author-koma_base:local
FROM ${BASE}
ENV JULIA_PKG_OFFLINE=false
COPY KomaMRIBase /opt/sources/KomaMRIBase
COPY KomaMRIFiles /opt/sources/KomaMRIFiles
COPY sanitize_julia.py /tmp/sanitize_julia.py
ARG PULSE_DESIGNER=true
ARG UNITFUL=false
ARG REVISE_VERSION=3.16.2
RUN python3 /tmp/sanitize_julia.py /opt/sources/KomaMRIBase "$PULSE_DESIGNER" \
    && if [ "$UNITFUL" = true ]; then julia --startup-file=no --project=/opt/author -e 'using Pkg; Pkg.add("Unitful")'; fi \
    && julia --startup-file=no --compiled-modules=no -O0 --heap-size-hint=700M --project=/opt/author -e 'using Pkg; Pkg.add(PackageSpec(name="Revise", version=ARGS[1]))' "$REVISE_VERSION" \
    && julia --startup-file=no --project=/opt/author -e 'using Revise, KomaMRIBase, KomaMRIFiles; ARGS[1] == "true" && @eval using Unitful' "$UNITFUL" \
    && python3 /tmp/sanitize_julia.py --depot /opt/julia-depot \
    && rm -rf /opt/julia-depot/registries /opt/julia-depot/logs /tmp/sanitize_julia.py
ENV JULIA_PKG_OFFLINE=true
