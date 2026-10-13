# Upgrade an existing, audited author image without rebuilding its MRI stack.
# Build context: this Dockerfile and sanitize_julia.py only. BASE must be an
# author image for the exact same arm; never cross PulseDesigner/Unitful variants.
ARG BASE
FROM ${BASE}
ARG REVISE_VERSION=3.16.2
ENV JULIA_PKG_OFFLINE=false
COPY sanitize_julia.py /tmp/sanitize_julia.py
RUN julia --startup-file=no --heap-size-hint=200M --project=/opt/author -e 'using Pkg; Pkg.add(PackageSpec(name="Revise", version=ARGS[1]); preserve=Pkg.PRESERVE_ALL)' "$REVISE_VERSION" \
    && julia --startup-file=no --compiled-modules=existing --heap-size-hint=200M --project=/opt/author -e 'using Revise, KomaMRIBase, KomaMRIFiles; @assert pkgversion(Revise) == VersionNumber(ARGS[1])' "$REVISE_VERSION" \
    && python3 /tmp/sanitize_julia.py --depot /opt/julia-depot \
    && rm -rf /opt/julia-depot/registries /opt/julia-depot/logs /tmp/sanitize_julia.py
ENV JULIA_PKG_OFFLINE=true
