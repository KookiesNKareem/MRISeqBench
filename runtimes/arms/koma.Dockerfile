# Build context: clean KomaMRICore/ plus the three fixed adapter scripts.
# BASE is the PulseDesigner-enabled author image from the same source revision.
ARG BASE=mriseqbench/author-pulseq_julia:local
FROM ${BASE}
ENV JULIA_PKG_OFFLINE=false JULIA_LOAD_PATH=/opt/simulator:@stdlib
RUN /opt/venv/bin/pip install --no-cache-dir pypulseq==1.5.0.post1
COPY KomaMRICore /opt/sources/KomaMRICore
RUN python3 -c 'import pathlib,re; root=pathlib.Path("/opt/sources/KomaMRICore"); p=root/"Project.toml"; p.write_text(re.sub(r"(?ms)^\[workspace\]\n.*?(?=^\[|\Z)", "", p.read_text())); p=root/"src/KomaMRICore.jl"; p.write_text(p.read_text().replace("include(\"precompile.jl\")", "")); (root/"src/precompile.jl").unlink(missing_ok=True)'
RUN mkdir /opt/simulator && julia --startup-file=no --project=/opt/simulator -e 'using Pkg; Pkg.develop([PackageSpec(path="/opt/sources/KomaMRIBase"), PackageSpec(path="/opt/sources/KomaMRIFiles"), PackageSpec(path="/opt/sources/KomaMRICore")]); Pkg.add(["JSON", "HDF5"])' \
    && julia --startup-file=no --project=/opt/simulator -e 'using KomaMRIBase, KomaMRIFiles, KomaMRICore, JSON, HDF5' \
    && rm -rf /opt/julia-depot/registries /opt/julia-depot/logs
COPY simulate.py /opt/mriseqbench/simulate
COPY simulate_koma.jl /opt/mriseqbench/simulate_koma.jl
RUN chmod 755 /opt/mriseqbench/simulate
ENV JULIA_PKG_OFFLINE=true MRISEQBENCH_SIMULATOR=koma
