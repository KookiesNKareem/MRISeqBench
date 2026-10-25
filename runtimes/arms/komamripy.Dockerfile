# Context: pinned KomaMRI root Project.toml/src/Artifacts.toml, KomaMRIPlots/,
# and the fixed simulate.py adapter.
# BASE is the matching unwarmed native Koma simulator image (all MRI components
# share a revision); finish with prewarm_simulator.py on a GPU.
ARG BASE=mriseqbench/sim-koma:local
FROM ${BASE}
ARG PYTHON_LIB_PACKAGE=libpython3.13
RUN apt-get update && apt-get install -y --no-install-recommends ${PYTHON_LIB_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*
ENV JULIA_PKG_OFFLINE=false JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=/opt/venv/bin/python3 \
    PYTHON_JULIACALL_EXE=/usr/local/julia-gpu/bin/julia PYTHON_JULIACALL_PROJECT=/opt/simulator
RUN /opt/venv/bin/pip install --no-cache-dir komamripy==0.0.12
COPY KomaMRI /opt/sources/KomaMRI
COPY KomaMRIPlots /opt/sources/KomaMRIPlots
COPY simulate.py /opt/mriseqbench/simulate
RUN chmod 755 /opt/mriseqbench/simulate
RUN python3 -c 'import pathlib,re; paths=["/opt/sources/KomaMRI", "/opt/sources/KomaMRIPlots"]; [(lambda p:p.write_text(re.sub(r"(?ms)^\[workspace\]\n.*?(?=^\[|\Z)", "", p.read_text())))(pathlib.Path(d)/"Project.toml") for d in paths]'
# Precompile extensions KomaMRI's dependencies trigger (e.g. CUDACore's
# SpecialFunctionsExt); services cannot compile into their noexec /tmp.
RUN julia --startup-file=no --project=/opt/simulator -e 'using Pkg; Pkg.develop([PackageSpec(path="/opt/sources/KomaMRIPlots"), PackageSpec(path="/opt/sources/KomaMRI")]); Pkg.add("PythonCall")' \
    && julia --startup-file=no --heap-size-hint=1G --project=/opt/simulator -e 'using Pkg; Pkg.precompile(); using CUDA, KomaMRI, PythonCall' \
    && /opt/venv/bin/python3 -c 'import komamripy as km; assert km.core is not None and km.files is not None' \
    && rm -rf /opt/julia-depot/registries /opt/julia-depot/logs
# Without Julia signal handling, juliacall intermittently segfaults on CUDA init.
ENV JULIA_PKG_OFFLINE=true MRISEQBENCH_SIMULATOR=komamripy PYTHON_JULIACALL_HANDLE_SIGNALS=yes
CMD ["/opt/mriseqbench/simulate", "--probe"]
