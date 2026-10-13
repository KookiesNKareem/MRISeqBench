# BASE must be an engine-specific, preinstalled image with python3, NumPy,
# h5py, PyPulseq and either MRzeroCore, komamripy or KomaMRI.
# Koma's Julia project must be /opt/simulator, and its depot /opt/julia-depot.
ARG BASE
FROM ${BASE}
ARG ENGINE
ENV MRISEQBENCH_SIMULATOR=${ENGINE} JULIA_DEPOT_PATH=/opt/julia-depot JULIA_PKG_OFFLINE=true
COPY runtimes/arms/simulate.py /opt/mriseqbench/simulate
COPY runtimes/arms/simulate_koma.jl /opt/mriseqbench/simulate_koma.jl
RUN chmod 755 /opt/mriseqbench/simulate
