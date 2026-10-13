# CPU-only simulator service; build with runtimes/arms/ as the context.
ARG BASE=python:3.11-slim
FROM ${BASE}
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && /opt/venv/bin/pip install --no-cache-dir MRzeroCore==1.1.1 pypulseq==1.5.0.post1 'numpy>=2,<3' 'h5py>=3.12,<4'
COPY simulate.py /opt/mriseqbench/simulate
RUN chmod 755 /opt/mriseqbench/simulate
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin MRISEQBENCH_SIMULATOR=mrzero OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
