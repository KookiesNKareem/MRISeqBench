# Build from the repository root. Set PI_VERSION to the agent version to test.
ARG BASE=python:3.11-slim
ARG NODE_BASE=node:22.19.0-bookworm-slim
FROM ${NODE_BASE} AS agent
ARG PI_PACKAGE=@earendil-works/pi-coding-agent
ARG PI_VERSION
RUN test -n "$PI_VERSION" && npm install -g ${PI_PACKAGE}@${PI_VERSION} && npm cache clean --force
FROM ${BASE}
COPY --from=agent /usr/local /usr/local
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir pypulseq==1.5.0.post1 'numpy>=2,<3' 'h5py>=3.12,<4'
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin
