# Julia MCP arms: a Julia author image plus the pinned aplavin/julia-mcp server
# (vendor/julia-mcp, MIT) in its own venv. Build from the repository root with
# BASE=mriseqbench/author-koma_base:local (j6) or author-pulseq_julia:local (j7).
# /opt/author and its packages stay exactly those of BASE.
ARG BASE
FROM ${BASE}
COPY runtimes/arms/julia-mcp-requirements.txt /tmp/mcp-requirements.txt
RUN python3 -m venv /opt/mcp-venv \
    && /opt/mcp-venv/bin/pip install --no-cache-dir --require-hashes -r /tmp/mcp-requirements.txt \
    && rm /tmp/mcp-requirements.txt
COPY vendor/julia-mcp/server.py vendor/julia-mcp/LICENSE /opt/julia-mcp/
