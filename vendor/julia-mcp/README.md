# Julia MCP source

Vendored from [aplavin/julia-mcp](https://github.com/aplavin/julia-mcp) at commit
`06e83b1569a736f16a3c58f4b39e44fc33ad28bb`, under the included MIT license.

The Julia MCP arm image copies `server.py` and `LICENSE`; its Python dependencies
are pinned with hashes in `runtimes/arms/julia-mcp-requirements.txt`. The upstream
project metadata and lockfile are retained here for dependency provenance.

Keep upstream source separate from project lint and formatting changes. The arm
integration lives in `src/mriseqbench/harness/arm_mcp.py` and its bridge/extension.
