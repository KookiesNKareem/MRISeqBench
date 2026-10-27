# Authoring and simulation arms

Each arm selects the authoring tools and optional agent-side simulator available
to the agent. The trusted judge and submission allowance are shared.

Simulator modifiers follow the abstract: **S1** = MRzero/Python, **S2** =
KomaMRI/Julia, and **S3** = KomaMRI through komamripy/Python. No simulator has
no S modifier. These modifiers describe agent access; the shared judge always
uses KomaMRI. P/J arm names remain authoring configuration names; the phantom
label **P1** is a separate axis. Full mapping: [labels](LABELS.md).

| Arm / simulator modifier | Authoring | PulseDesigner | Unitful | Simulator | Julia tools |
|---|---|:---:|:---:|---|---|
| P0 (`p0_pypulseq@1`) | PyPulseq | — | — | None | — |
| P1 S1 (`p1_pypulseq_mrzero@1`) | PyPulseq | — | — | S1 · MRzero | — |
| P2 S3 (`p2_pypulseq_koma@1`) | PyPulseq | — | — | S3 · KomaMRI / komamripy | — |
| J0 (`j0_koma_base@1`) | KomaMRIBase / KomaMRIFiles | No | No | None | CLI |
| J1 (`j1_pulseq_julia@1`) | KomaMRIBase / KomaMRIFiles | Yes | No | None | CLI |
| J2 (`j2_pulseq_julia_unitful@1`) | KomaMRIBase / KomaMRIFiles | Yes | Yes | None | CLI |
| J3 S2 (`j3_koma_base_koma@1`) | KomaMRIBase / KomaMRIFiles | No | No | S2 · KomaMRI | CLI |
| J4 S2 (`j4_pulseq_julia_koma@1`) | KomaMRIBase / KomaMRIFiles | Yes | No | S2 · KomaMRI | CLI |
| J5 S2 (`j5_pulseq_julia_unitful_koma@1`) | KomaMRIBase / KomaMRIFiles | Yes | Yes | S2 · KomaMRI | CLI |
| J6 (`j6_koma_base_julia_mcp@1`) | KomaMRIBase / KomaMRIFiles | No | No | None | MCP |
| J7 S2 (`j7_pulseq_julia_mcp_koma@1`) | KomaMRIBase / KomaMRIFiles | Yes | No | S2 · KomaMRI | MCP |

```bash
uv run mriseqbench arms
uv run mriseqbench run --experiment experiments/pi_p0_pypulseq.yaml
```

Arms run offline in local Docker images ([recipes](../runtimes/arms/)) with
read-only runtime files; the catalog, judge, credentials and Docker socket are
not mounted, and a missing tool or failed audit stops the run. Author containers
get `arm_runtime.author_memory_mb` (default 2048), one CPU and no swap.

`./bin/simulate [sequence.seq]` sends the exact submitted bytes and public inputs
to a separate CUDA simulator container; it returns diagnostics, not scores.
KomaMRI simulator images are finished on a GPU by
[`prewarm_simulator.py`](../runtimes/arms/prewarm_simulator.py), which caches the
CUDA compilation in a pkgimage (`SimWarm`); the adapters refuse an unwarmed image.
Select its GPU with `MRISEQBENCH_SIMULATOR_GPU`. At most `concurrency.simulators`
(default 4) simulations run at once per campaign. Simulation counts against agent
time; waiting for a free simulator slot, like trusted grading, does not (recorded
as `simulator_queue_wall_s`). Julia arms share one persistent
session through `./bin/julia` (use `includet` for Revise edits); package loading
happens before timing starts. MCP arms (`tool_interface: julia_mcp`) instead
run Julia only through the `julia_eval` tool of the vendored, pinned
[aplavin/julia-mcp](../vendor/julia-mcp/) server (MIT); Pi gets file tools but no
shell, and benchmark controls are called from Julia with `run(...)`. Their author
images add the MCP layer from `runtimes/arms/julia-mcp.Dockerfile`.

Model runs need `agent.model` and `api_proxy.provider_endpoint`. The proxy pins
the endpoint (no fallbacks), owns sampling (default temperature 0.6, top-p 0.95;
`null` omits either for endpoints that reject it), medium reasoning and the
output cap (`max_output_tokens`; by default the endpoint's own maximum recorded
in `harness/pi_models.json`, sent explicitly because an omitted cap gets a
smaller provider default), disables response caching and
records the effective request settings in `api.jsonl`. For `anthropic/*` models it drops `top_k` and replaces
client prompt-cache breakpoints with one automatic 5-minute breakpoint. `concurrency.agents` and
`concurrency.graders` bound active trials and grading (default 1 each).

Workspaces provide `./bin/status` (charged time, remaining seconds, submissions)
and save every feedback response under `feedback/`; `./bin/feedback` re-prints
the last disclosed response without grading.
