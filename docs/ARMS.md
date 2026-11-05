# Authoring and simulation arms

Each arm selects the authoring tools and optional agent-side simulator available
to the agent. The trusted judge and submission allowance are shared.

Simulator modifiers follow the abstract: **S1** = MRzero/Python, **S2** =
KomaMRI/Julia, and **S3** = KomaMRI through komamripy/Python. No simulator has
no S modifier. These modifiers describe agent access; the shared judge always
uses KomaMRI. P/J arm names remain authoring configuration names; the phantom
label **P1** is a separate axis. Full mapping: [labels](LABELS.md).

Figure 1 of the [MRISeqBench Abstract](https://docs.google.com/document/d/1ssPmltYEzQQNVuKCOyroliBeHsXi6e7JOip7OZzillc/edit)
is authoritative for public arm labels and tool names. Historical manifest IDs
and saved campaign names are retained for reproducibility; their numeric prefixes
must not be interpreted as current labels.

| Public arm | Authoring and simulator tools | Manifest reference |
|---|---|---|
| P0 | PyPulseq | `p0_pypulseq@1` |
| P0 S1 | PyPulseq + MRzero | `p1_pypulseq_mrzero@1` |
| P0 S3 | PyPulseq + komamripy | `p2_pypulseq_koma@1` |
| J0 | PulseqBase.jl | `j0_koma_base@1` |
| J1 | Pulseq.jl | `j1_pulseq_julia@1` |
| J2 | Pulseq.jl + Unitful.jl | `j2_pulseq_julia_unitful@1` |
| J3 | PulseqBase.jl + Julia MCP | `j6_koma_base_julia_mcp@1` |
| J4 S2 | Pulseq.jl + Julia MCP + KomaMRI.jl | `j7_pulseq_julia_mcp_koma@1` |

Additional CLI simulator combinations remain available outside Figure 1:

| Public arm | Tools | Historical manifest reference |
|---|---|---|
| J0 S2 | PulseqBase.jl + KomaMRI.jl | `j3_koma_base_koma@1` |
| J1 S2 | Pulseq.jl + KomaMRI.jl | `j4_pulseq_julia_koma@1` |
| J2 S2 | Pulseq.jl + Unitful.jl + KomaMRI.jl | `j5_pulseq_julia_unitful_koma@1` |

J4 S2 combines Pulseq.jl, Julia MCP, and KomaMRI.jl simulator access.
The historical `j4_pulseq_julia_koma` configuration is **J1 S2**, which uses CLI
Julia tools. Historical J7 results are **J4 S2** results under the current labels.
The pinned runtime internally names the PulseqBase.jl toolset
`KomaMRIBase`/`KomaMRIFiles`, and the Pulseq.jl authoring feature `PulseDesigner`.
Relabeling does not change those installed packages or runtime capabilities.

Experiment `arm` fields accept public references such as `J1 S2@1`, `J4 S2@1`,
and `P0 S3@1`, normalizing them to historical IDs before execution.
For J1 S2, use `experiments/pi_j4_pulseq_julia_koma.yaml`; for J4 S2,
use `experiments/pi_j4_s2.yaml`. Existing experiment filenames
are historical; each template identifies its current public label.

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
