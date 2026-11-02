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

Koma uses `KomaMRI.reconstruct_with_labels`; komamripy invokes its public
`reconstruct_with_labels` alias of the same implementation after simulation. The simulator project/source must provide that API (KomaMRI 0.15 or
newer). Alongside `signal.h5`, they return `reconstruction.h5` and
`reconstruction.json`. Each JSON image entry identifies its HDF5 group, acquisition
labels, source label values, shape, named axes and trajectory rescaling. HDF5
`image_real`/`image_imag` datasets expose `(x, y, z, echo, coil, repetition)`
in Python/h5py; Julia HDF5 readers reverse that dimension order. The client saves
a central-partition, coil-RSS magnitude preview for each label group, showing all
echoes on a shared magnitude scale across the returned label groups. The complete complex arrays retain all partitions and coils.

For mapping, author `REP`/`SET` labels for preparation/TI frames and `ECO` for
echoes. Koma's default policy separates repetitions, sets and slices, retains the
echo axis, and combines segments. Missing labels are not inferred from task names.
For segmented mapping, keep `REP`/`SET` fixed within each preparation frame,
use `SEG` for segments and `LIN` for phase-encoding lines. Supply
`Nx`/`Ny`/`Nz` and `FOV` definitions for reconstruction geometry. A failed
reconstruction is reported in `reconstruction.json` while raw signal is retained.
MRzero supports explicit CPU or CUDA selection with `MRISEQBENCH_SIMULATOR_DEVICE`;
it does not silently fall back from a requested CUDA device. Metal is not wired
for MRzero. MRzero uses its native `MRzeroCore.reco_adjoint`, reconstructing each observed
`SLC`/`SET`/`REP`/`AVG`/`PHS`/`ECO` group separately with the same HDF5/JSON
image output layout. Contributions are accumulated in bounded sample chunks to
avoid allocating the complete sample-by-voxel DFT matrix. It also needs explicit
`Nx`/`Ny`/`Nz` and `FOV` definitions.

Read a complex image in Python with `h5py` from `image_001/image_real` plus
`1j * image_001/image_imag`. In Julia, use `KomaMRIFiles.HDF5.h5read` for
each dataset and reverse the six dimensions with `permutedims(data, (6,5,4,3,2,1))`.
Use the JSON metadata to select frames and echoes before fitting a relaxation curve.

Use the images to inspect the candidate and rerun after edits. This direct
reconstruction uses only public inputs; it does not expose the judge's reference,
NRMSE or pass/fail score.

KomaMRI simulator images are finished on a GPU by
[`prewarm_simulator.py`](../runtimes/arms/prewarm_simulator.py), which caches the
CUDA compilation in a pkgimage (`SimWarm`); the adapters refuse an unwarmed image.
Select its GPU with `MRISEQBENCH_SIMULATOR_GPU`. At most `concurrency.simulators`
(default 4) simulations run at once per campaign. Simulation counts against agent
time; waiting for a free simulator slot, like trusted grading, does not (recorded
as `simulator_queue_wall_s`). Julia arms share one persistent
session through `./bin/julia` (run `include("design.jl"); build_sequence()` after
each edit); package loading
happens before timing starts. MCP arms (`tool_interface: julia_mcp`) instead
run Julia only through the `julia_eval` tool of the vendored, pinned
[aplavin/julia-mcp](../vendor/julia-mcp/) server (MIT); Pi gets file tools but no
shell, and benchmark controls are called from Julia with `run(...)`. Their author
images add the MCP layer from `runtimes/arms/julia-mcp.Dockerfile`.

For native runs, the simulator adapters accept
`MRISEQBENCH_SIMULATOR_DEVICE=cpu|metal|cuda` (default `cuda`). Metal requires
a native macOS process and a functional Metal.jl environment; Linux Docker
cannot expose it. The adapters verify the requested backend and report it in
`simulation.json`. `MRISEQBENCH_SIMULATOR_PROJECT` selects a native Julia
project; warm persistent native workers before starting the agent clock.

The trusted evaluator accepts `MRISEQBENCH_KOMA_DEVICE=cpu|metal|cuda` and
`MRISEQBENCH_KOMA_PROJECT` (defaults: CUDA and the benchmark project). Grading
defaults to Float64 and rejects a selected backend without Float64 support.
For Apple Metal grading, explicitly select `MRISEQBENCH_KOMA_PRECISION=f32`.
This changes simulation precision; record the backend and precision with results
and qualify it against Float64 reference replays. CPU and CUDA grading retain
Float64. Sampling requirements and acceptance thresholds are independent of this
precision selector.
`MRISEQBENCH_BART_DEVICE=cpu|cuda` selects CPU or CUDA reconstruction; point
`MRISEQBENCH_BART` at the qualified BART v1.0.00 executable. Native backend and
runtime choices must be recorded alongside results, separately from CUDA runs.

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

## Reproducible Julia author builds

Every Julia author project includes FFTW 1.10.0, whether PulseDesigner, Unitful
or a simulator is enabled. NumPy FFT is available in Python arms. Julia's
Statistics standard library still needs `using Statistics` before calling `mean`.

Include `sanitize_julia.py` and `koma-block-duration.patch` alongside the pinned
KomaMRIBase/KomaMRIFiles source folders in the Julia Docker build context. The
sanitizer applies the patch before compilation and API stripping. It propagates
scanner definitions into assembled blocks and uses scanner-aware duration for
RF ringdown and ADC dead time. Explicit Duration constraints remain enforced.
The raw ADC sampling convention is unchanged. An incompatible patch fails the
build; already-patched source is accepted. Simulator images inherit this author
source. Keep the patch until the pinned upstream revision contains the fix.

To build an optional native author sysimage, use a separate builder project with
PackageCompiler installed and a prepared author project for exactly one arm:

```sh
julia --project=/absolute/path/to/builder ops/build_author_sysimage.jl \
  /absolute/path/to/author-project /absolute/path/to/output j2_pulseq_julia_unitful
```

The output includes `author.dylib`/`author.so` and a TOML receipt with Julia,
manifest, session-source and sysimage hashes. The workload compiles generic
authoring, FFT, read/write, trajectory and timing/hardware APIs; it contains no
task solution or judge input. Start the matching persistent author with `-J` and
that project's manifest. A sysimage reduces startup/JIT work; new user methods
can still compile on first execution.

The simulator now checks active RF roles before reconstructing: Undefined RF
produces an actionable `rf_roles` failure asking for explicit use labels. It does
not infer pulse roles or add a new geometry-definition check. Trigger resolution
uses the same cardiac timing in Julia and komamripy. The author still supplies
physical encoding and acquisition labels for each frame.
