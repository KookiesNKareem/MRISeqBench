# Authoring and simulation arms

Each arm selects installed authoring tools and optional agent-side simulation.
The trusted judge and submission budget are shared across arms.

The Python comparison uses stored arm IDs **P0–P2** in job IDs, workspaces,
results and figures. The versioned catalog references remain compatible:

| Comparison ID | Catalog reference | Authoring | Agent simulator |
|---|---|---|---|
| P0 | `a2_pypulseq@1` | PyPulseq | None |
| P1 | `s2_pypulseq_mrzero@1` | PyPulseq | MRzero |
| P2 | `s3_pypulseq_komamripy@1` | PyPulseq | KomaMRI through komamripy |

These simulator differences are part of the comparison. Agent simulation and
its queue waits count against charged author time; trusted grading does not.

| Arm | Authoring | PulseDesigner | Unitful | Simulator |
|---|---|:---:|:---:|---|
| `a2_pypulseq@1` | PyPulseq | — | — | None |
| `a3_koma_base@1` | KomaMRIBase / KomaMRIFiles | No | No | None |
| `a4_pulseq_julia@1` | KomaMRIBase / KomaMRIFiles | Yes | No | None |
| `a5_pulseq_julia_unitful@1` | KomaMRIBase / KomaMRIFiles | Yes | Yes | None |
| `s2_pypulseq_mrzero@1` | PyPulseq | — | — | MRzero |
| `s3_pypulseq_komamripy@1` | PyPulseq | — | — | KomaMRI / komamripy |
| `s4_koma_base_koma@1` | KomaMRIBase / KomaMRIFiles | No | No | KomaMRI |
| `s5_pulseq_julia_koma@1` | KomaMRIBase / KomaMRIFiles | Yes | No | KomaMRI |
| `s6_pulseq_julia_unitful_koma@1` | KomaMRIBase / KomaMRIFiles | Yes | Yes | KomaMRI |

Configure the model, suite and local images in the matching experiment YAML:

```bash
uv run mriseqbench arms
uv run mriseqbench run --experiment experiments/pi_a2_pypulseq.yaml
```

Restricted arms require local Docker images and run offline with read-only
runtime files. The host catalog, judge, credentials and Docker socket are not
mounted. Missing tools or failed audits stop the run; there is no host fallback.
Build recipes: [`runtimes/arms/`](../runtimes/arms/).

Simulation uses a separate container and fixed adapter receiving exact submitted
Pulseq bytes plus public phantom/field inputs. It returns simulation artifacts,
not judge scores, and cannot execute arbitrary agent programs. Use
`./bin/simulate [sequence.seq]` when the selected arm enables simulation.
All simulator dynamics run on CUDA. A missing GPU, CPU-only simulator image, or
CPU execution receipt fails runtime qualification; no CPU fallback is accepted.
Select one device with `MRISEQBENCH_SIMULATOR_GPU` (or `CUDA_VISIBLE_DEVICES`).
Rootless Docker without an NVIDIA runtime automatically uses device-node access;
it can also be selected explicitly with
`MRISEQBENCH_SIMULATOR_GPU_ACCESS=devices`: only that GPU's device nodes and
read-only CUDA driver libraries are exposed, without restarting the daemon.
Input parsing, graph preparation, and output serialization remain host work.
The Koma simulator image uses Julia 1.12.6 with the native evaluator's pinned
CUDA 6.4.0 compiler stack. Package caches and lazy CUDA artifacts are prepared
during the image build; offline simulation never downloads dependencies.

For an operational runtime change, use one resource snapshot and one GPU smoke
check for each changed simulator. Verify changed files and reuse existing source
and qualification receipts. Do not repeat timed preflight sampling, whole-archive
hash passes, or paid benchmark runs for metadata-only changes. Keep GPU-only
execution, resource caps, financial accounting, and scientific acceptance checks.

Julia arms use a persistent process through `./bin/julia`; permitted packages and
generic warmup load before timing begins. Agent compilation and execution count
against the trial budget. Use `includet("design.jl")` for Revise-based edits.

Restricted arms remain offline; report the selected arm when comparing results.

Author containers have `arm_runtime.author_memory_mb` (default 2048) hard memory
limits, zero swap, one CPU, and a 128 MiB Node heap; audit probes use 768 MiB.
Under rootless Docker, containers run as container root, which maps to the
invoking host user. CUDA simulator containers use the established
2 GiB, one-CPU, zero-swap envelope and writable temporary CUDA caches. Physical
RAM is limited by the container; process virtual/data limits are incompatible
with CUDA's address mappings. The Pi image version must match
`agent.pi_version` (default `0.87.0`) before a model run starts; immutable image
IDs and the verified version are saved in `arm_runtime.json`. Native Pi runners
are also version checked. No image is pulled automatically.

Model runs require an explicit model ID and `api_proxy.provider_endpoint`;
fill both placeholders in the Pi experiment templates. The proxy forces the
chosen endpoint with provider/model fallbacks disabled and parameter support
required. It also owns temperature, top-p, reasoning effort and output limits,
and records effective request settings and the tool-schema hash in `api.jsonl`.
Default settings are temperature 0.6, top-p 0.95, medium reasoning and 16384 output
tokens. Configure the proxy fields to change them; Pi's thinking level follows
the requested configuration, but Pi may normalize its local label using its model
registry. `result.json.pi_settings` records that label separately from the actual
host-enforced API reasoning effort; `api.jsonl` is the effective request evidence.

Automatic compaction defaults off (`agent.auto_compaction: false`), and Pi cache
warming is off to prevent hidden refresh requests. Each trial receives a distinct
host-owned session ID, kept stable across its model calls. The mailbox preserves
the approved session header and response generation/cache headers. The host sends
`X-OpenRouter-Cache: false` on every request; an unexpected response-cache HIT
revokes access and leaves the trial unscored. Provider prompt caching remains
enabled, with cached-input fraction and actual charges in the accounting receipt.

`concurrency.agents` and `concurrency.graders` independently bound active trials
and trusted grading. Both default to one. Inputs are materialized serially before
trial execution. Queued grading time is excluded from the agent budget; ordinary
tools and agent simulation still count. Results are written atomically per trial;
`results.jsonl` is rebuilt from them. An infrastructure failure leaves that trial
resumable without inventing a verdict while other trials continue; see
[run storage](RUN_STORAGE.md). Keep both limits at one on the current 2 GiB Colima
VM; measure memory and throughput before increasing them on a larger worker.

Every prepared agent workspace includes `./bin/status` for charged time, remaining
seconds, grading credit and submission allowance. It works during grading and
does not require a sequence or consume a submission. Multi-attempt prompts advise
an initial submission in the first half of the budget; single-shot prompts reserve
time for final checks without imposing that checkpoint.

Feedback commands save the complete JSON under `feedback/` before printing a
compact summary with failed checks first. `feedback/latest-submit.json` is the
latest submission response; `--full` prints the full payload. If a client loses
the response, `./bin/feedback` retrieves the last disclosed submission response
(or the last basic-check response before any submission), without new grading.
This route never exposes a blind final verdict. The same authenticated commands
work through the offline mailbox and gateway.

`ARM_API.md` documents the selected package's imports, units, event timing and
Pulseq export conventions. `EVALUATION.md` distinguishes the independent reference
phase from actual submitted ADC demodulation and states that complete excitation
history and acquired transients are evaluated. These are interface descriptions;
scientific thresholds and submission limits are unchanged.
