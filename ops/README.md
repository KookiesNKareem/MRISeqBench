# Campaign operations

A campaign is one `mriseqbench run` process under systemd. Rerunning the same
`--out` resumes it, so restarts, reboots and crashes need no manual recovery.
Set `RESULTS_DIR` to an absolute directory outside the source checkout for each
campaign. Keep deployment credentials and host inventories outside version control.

## One-time setup

```bash
git clone <repo> ~/mriseqbench/src && cd ~/mriseqbench/src   # or rsync a checkout
uv sync --locked && julia --project=. -e 'using Pkg; Pkg.instantiate()'
bash runtimes/bart/install.sh
# Install rootless Docker and its networking dependencies before starting the service.
# Use a filesystem supported by Docker's overlay2 storage driver.
mkdir -p ~/mriseqbench/docker-data ~/.config/systemd/user ~/mriseqbench/campaigns
cp ops/*.service ~/.config/systemd/user/ && systemctl --user daemon-reload
systemctl --user enable --now mriseqbench-dockerd
DOCKER_HOST=unix:///run/user/$(id -u)/mriseqbench-docker.sock docker images
```

## Run a campaign

`~/mriseqbench/campaigns/<name>.env` (chmod 600):

```bash
EXPERIMENT=experiments/pi_p2_pypulseq_koma.yaml
RESULTS_DIR=/absolute/path/to/results/my-campaign
OPENROUTER_API_KEY=...
# Replace GPU-EXAMPLE with a UUID reported by nvidia-smi.
CUDA_VISIBLE_DEVICES=GPU-EXAMPLE
MRISEQBENCH_SIMULATOR_GPU=GPU-EXAMPLE
# Set absolute paths to your pinned Julia installation and CUDA BART build.
# systemd environment files do not expand $HOME.
PATH=/absolute/path/to/julia/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
MRISEQBENCH_BART=/absolute/path/to/bart
MRISEQBENCH_KOMA_DEVICE=cuda
JULIA_CUDA_HARD_MEMORY_LIMIT=8GiB
JULIA_PKG_PRECOMPILE_AUTO=0
OPENBLAS_NUM_THREADS=1
# Canary; leave empty for the full campaign.
RUN_ARGS=--cases gre_t1w* --repeats 1
```

Fill in the experiment template's model and provider endpoint before launching.
All example paths and GPU UUIDs must be replaced with your installation values.

systemd keeps a trailing `# ...` as part of the value, so comments in this
file go on their own lines.

```bash
systemctl --user start mriseqbench@<name>       # launch or resume
journalctl --user -u mriseqbench@<name> -f      # progress
systemctl --user stop mriseqbench@<name>        # graceful stop; start resumes
```

After a canary, clear `RUN_ARGS` and start again: the canary's
trials are reused. Exit 1 (resumable trials left) and exit 4 (Docker daemon
unavailable) restart after 10 minutes;
exit 2 (configuration, or code/catalog changed since launch) and exit 3 (provider
account exhausted) do not. To change code or the experiment,
use a new campaign name.

The API proxy retries OpenRouter's temporary HTTP 402
`in_flight_budget_exhausted` responses with the provider's `Retry-After` delay
(up to 300 seconds per wait), excluding those waits from agent time. Persistent
temporary rejection leaves the affected trial unscored rather than stopping
new campaign work. Genuine account-credit exhaustion remains a campaign stop.

## Change grading capacity during a run

The runner reloads its grading pool while submissions wait. Increasing capacity
admits queued jobs in FIFO order. Reducing capacity, removing a GPU, or setting
its slots to zero drains existing grades; pending work is assigned to the
remaining pool. Running evaluations finish on their original device with their
existing checkpoints, and agent sessions continue without a restart.

For local GPU grading, create a campaign configuration file before launch,
containing UUIDs from `nvidia-smi --query-gpu=index,uuid --format=csv`:

```json
[
  {"uuid": "GPU-EXAMPLE-SECOND", "slots": 2},
  {"uuid": "GPU-EXAMPLE", "slots": 1}
]
```

Add these environment settings before starting a campaign with the updated
runner and worker:

```bash
MRISEQBENCH_EVALUATION_WORKER=/absolute/path/to/MRISeqBench/ops/local_gpu_worker.py
MRISEQBENCH_GRADING_GPU_CONFIG=/absolute/path/to/campaign-config/grading-gpus.json
```

The total slots in this file determines grading concurrency. To change the GPU
allocation while the campaign runs, execute on the campaign host:

```bash
python3 ops/set_grading_pool.py \
  --config /absolute/path/to/campaign-config/grading-gpus.json \
  --gpu GPU-EXAMPLE=2 --gpu GPU-EXAMPLE-SECOND=1
```

Each `--gpu` accepts an index or UUID and a slot count. Supply the complete
desired pool: omit a device to drain it, or use `--gpu GPU-EXAMPLE=0` to stop new work on
that device. Slot counts and their total must be integers from 0 to 64. Qualify
new hardware using the same CUDA/native and reconstruction controls as the
existing graders. The command validates device identities and atomically
replaces the pool file. Waiting dispatchers reload it within approximately
0.2 seconds; each dispatch receipt records the chosen device and pool settings.

Campaigns without a GPU pool create `<campaign>/grading-slots.json` with the
experiment's initial grader count. Resize them with:

```bash
python3 ops/set_grading_pool.py --config <campaign>/grading-slots.json --slots 4
```

Zero pauses new grading while active work drains. Missing or malformed live
configuration also pauses admission until repaired. Pool status, including
active/waiting counts and configuration errors, is written to
`<campaign>/grading-pool-status.json`; operational changes are recorded in
`<campaign>/grading-pool-changes.jsonl` and campaign accounting. These pool
changes retain the experiment/source identity, verdicts, and agent allowance.

Processes started with an older runner retain their old scheduler. Use the
updated source and worker when starting the next campaign; replacing source
files does not upgrade an already running Python process.

Qualified SSH grading hosts can join the same GPU pool without restarting
agents. Place `remote-grading-hosts.json` beside `grading-gpus.json`, keyed by
actual remote GPU UUID:

```json
{
  "GPU-remote-uuid": {
    "endpoint": "root@gpu-host.example",
    "port": 22,
    "key": "/absolute/path/to/grading-key",
    "root": "/absolute/path/to/frozen/source",
    "gpu_name": "NVIDIA H100 PCIe",
    "qualified_at": "<qualification timestamp>"
  }
}
```

Install the identical frozen Python/Julia project, BART commit and native CUDA
precision on the host. Run the same native CUDA and BART controls before adding
its UUID with `--gpu GPU-remote-uuid=4` alongside every retained local device.
The operator command accepts UUIDs from this qualified registry. Register the
host before increasing capacity; drain its pool entry before removing the host.
SSH host keys must already be trusted on the campaign host. The remote SSH
environment must make the pinned Julia executable available on `PATH` and
provide `MRISEQBENCH_BART` if BART is not in its standard installation location.
The worker inherits the remote account's home directory and Julia depot.

`remote_gpu_dispatch.py` transfers only grading inputs and checkpoints to
identical scratch paths, supervises the scientific worker, and kills its whole
process group if the SSH input channel closes. Downloads go to a staging folder
on the campaign host; every file must match the remote SHA-256 manifest before
publishing the verdict. Transport failures remain incomplete evaluations. Live
worker logs are never overwritten by scratch copies. Verified outputs return to
the campaign host before remote scratch files are deleted. Preserve failed
transfers for recovery; do not treat them as agent failures. Manage cloud lease
deadlines independently of job execution and drain admission before retiring a host.

## Native macOS author isolation

Native pilots must wrap both the agent subprocess and its persistent Julia author
with `mriseqbench.harness.native_boundary.isolate_judge`. The private checkout,
controls, reference arrays and transient native-worker files are unreadable;
only the author workspace and declared public libraries are exceptions. Public
libraries are read-only. Among local services, only that trial's authenticated
feedback/simulation port is reachable; dashboard and grader IPC access is denied.
The trusted judge remains outside this boundary. The wrapper fails closed when
the required macOS sandbox is unavailable. Never replace it with prompt-only rules.


## Native agent simulation feedback

For new native campaigns, copy `ops/grading_mcp.py` into the campaign's frozen
runtime and allow the `simulate` MCP tool in simulator-enabled arms. It is
advertised only when the workspace has `bin/simulate`. The matching source
`arm_simulation_client.py` must also be frozen into that campaign.

`grading.simulate` returns actual PNG image content together with the existing
public file/timing/hardware check, the candidate SHA-256 and reconstruction
status. Complete complex images, labels and coordinate arrays stay in
`simulation/`; stdout contains a compact summary. The tool consumes no graded
submission and exposes no private targets, image scores or SAR result. Image
content is returned for every reconstructed label group, including ECO/REP
previews. A candidate edit requires another simulation. Existing frozen campaigns
keep their original adapter and client.

## Native campaign monitor and author sysimages

The relocatable [campaign monitor](monitor/README.md) displays existing trial
progress, simulator previews and judge submissions; it does not schedule work.
The [author sysimage recipe](build_author_sysimage.jl) builds one isolated Julia
author profile at a time and records its source/manifest hashes. The native
pilot's queue manager was process orchestration (launch, recovery, collection),
not an additional model agent. Campaign-specific launchers remain separate from
the reusable monitor.

Persistent grading transport (`mriseqbench.evaluation.warm_service`) and static
GPU batching share the Koma worker implementation. The currently wired backends
are CPU, CUDA/Float64 and Metal/Float32. Linux Docker cannot expose Apple Metal.
Dynamic-field work keeps its conservative integration ranges. The measured
performance gain was qualified on Metal; it is not a CUDA or AMD performance
claim. AMDGPU and oneAPI are not wired into this harness.

## Qualify fixed phantom density before trials

Select a known reference sequence and a resolved case using the new fixed-slab
contract. Keep the backend and precision identical between the two densities:

```sh
MRISEQBENCH_KOMA_DEVICE=metal MRISEQBENCH_KOMA_PRECISION=f32 \
MRISEQBENCH_KOMA_PROJECT=/absolute/path/to/metal-project \
  uv run python ops/qualify_phantom_density.py --case /absolute/path/to/case.json \
  --sequence /absolute/path/to/control.seq --output /absolute/path/to/new-qualification
```

Use CUDA/Float64 or CPU/Float64 on those backends. Put the Julia binary matching
the selected project Manifest.toml first on PATH; do not rely on juliaup's current
default version. This tool starts no agents,
changes no acceptance limits and leaves the evaluator's calibrated flag alone.
Review native image quality as well as density agreement for each task/scanner
profile before marking it calibrated. A stable but wrong sequence is not a
positive control. New fixed profiles are intentionally uncalibrated meanwhile.

## Polling token accounting

Keep polling-only responses visible as a separate category, including their input
and output usage. Cached input is already part of input tokens and is not added
again. An audited response map labels only wholly polling responses; mixed
productive/status responses stay in the other category. This classifies complete
provider responses, not the causal origin of each context token.

```sh
uv run python ops/report_polling_tokens.py --campaign /absolute/path/to/campaign \
  --audit /absolute/path/to/polling-response-audit.csv \
  --output /absolute/path/to/token-categories.csv
```

The audit must cover the same unique trial/response pairs and token totals as
`response-usage.csv`; the command rejects missing or inconsistent rows. It reads
results without changing trials or provider accounting.
