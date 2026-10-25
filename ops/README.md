# Ultron campaigns

A campaign is one `mriseqbench run` process under systemd. Rerunning the same
`--out` resumes it, so restarts, reboots and crashes need no manual recovery.
Campaign `<name>` writes its results to
`/ssd-pool/kfareed/MRISeqBench-results/campaigns/<name>`.

## One-time setup

```bash
git clone <repo> ~/mriseqbench/src && cd ~/mriseqbench/src   # or rsync a checkout
uv sync --locked && julia --project=. -e 'using Pkg; Pkg.instantiate()'
bash runtimes/bart/install.sh
# Arm images and rootless Docker tools live outside the checkout (ext4; /var/tmp
# has no age-based cleanup on Ultron, and overlay2 does not suit the ZFS SSD).
ln -s /var/tmp/kfareed-mriseqbench-docker-20261016 ~/mriseqbench/docker-data
ln -s ~/mriseqbench/julia-alltasks-3x-20261016/infra/rootless-tools ~/mriseqbench/rootless-tools
mkdir -p ~/.config/systemd/user ~/mriseqbench/campaigns
mkdir -p /ssd-pool/kfareed/MRISeqBench-results/campaigns
cp ops/*.service ~/.config/systemd/user/ && systemctl --user daemon-reload
systemctl --user enable --now mriseqbench-dockerd
DOCKER_HOST=unix:///run/user/$(id -u)/mriseqbench-docker.sock docker images
```

## Run a campaign

`~/mriseqbench/campaigns/<name>.env` (chmod 600):

```bash
EXPERIMENT=experiments/run_gpt6luna_p2_pypulseq_koma.yaml
OPENROUTER_API_KEY=...
# GPU 3 (TITAN RTX) for the grader and the simulator.
CUDA_VISIBLE_DEVICES=GPU-b67755a7-71f0-087b-9877-422e4b48b818
MRISEQBENCH_SIMULATOR_GPU=GPU-b67755a7-71f0-087b-9877-422e4b48b818
# Grader runtime (systemd does not expand $HOME): Julia 1.12.6 first on PATH,
# the CUDA BART build, CUDA KomaMRI replay.
PATH=/home/kfareed/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin:/home/kfareed/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
MRISEQBENCH_BART=/home/kfareed/.local/share/mriseqbench-bart-v1.0.00/bart
MRISEQBENCH_KOMA_DEVICE=cuda
JULIA_CUDA_HARD_MEMORY_LIMIT=8GiB
JULIA_PKG_PRECOMPILE_AUTO=0
OPENBLAS_NUM_THREADS=1
# Canary; leave empty for the full campaign.
RUN_ARGS=--cases gre_t1w* --repeats 1
```

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

For local GPU grading, create a campaign-specific JSON file on Ultron before
launch, containing UUIDs from `nvidia-smi --query-gpu=index,uuid --format=csv`:

```json
[
  {"uuid": "GPU-c24ca001-f55b-f1a9-bb74-41d0d9087e68", "slots": 2},
  {"uuid": "GPU-b67755a7-71f0-087b-9877-422e4b48b818", "slots": 1}
]
```

Add these environment settings before starting a campaign with the updated
runner and worker:

```bash
MRISEQBENCH_EVALUATION_WORKER=/home/kfareed/mriseqbench/src/ops/local_gpu_worker.py
MRISEQBENCH_GRADING_GPU_CONFIG=/ssd-pool/kfareed/MRISeqBench-results/campaign-support/<name>/grading-gpus.json
```

The total slots in this file determines grading concurrency. To add GPU 1 while
the campaign runs, execute on Ultron:

```bash
python3 ops/set_grading_pool.py \
  --config /ssd-pool/kfareed/MRISeqBench-results/campaign-support/<name>/grading-gpus.json \
  --gpu 2=2 --gpu 3=1 --gpu 1=1
```

Each `--gpu` accepts an index or UUID and a slot count. Supply the complete
desired pool: omit a device to drain it, or use `--gpu 1=0` to stop new work on
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
    "key": "/home/kfareed/.ssh/grading-key",
    "root": "/absolute/path/to/frozen/source",
    "gpu_name": "NVIDIA H100 PCIe",
    "qualified_at": "2026-10-24T23:00:00Z"
  }
}
```

Install the identical frozen Python/Julia project, BART commit and native CUDA
precision on the host. Run the same native CUDA and BART controls before adding
its UUID with `--gpu GPU-remote-uuid=4` alongside every retained local device.
The operator command accepts UUIDs from this qualified registry. Register the
host before increasing capacity; drain its pool entry before removing the host.
SSH host keys must already be trusted on Ultron.

`remote_gpu_dispatch.py` transfers only grading inputs and checkpoints to
identical scratch paths, supervises the scientific worker, and kills its whole
process group if the SSH input channel closes. Downloads go to a staging folder
on Ultron; every file must match the remote SHA-256 manifest before publishing
the verdict. Transport failures remain incomplete evaluations. Live Ultron
worker logs are never overwritten by scratch copies. Verified outputs return to
Ultron before deleting remote scratch files. Preserve failed transfers for
recovery; do not treat them as agent failures. Cloud leases also need a verified
external termination guard and an admission-drain deadline; the legacy RunPod
CLI's `--terminate-after` flag does not enforce a deadline.
