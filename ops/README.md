# Ultron campaigns

A campaign is one `mriseqbench run` process under systemd. Rerunning the same
`--out` resumes it, so restarts, reboots and crashes need no manual recovery.

## One-time setup

```bash
# Reuse the existing arm images and rootless tools from their neutral paths.
# The old campaign dockerd must be stopped (it is).
mv ~/mriseqbench/julia-alltasks-3x-20261016/infra/docker-data ~/mriseqbench/docker-data
mv ~/mriseqbench/julia-alltasks-3x-20261016/infra/rootless-tools ~/mriseqbench/rootless-tools

git clone <repo> ~/mriseqbench/src && cd ~/mriseqbench/src   # or rsync a checkout
uv sync --locked && julia --project=. -e 'using Pkg; Pkg.instantiate()'
bash runtimes/bart/install.sh
mkdir -p ~/.config/systemd/user ~/mriseqbench/campaigns ~/mriseqbench/runs
cp ops/*.service ~/.config/systemd/user/ && systemctl --user daemon-reload
systemctl --user enable --now mriseqbench-dockerd
DOCKER_HOST=unix:///run/user/$(id -u)/mriseqbench-docker.sock docker images
```

## Run a campaign

`~/mriseqbench/campaigns/<name>.env` (chmod 600):

```bash
EXPERIMENT=experiments/pi_a2_pypulseq.yaml
OPENROUTER_API_KEY=...
CUDA_VISIBLE_DEVICES=GPU-b67755a7-71f0-087b-9877-422e4b48b818   # GPU 3 (TITAN RTX), grader
MRISEQBENCH_SIMULATOR_GPU=GPU-b67755a7-71f0-087b-9877-422e4b48b818
RUN_ARGS=--cases gre_t1w* --repeats 1                          # canary; empty = full
```

```bash
systemctl --user start mriseqbench@<name>       # launch or resume
journalctl --user -u mriseqbench@<name> -f      # progress
systemctl --user stop mriseqbench@<name>        # graceful stop; start resumes
```

The experiment must set `api_proxy.max_campaign_usd` (and optionally
`max_trial_usd`). After a canary, clear `RUN_ARGS` and start again: the canary's
trials are reused. Exit 1 (resumable trials left) restarts after 10 minutes;
exit 2 (configuration, or code/catalog changed since launch) and exit 3 (spend
cap or provider account exhausted) do not. To change code or the experiment,
use a new campaign name.
