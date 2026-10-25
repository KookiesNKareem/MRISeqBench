#!/bin/bash
# From the campaign host: push the source tree to a rented GPU host and run
# ops/gpu_host_bootstrap.sh there. Usage: push_gpu_host.sh user@host [port] [key]
set -euo pipefail
HOST=$1; PORT=${2:-22}; KEY=${3:-$HOME/.ssh/mriseqbench-lambda}
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $PORT -i $KEY"
SRC=/home/kfareed/mriseqbench/src
$SSH "$HOST" 'S=$(command -v sudo || true); $S mkdir -p /home/kfareed/mriseqbench/src && $S chown -R $(id -u):$(id -g) /home/kfareed'
rsync -a --delete --exclude .venv --exclude .git --exclude __pycache__ --exclude .ruff_cache --exclude .pytest_cache -e "$SSH" "$SRC/" "$HOST:$SRC/"
$SSH "$HOST" "bash $SRC/ops/gpu_host_bootstrap.sh" 2>&1 | tail -40
