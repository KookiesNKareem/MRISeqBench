#!/usr/bin/env python3
"""Atomically replace live grading capacity; execute on the grading host."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from remote_gpu_dispatch import remote_hosts

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mriseqbench.harness.grading_pool import atomic_json, slot_count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--gpu",
        action="append",
        metavar="INDEX_OR_UUID=SLOTS",
        help="repeat for the complete desired GPU pool; omit GPUs to drain them",
    )
    mode.add_argument(
        "--slots", type=int, help="capacity for a non-GPU grading-slots.json file"
    )
    args = parser.parse_args(argv)
    if not args.config.exists():
        parser.error("config must already exist (use the running campaign's config)")
    try:
        try:
            previous = json.loads(args.config.read_text())
        except ValueError:
            previous = None  # The explicit mode can also repair malformed JSON.
        if args.gpu:
            inventory = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                text=True,
                timeout=10,
            )
            indices = dict(
                line.strip().split(", ", 1) for line in inventory.splitlines()
            )
            known = set(indices.values()) | set(remote_hosts(args.config))
            devices = []
            for value in args.gpu:
                key, count = value.rsplit("=", 1)
                uuid = indices.get(key, key)
                if uuid not in known:
                    raise ValueError(f"GPU {key!r} is absent from nvidia-smi")
                devices.append({"uuid": uuid, "slots": slot_count(int(count))})
            # Validate before replacing the live file, including duplicate UUIDs
            # and the total slot bound. Never expose a partial update to waiters.
            if len({d["uuid"] for d in devices}) != len(devices):
                raise ValueError("duplicate grading GPU")
            slot_count(sum(d["slots"] for d in devices))
            if previous is not None and not isinstance(previous, list):
                raise ValueError("--gpu needs a GPU pool list, not a slots object")
            settings = devices
        else:
            if previous is not None and (
                not isinstance(previous, dict) or set(previous) != {"slots"}
            ):
                raise ValueError("--slots needs a non-GPU grading-slots.json file")
            settings = {"slots": slot_count(args.slots)}
        atomic_json(args.config, settings)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    print(
        f"Updated {args.config}; queued work will reallocate, active grades will drain."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
