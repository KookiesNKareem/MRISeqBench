"""Live operational admission; model/scientific settings are never writable here."""

import argparse
import json
import os
import threading
import time
import uuid
from collections import deque
from datetime import UTC, datetime
from pathlib import Path


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(
        path.name + f".{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def validate(value, ceiling=8):
    if set(value) != {"revision", "remote_slots", "admit_new", "routing"}:
        raise ValueError("Only simulation admission, routing and capacity are writable")
    if type(value["remote_slots"]) is not int or not 0 <= value["remote_slots"] <= ceiling:
        raise ValueError(f"remote_slots must be between 0 and the qualified ceiling {ceiling}")
    if type(value["admit_new"]) is not bool or value["routing"] != "runpod":
        raise ValueError("Invalid admission or unqualified simulation route")
    if type(value["revision"]) is not int or value["revision"] < 1:
        raise ValueError("Invalid control revision")
    return value


class LiveSlots:
    """FIFO admission with live shrink/drain, preserving current holders.

    Callers keep their existing excluded-queue context around acquire().
    Changing capacity never changes an author allowance or kills a job.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.condition = threading.Condition()
        self.waiters = deque()
        self.holders = {}
        self.broken = threading.Event()
        self.free = self
        self.reapers = []
        self.config = None
        self.hosts = []
        self.error = None
        self.stopped = threading.Event()
        with self.condition:
            self._refresh()
        self.watcher = threading.Thread(
            target=self._watch, daemon=True, name="simulation-live-control"
        )
        self.watcher.start()

    def _watch(self):
        while not self.stopped.wait(0.25):
            with self.condition:
                self._refresh()
                self.condition.notify_all()

    def close(self):
        self.stopped.set()
        self.watcher.join(1)

    def _refresh(self):
        try:
            registry = self.directory / "simulation-hosts.json"
            hosts = []
            if registry.exists():
                hosts = json.loads(registry.read_text())["hosts"]
                assert hosts and len({h["id"] for h in hosts}) == len(hosts)
                for host in hosts:
                    proof = json.loads(Path(host["qualification"]).read_text())
                    if not (proof.get("passed") and proof.get("eight_concurrent_CUDA_slots")):
                        raise ValueError("A registered simulation host is not qualified")
                    if host["slots"] != 8 or not host["host"].get("endpoint"):
                        raise ValueError("Invalid qualified simulation host")
                # Registry changes cannot strand or reassign an active lease.
                identities = {h["id"] for h in hosts}
                if any(h.get("host_id") not in identities for h in self.holders.values()):
                    raise ValueError("Drain a host's existing jobs before removing it")
                by_id = {h["id"]: h for h in hosts}
                if any(by_id[h["host_id"]]["host"] != h["host"] for h in self.holders.values()):
                    raise ValueError("An active simulation host's identity cannot change")
            config = validate(
                json.loads((self.directory / "simulation-control.json").read_text()),
                ceiling=sum(h["slots"] for h in hosts) if hosts else 8,
            )
            qualification = json.loads(
                (self.directory / "simulation-qualification.json").read_text()
            )
            if not qualification.get("passed"):
                raise ValueError("Simulation host qualification is incomplete")
            self.config, self.error = config, None
            self.hosts = hosts
            if self.broken.is_set():
                self.error = getattr(self, "fault", "Simulation cleanup is unconfirmed")
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            self.error = str(exc)  # Fail closed; existing holders continue.
        self._publish()

    def _publish(self):
        atomic(
            self.directory / "simulation-control-state.json",
            {
                "at": datetime.now(UTC).isoformat(),
                "controller_pid": os.getpid(),
                "applied_revision": (self.config or {}).get("revision"),
                "config": self.config,
                "active": len(self.holders),
                "waiting": len(self.waiters),
                "error": self.error,
                "broken": self.broken.is_set(),
                "jobs": list(self.holders.values()),
                "qualified_capacity": sum(h["slots"] for h in self.hosts) if self.hosts else 8,
            },
        )

    def acquire(self, cancelled):
        ticket = uuid.uuid4().hex
        with self.condition:
            self.waiters.append(ticket)
            try:
                while not (cancelled() or self.broken.is_set()):
                    self._refresh()
                    if (
                        not self.error
                        and self.config["admit_new"]
                        and self.waiters[0] == ticket
                        and len(self.holders) < self.config["remote_slots"]
                    ):
                        placements = [
                            (host, i)
                            for host in self.hosts
                            for i in range(host["slots"])
                            if (host["id"], i) not in {
                                (h.get("host_id"), h["slot"]) for h in self.holders.values()
                            }
                        ]
                        if self.hosts and not placements:
                            self.condition.wait(0.25)
                            continue
                        host, physical_slot = placements[0] if placements else (None, None)
                        self.holders[threading.get_ident()] = {
                            "lease": ticket,
                            "route": "runpod",
                            "acquired_at": time.time(),
                            "slot": physical_slot if host else next(
                                i
                                for i in range(8)
                                if i not in {h["slot"] for h in self.holders.values()}
                            ),
                            **({"host_id": host["id"], "host": host["host"]} if host else {}),
                        }
                        return True
                    self.condition.wait(0.25)
                return False
            finally:
                self.waiters.remove(ticket)
                self._publish()
                self.condition.notify_all()

    def release(self):
        with self.condition:
            if self.holders.pop(threading.get_ident(), None) is None:
                raise RuntimeError("Simulation release without its admission lease")
            self._publish()
            self.condition.notify_all()

    def reap(self, name):
        # The remote adapter must verify process-group cleanup before releasing.
        self.broken.set()
        with self.condition:
            self.error = f"Remote cleanup unconfirmed: {name}"
            self.fault = self.error
            self._publish()
            self.condition.notify_all()

    def drain(self):
        with self.condition:
            while self.holders:
                self.condition.wait(0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    update = sub.add_parser("set")
    update.add_argument("--remote-slots", type=int)
    update.add_argument("--admit-new", choices=["true", "false"])
    sub.add_parser("drain")
    args = parser.parse_args()
    if args.command == "status":
        print((args.directory / "simulation-control-state.json").read_text())
        return
    path = args.directory / "simulation-control.json"
    registry = args.directory / "simulation-hosts.json"
    ceiling = sum(h["slots"] for h in json.loads(registry.read_text())["hosts"]) if registry.exists() else 8
    before = validate(json.loads(path.read_text()), ceiling)
    after = {**before, "revision": before["revision"] + 1}
    if args.command == "drain":
        after["admit_new"] = False
    else:
        if args.remote_slots is not None:
            after["remote_slots"] = args.remote_slots
        if args.admit_new is not None:
            after["admit_new"] = args.admit_new == "true"
    validate(after, ceiling)
    atomic(path, after)
    with (args.directory / "control-audit.jsonl").open("a") as log:
        log.write(
            json.dumps(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "actor_uid": os.getuid(),
                    "before": before,
                    "after": after,
                }
            )
            + "\n"
        )
    print(json.dumps(after))


if __name__ == "__main__":
    main()
