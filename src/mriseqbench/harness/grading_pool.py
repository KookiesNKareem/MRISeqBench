"""Live operational grading capacity, independent of scientific run identity."""

import hashlib
import json
import os
import tempfile
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".grading-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def slot_count(value):
    if type(value) is not int or not 0 <= value <= 64:
        raise ValueError("grading slots must be an integer from 0 to 64")
    return value


def read_gpu_pool(path):
    devices = json.loads(Path(path).read_text())
    if not isinstance(devices, list):
        raise ValueError("grading GPU config must be a list of uuid/slots objects")  # noqa: TRY004
    seen = set()
    for device in devices:
        if not isinstance(device, dict) or set(device) != {"uuid", "slots"}:
            raise ValueError("each grading GPU needs exactly uuid and slots")
        uuid = device["uuid"]
        if not isinstance(uuid, str) or not uuid.strip() or uuid in seen:
            raise ValueError("grading GPU UUIDs must be nonempty and unique")
        seen.add(uuid)
        slot_count(device["slots"])
    slot_count(sum(device["slots"] for device in devices))
    return devices


class GradingSlots:
    """FIFO admission with hot reload; reducing capacity never kills a holder.

    A malformed/missing update pauses new admission until repaired. Existing
    holders still release normally. No watcher thread or agent restart is needed.
    """

    poll_s = 0.2

    def __init__(self, config, *, gpu=False, status=None, history=None):
        self.config = Path(config)
        self.gpu = gpu
        self.status_path = Path(status) if status else None
        self.history_path = Path(history) if history else None
        self.condition = threading.Condition()
        self.waiters = deque()
        self.active = 0
        self.capacity = 0
        self.error = None
        self.revision = None
        self._published = None
        self._refresh(initial=True)

    @classmethod
    def for_campaign(cls, output, default):
        output = Path(output)
        gpu = os.environ.get("MRISEQBENCH_GRADING_GPU_CONFIG")
        config = Path(gpu) if gpu else output / "grading-slots.json"
        if not gpu and not config.exists():
            atomic_json(config, {"slots": slot_count(default)})
        return cls(
            config,
            gpu=bool(gpu),
            status=output / "grading-pool-status.json",
            history=output / "grading-pool-changes.jsonl",
        )

    def _refresh(self, *, initial=False):
        try:
            if self.gpu:
                settings = read_gpu_pool(self.config)
                capacity = sum(device["slots"] for device in settings)
            else:
                settings = json.loads(self.config.read_text())
                if not isinstance(settings, dict) or set(settings) != {"slots"}:
                    raise ValueError("grading config must contain exactly slots")
                capacity = slot_count(settings["slots"])
            error = None
            revision = hashlib.sha256(
                json.dumps(settings, sort_keys=True).encode()
            ).hexdigest()
        except (OSError, ValueError, TypeError) as exc:
            if initial:
                raise ValueError(f"Invalid grading pool {self.config}: {exc}") from exc
            settings, capacity, error, revision = None, 0, str(exc), None
        changed = (revision, error) != (self.revision, self.error)
        self.capacity, self.error, self.revision = capacity, error, revision
        if changed and self.history_path:
            with self.history_path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "at": datetime.now(UTC).isoformat(),
                            "config": str(self.config),
                            "settings": settings,
                            "configured_slots": capacity,
                            "config_error": error,
                            "revision": revision,
                        }
                    )
                    + "\n"
                )
        return changed

    def _state(self):
        return {
            "config": str(self.config),
            "configured_slots": self.capacity,
            "active": self.active,
            "waiting": len(self.waiters),
            "config_error": self.error,
            "revision": self.revision,
        }

    def _publish(self):
        state = self._state()
        if self.status_path and state != self._published:
            atomic_json(self.status_path, state)
            self._published = state

    def stats(self):
        with self.condition:
            self._refresh()
            self._publish()
            return self._state()

    def acquire(self, blocking=True, timeout=None, *, stop=None):
        if not blocking and timeout is not None:
            raise ValueError("can't specify timeout for non-blocking acquire")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative")
        deadline = None if timeout is None else time.monotonic() + timeout
        token = object()
        with self.condition:
            self.waiters.append(token)
            try:
                while True:
                    changed = self._refresh()
                    if stop is not None and stop.is_set():
                        return False
                    if self.waiters[0] is token and self.active < self.capacity:
                        self.waiters.popleft()
                        self.active += 1
                        try:
                            self._publish()
                        except BaseException:
                            self.active -= 1
                            self.condition.notify_all()
                            raise
                        self.condition.notify_all()
                        return True
                    remaining = (
                        None if deadline is None else deadline - time.monotonic()
                    )
                    if not blocking or (remaining is not None and remaining <= 0):
                        return False
                    if changed:
                        self.condition.notify_all()
                    self._publish()
                    self.condition.wait(
                        self.poll_s
                        if remaining is None
                        else min(self.poll_s, remaining)
                    )
            finally:
                if token in self.waiters:
                    self.waiters.remove(token)
                    self._publish()
                    self.condition.notify_all()

    def release(self):
        with self.condition:
            if self.active <= 0:
                raise ValueError("grading slot released without an acquisition")
            self.active -= 1
            self._refresh()
            self._publish()
            self.condition.notify_all()
