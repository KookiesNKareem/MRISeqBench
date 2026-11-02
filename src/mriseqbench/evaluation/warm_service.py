"""Campaign-local transport to one persistent, private Koma GPU worker."""

import atexit
import fcntl
import json
import os
import select
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path

import h5py

from . import s1_koma
from ..storage import atomic_json


class SharedKomaWorker:
    sequence = s1_koma.KomaWorker.sequence

    def __init__(self, ready):
        config = json.loads(Path(ready).read_text())
        services = config.get("workers", [config])
        self.lease = None
        if "workers" in config:
            while self.lease is None:
                for service in services:
                    lease = Path(service["socket"]).with_name("lease.lock").open("a+")
                    try:
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lease.close()
                        continue
                    self.lease = lease
                    self.service = service
                    atexit.register(lease.close)
                    break
                if self.lease is None:
                    time.sleep(0.02)
        else:
            self.service = config
        self.directory = tempfile.TemporaryDirectory(prefix="mriseqbench-koma-client-")
        self.path = Path(self.directory.name)
        self.lock = threading.Lock()
        self.sequence_cache = {}
        self.inversion_cache = {}
        self.execution = self.service["execution"]
        atexit.register(self.directory.cleanup)

    def call(self, request, write=None, read=None):
        with self.lock:
            source, target = self.path / "input.h5", self.path / "output.h5"
            if write:
                with h5py.File(source, "w") as handle:
                    write(handle)
            deadline = s1_koma._DEADLINE.get()
            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            if remaining == 0:
                raise TimeoutError("KomaMRI evaluation budget exhausted")
            message = {"request": request, "client": str(self.path), "pid": os.getpid(),
                       "input": str(source) if write else None,
                       "output": str(target) if read else None, "budget_s": remaining}
            with socket.socket(socket.AF_UNIX) as connection:
                connection.settimeout(remaining)
                connection.connect(self.service["socket"])
                connection.sendall((json.dumps(message, allow_nan=False) + "\n").encode())
                with connection.makefile("rb") as stream:
                    line = stream.readline()
            if not line:
                raise RuntimeError("Persistent Koma grader disconnected; submission is unscored")
            response = json.loads(line)
            if not response["ok"]:
                raise s1_koma.worker_error(response["error"])
            self.execution = response.get("execution") or self.execution
            if read:
                with h5py.File(target) as handle:
                    return read(handle)
            return response


def serve(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (directory / "service.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    endpoint = directory / "worker.sock"
    ready = directory / "READY.json"
    ready.unlink(missing_ok=True)
    endpoint.unlink(missing_ok=True)
    engine = s1_koma.KomaWorker()
    version = engine.call({"operation": "versions"})
    assert version["execution"]["backend"] == "metal"
    assert version["execution"]["precision"] == "f32"
    info = {"pid": os.getpid(), "worker_pid": engine.process.pid,
            "socket": str(endpoint), "execution": version["execution"],
            "started_unix_s": time.time(), "calls": 0, "client": None}
    last_client = None
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(endpoint))
        endpoint.chmod(0o600)
        listener.listen(16)
        atomic_json(ready, info)
        while True:
            connection, _ = listener.accept()
            with connection, connection.makefile("rb") as stream:
                message = json.loads(stream.readline())
                started = time.monotonic()
                info.update(client=message["client"], client_pid=message["pid"],
                            operation=message["request"]["operation"], active=True)
                atomic_json(directory / "status.json", info)
                outcome = {}

                def execute():
                    nonlocal engine, last_client
                    try:
                        if engine.process.poll() is not None:
                            engine = s1_koma.KomaWorker()
                            last_client = None
                        if last_client != message["client"]:
                            engine.call({"operation": "clear_cache"})
                            last_client = message["client"]
                        if message["input"]:
                            shutil.copyfile(message["input"], engine.path / "input.h5")
                        with s1_koma.simulation_budget(message["budget_s"]):
                            response = engine.call(message["request"])
                        if message["output"]:
                            shutil.copyfile(engine.path / "output.h5", message["output"])
                        outcome["response"] = response
                    except Exception as exc:
                        outcome["response"] = {"ok": False, "error": str(exc).removeprefix("KomaMRI: ")}

                task = threading.Thread(target=execute)
                task.start()
                disconnected = False
                while task.is_alive():
                    if select.select([connection], [], [], 0.1)[0] and not connection.recv(1, socket.MSG_PEEK):
                        engine.process.terminate()
                        disconnected = True
                        break
                task.join()
                info.update(active=False, worker_pid=engine.process.pid,
                            calls=info["calls"] + 1, last_call_s=time.monotonic() - started)
                atomic_json(directory / "status.json", info)
                if not disconnected:
                    try:
                        connection.sendall((json.dumps(outcome["response"]) + "\n").encode())
                    except BrokenPipeError:
                        pass


if __name__ == "__main__":
    import sys
    serve(sys.argv[1])
