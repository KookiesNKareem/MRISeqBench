"""Trusted remote Docker execution; author containers retain network=none."""

import hashlib
import json
import secrets
import shlex
import subprocess
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..io import write_json


def ssh(host):
    return [
        "ssh",
        "-4",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-i",
        host["key"],
        host["endpoint"],
    ]


def remote(host, args, **kwargs):
    return subprocess.run(ssh(host) + [shlex.join(args)], check=True, **kwargs)


def sync(host, source, destination):
    subprocess.run(
        [
            "rsync",
            "-a",
            "--safe-links",
            "--protect-args",
            "--rsync-path=sudo rsync",
            "-e",
            shlex.join(ssh(host)[:-1]),
            source,
            destination,
        ],
        check=True,
        timeout=600,
        stdout=subprocess.DEVNULL,
    )


def manifest(directory):
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(directory).rglob("*"))
        if p.is_file() and not p.is_symlink() and p.name != ".transfer-manifest.json"
    }


def request_host_drain(host, state_dir):
    """A worker pool can report its own drain, never retire a shared VM.

    The host lease owner must separately verify every pool, artifact return
    and admission closure before authorizing an early shutdown.
    """
    write_json(
        Path(state_dir) / (host["instance_id"] + "-drain-request.json"),
        {
            "reason": "this worker pool's assigned trial queue drained",
            "shutdown_requested": False,
            "requires_host_lease_owner_verification": True,
        },
    )


class Fleet:
    def __init__(self, config):
        self.config = json.loads(Path(config).read_text())
        self.hosts = self.config["hosts"]
        self.broker_port = self.config.get("broker_port", 8766)
        if type(self.broker_port) is not int or not 1 <= self.broker_port <= 65535:
            raise ValueError("remote broker port must be an integer from 1 to 65535")
        helpers = [
            Path(__file__).with_name("remote_author_worker.py"),
            Path(__file__).with_name("arm_gateway.py"),
        ]
        version = hashlib.sha256(b"".join(p.read_bytes() for p in helpers)).hexdigest()[
            :16
        ]
        self.helper_root = f"/opt/mriseqbench/transport-{version}"
        self.tokens = {}
        self.lock = threading.Lock()
        self.slots = [threading.BoundedSemaphore(h["slots"]) for h in self.hosts]
        self.pending = [0] * len(self.hosts)
        for index in self.config["assignments"].values():
            self.pending[index] += 1
        fleet = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                token, _, name = self.path.lstrip("/").partition("/")
                with fleet.lock:
                    directory = fleet.tokens.get(token)
                if (
                    directory is None
                    or len(name) != 32
                    or any(c not in "0123456789abcdef" for c in name)
                ):
                    self.send_error(403)
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 24 * 1024 * 1024:
                    self.send_error(413)
                    return
                request = directory / (name + ".request")
                if request.exists():
                    self.send_error(409)
                    return
                temporary = request.with_suffix(".remote-tmp")
                temporary.write_bytes(self.rfile.read(size))
                temporary.replace(request)
                response = request.with_suffix(".response")
                self.send_response(200)
                self.send_header("Content-Type", "application/jsonl")
                self.end_headers()
                try:
                    while not response.exists():
                        with fleet.lock:
                            if token not in fleet.tokens:
                                raise ConnectionError("trial revoked")
                        time.sleep(0.02)
                    with response.open("rb") as stream:
                        while True:
                            line = stream.readline()
                            if line and line.endswith(b"\n"):
                                self.wfile.write(line)
                                self.wfile.flush()
                                if json.loads(line).get("done"):
                                    return
                            else:
                                if line:
                                    stream.seek(-len(line), 1)
                                with fleet.lock:
                                    if token not in fleet.tokens:
                                        raise ConnectionError("trial revoked")
                                time.sleep(0.02)
                except (OSError, ValueError):
                    return

            def log_message(self, *_):
                pass

        class BrokerServer(ThreadingHTTPServer):
            request_queue_size = 256

        self.server = BrokerServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.tunnels = []
        for host in self.hosts:
            child = subprocess.Popen(
                ssh(host)
                + [
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-NT",
                    "-R",
                    f"127.0.0.1:{self.broker_port}:127.0.0.1:{self.server.server_port}",
                ],
                stdout=subprocess.DEVNULL,
            )
            self.tunnels.append(child)
        time.sleep(1)
        if any(p.poll() is not None for p in self.tunnels):
            raise RuntimeError("author reverse tunnel failed")

    @contextmanager
    def launch(self, runtime, command, workspace, control, env):
        from . import arm_runtime

        index = self.config["assignments"][workspace.parent.name]
        host = self.hosts[index]
        self.slots[index].acquire()
        name = "mriseqbench-cloud-" + secrets.token_hex(12)
        token = secrets.token_hex(32)
        spec_path = control / "remote-author-spec.json"
        ready = False
        try:
            remote(
                host,
                [
                    "sudo",
                    "mkdir",
                    "-p",
                    str(workspace),
                    str(control / "transport"),
                    self.helper_root,
                ],
                timeout=30,
            )
            sync(
                host,
                str(workspace) + "/",
                host["endpoint"] + ":" + str(workspace) + "/",
            )
            # Rootful cloud Docker uses uid 0 without DAC override capabilities.
            # Uploaded files retain the campaign host's uid, so make only these public
            # writable mounts belong to the sandbox's cloud uid.
            remote(
                host,
                [
                    "sudo",
                    "chown",
                    "-R",
                    "0:0",
                    str(workspace),
                    str(control / "transport"),
                ],
                timeout=60,
            )
            for path in [
                Path(__file__).with_name("remote_author_worker.py"),
                Path(__file__).with_name("arm_gateway.py"),
            ]:
                sync(
                    host,
                    str(path),
                    host["endpoint"] + ":" + self.helper_root + "/" + path.name,
                )
            args, image = arm_runtime.container_args(runtime["author_image"], name=name)
            args[0] = "docker"
            args.remove("--rm")
            args[args.index("--user") + 1] = "0:0"
            args = arm_runtime.with_memory(args, runtime["author_memory_mb"])
            args += (
                arm_runtime.mount(workspace, workspace, readonly=False)
                + arm_runtime.mount(
                    control / "transport", "/run/mriseqbench", readonly=False
                )
                + arm_runtime.mount(
                    self.helper_root + "/arm_gateway.py", "/mriseqbench-gateway.py"
                )
                + ["--workdir", str(workspace)]
            )
            for k, value in env.items():
                args += ["--env", f"{k}={value}"]
            args += [image, "python3", "/mriseqbench-gateway.py", json.dumps(command)]
            write_json(
                spec_path,
                {
                    "command": ["sudo", *args],
                    "name": name,
                    "directory": str(control / "transport"),
                    "token": token,
                    "broker_url": f"http://127.0.0.1:{self.broker_port}",
                },
            )
            sync(host, str(spec_path), host["endpoint"] + ":" + str(spec_path))
            with self.lock:
                self.tokens[token] = control / "transport"
            ready = True
            write_json(
                control / "remote-author-host.json",
                {
                    **host,
                    "container": name,
                    "transport": "ssh-reverse-mailbox",
                    "author_image": image,
                },
            )
            yield (
                ssh(host)
                + [
                    shlex.join(
                        [
                            "sudo",
                            "python3",
                            self.helper_root + "/remote_author_worker.py",
                            str(spec_path),
                        ]
                    )
                ],
                {
                    "backend": "docker",
                    "remote_host": host,
                    "isolated": True,
                    "internet": False,
                    "container": name,
                    "persistent_container": False,
                },
            )
        finally:
            failure = None
            try:
                state = remote(
                    host,
                    ["sudo", "docker", "inspect", "--format", "{{json .State}}", name],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                write_json(
                    control / "author-container-state.json", json.loads(state.stdout)
                )
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                write_json(
                    control / "remote-state-read-error.json", {"error": str(exc)}
                )
            try:
                remote(
                    host,
                    ["sudo", "docker", "rm", "--force", name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=60,
                )
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                failure = exc
            if ready:
                try:
                    script = (
                        "import json,hashlib;from pathlib import Path;p=Path("
                        + repr(str(workspace))
                        + ");d={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and not f.is_symlink() and f.name!='.transfer-manifest.json'};print(json.dumps(d,sort_keys=True))"
                    )
                    expected = json.loads(
                        remote(
                            host,
                            ["sudo", "python3", "-c", script],
                            capture_output=True,
                            text=True,
                            timeout=120,
                        ).stdout
                    )
                    sync(
                        host,
                        host["endpoint"] + ":" + str(workspace) + "/",
                        str(workspace) + "/",
                    )
                    if manifest(workspace) != expected:
                        raise ValueError("remote author transfer SHA256 mismatch")
                    write_json(
                        control / "remote-author-transfer.json",
                        {
                            "status": "verified",
                            "files": expected,
                            "instance_id": host["instance_id"],
                        },
                    )
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    failure = exc
            with self.lock:
                self.tokens.pop(token, None)
                self.pending[index] -= 1
                retire = self.pending[index] == 0
            self.slots[index].release()
            if retire and failure is None and self.config.get("state_dir"):
                request_host_drain(host, self.config["state_dir"])
            if failure is not None:
                raise RuntimeError(
                    "Remote author infrastructure/transfer failure; preserve trial"
                ) from failure

    def close(self):
        self.server.shutdown()
        for child in self.tunnels:
            child.terminate()
