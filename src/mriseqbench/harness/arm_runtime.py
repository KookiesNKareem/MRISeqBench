"""Image-pinned, offline arm execution; no host runtimes or package directories.

Authoring and simulation live in separate containers. The agent can invoke only
one fixed simulator adapter, never evaluate arbitrary code in its environment.
"""

import base64
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path

from ..io import write_json
from ..process import execute
from .sandbox import environment

IMAGE_ID = re.compile(r"sha256:[a-f0-9]{64}\Z")
CAMPAIGN = None  # Set by the runner; labels containers for orphan cleanup.


class DockerUnavailable(RuntimeError):
    """The Docker daemon or client transport failed: retryable infrastructure,
    unlike a ValueError (missing image, failed audit/probe: configuration)."""


DAEMON_DOWN = re.compile(
    r"Cannot connect to the Docker daemon|error during connect|"
    r"Is the docker daemon running|connection refused",
    re.IGNORECASE,
)


def docker_output(args, timeout=120):
    try:
        result = subprocess.run(
            ["docker", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DockerUnavailable(
            f"arm runtime requires a working Docker daemon: {exc}"
        ) from exc
    if result.returncode:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"exit status {result.returncode}"
        )
        if DAEMON_DOWN.search(detail):
            raise DockerUnavailable(f"Docker daemon unavailable: {detail[-2000:]}")
        raise ValueError(f"arm runtime command failed: {detail[-2000:]}")
    return result.stdout.strip()


def pin_image(image):
    # Never pull automatically. Missing dependencies are configuration failures.
    value = docker_output(["image", "inspect", "--format", "{{.Id}}", image])
    if not IMAGE_ID.fullmatch(value):
        raise ValueError("Docker did not return an immutable image ID")
    return value


@functools.cache
def rootless():
    # Rootless dockerd maps container root to the invoking host user.
    return "rootless" in docker_output(["info", "--format", "{{.SecurityOptions}}"])


def remove_campaign_containers(campaign):
    names = docker_output(
        ["ps", "-aq", "--filter", f"label=mriseqbench.campaign={campaign}"]
    ).split()
    if names:
        docker_output(["rm", "--force", *names])


def with_memory(args, megabytes):
    args = [arg for arg in args if not arg.startswith(("--memory=", "--memory-swap="))]
    return args + [f"--memory={megabytes}m", f"--memory-swap={megabytes}m"]


def container_args(image, *, name=None):
    executable = shutil.which("docker")
    if executable is None:
        raise ValueError(
            "restricted arms require Docker; no native fallback is allowed"
        )
    args = [
        executable,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=256",
        "--memory=768m",
        "--memory-swap=768m",
        "--cpus=1",
        "--env",
        "NODE_OPTIONS=--max-old-space-size=128",
        "--no-healthcheck",
        "--env",
        "HOME=/tmp",
        "--env",
        "XDG_CACHE_HOME=/tmp/.cache",
        "--env",
        "PYTHONNOUSERSITE=1",
        "--user",
        "0:0" if rootless() else f"{os.getuid()}:{os.getgid()}",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=512m",
        "--entrypoint",
        "",
    ]
    if name:
        args += ["--name", name]
    if CAMPAIGN:
        args += ["--label", f"mriseqbench.campaign={CAMPAIGN}"]
    return args, image


def mount(source, target, *, readonly=True):
    # --mount has its own CSV syntax; reject delimiters rather than misparse paths.
    if any(c in str(p) for p in (source, target) for c in (",", "\n", "\r")):
        raise ValueError("arm mount paths cannot contain commas or newlines")
    return [
        "--mount",
        f"type=bind,source={source},target={target}"
        + (",readonly" if readonly else ""),
    ]


def simulator_gpu_args():
    """Expose one scheduler-selected CUDA device only to the simulator service."""
    device = os.environ.get(
        "MRISEQBENCH_SIMULATOR_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "")
    )
    if not re.fullmatch(r"(?:[0-9]+|GPU-[a-fA-F0-9-]+|MIG-[a-zA-Z0-9/-]+)", device):
        raise ValueError(
            "Simulation requires one selected CUDA GPU; CPU fallback is disabled"
        )
    cache_args = [
        "--env",
        "JULIA_DEPOT_PATH=/tmp/julia-depot:/opt/julia-depot",
        "--env",
        "CUDA_CACHE_PATH=/tmp/cuda-cache",
        "--env",
        "JULIA_CUDA_HARD_MEMORY_LIMIT=2GiB",
    ]
    mode = os.environ.get("MRISEQBENCH_SIMULATOR_GPU_ACCESS", "auto")
    if mode == "auto":
        runtimes = json.loads(
            docker_output(["info", "--format", "{{json .Runtimes}}"], 10)
        )
        mode = "nvidia" if "nvidia" in runtimes else "devices"
    if mode == "devices":
        # Rootless daemons can expose device nodes without changing/restarting
        # a shared daemon. Bind only the selected device and CUDA driver ABI.
        rows = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
        choices = [tuple(part.strip() for part in row.split(",")) for row in rows]
        matches = [(index, uuid) for index, uuid in choices if device in (index, uuid)]
        if len(matches) != 1:
            raise ValueError("Selected simulator GPU was not found")
        index, uuid = matches[0]
        args = []
        for node in (f"/dev/nvidia{index}", "/dev/nvidiactl", "/dev/nvidia-uvm"):
            if not Path(node).exists():
                raise ValueError(f"Required CUDA device is missing: {node}")
            args += ["--device", node]
        directory = Path("/usr/lib/x86_64-linux-gnu")
        for library in ("libcuda.so.1", "libnvidia-ptxjitcompiler.so.1"):
            source = directory / library
            if not source.is_file():
                raise ValueError(f"Required CUDA driver library is missing: {library}")
            args += mount(source.resolve(), "/usr/local/nvidia/lib64/" + library)
        return (
            args
            + cache_args
            + [
                "--env",
                "LD_LIBRARY_PATH=/usr/local/nvidia/lib64",
                "--env",
                f"CUDA_VISIBLE_DEVICES={uuid}",
            ]
        )
    if mode != "nvidia":
        raise ValueError("Unknown simulator GPU access mode")
    return [
        "--gpus",
        f"device={device}",
        "--env",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
    ] + cache_args


def configure_simulator_args(args):
    # CUDA reserves large virtual mappings; an RLIMIT_DATA/AS cap can reject
    # initialization despite low RSS. Bound physical RAM with the container.
    return with_memory(args, 2048) + simulator_gpu_args()


def infrastructure_failure(isolation, execution):
    """Why an agent session that ended without a final submission is an
    infrastructure failure (Docker daemon or author container lost), or None.

    Rule, from the retained (never --rm) author container's Docker state:
    - harness-ended sessions (timeout, submission) are never infrastructure;
    - docker run failure (125), an unreachable daemon, or a container that is
      gone or still running after its client exited is infrastructure;
    - OOMKilled is the agent's: only the agent's processes (plus the bounded
      gateway/session harness) run under the published author memory limit;
    - otherwise a non-persistent container whose PID 1 (the gateway) was killed
      by a signal (137/143) was killed from outside: the agent cannot signal-kill
      its namespace's init, and its own processes' deaths surface as their exit
      codes through the gateway (signals as 256-n). For persistent (Julia/MCP)
      containers the agent runs via `docker exec`, so any stop is external.
    """
    if execution.get("status") in {"timeout", "submitted"}:
        return None
    if execution.get("returncode") == 125:
        return "docker could not run the author container (exit 125)"
    if not isolation.get("remote_host"):
        try:
            docker_output(["info", "--format", "{{.ServerVersion}}"], timeout=30)
        except (ValueError, DockerUnavailable) as exc:
            return f"docker daemon unavailable: {exc}"
    name = isolation.get("container")
    if not name:
        return None
    try:
        if isolation.get("remote_host"):
            from .remote_author import remote

            state = json.loads(
                remote(
                    isolation["remote_host"],
                    ["sudo", "docker", "inspect", "--format", "{{json .State}}", name],
                    capture_output=True,
                    text=True,
                    timeout=30,
                ).stdout
            )
        else:
            state = json.loads(
                docker_output(
                    ["inspect", "--format", "{{json .State}}", name], timeout=30
                )
            )
    except (ValueError, DockerUnavailable, OSError, subprocess.SubprocessError) as exc:
        return f"author container lost: {exc}"
    isolation["container_state"] = state
    if state.get("OOMKilled"):
        return None
    if isolation.get("persistent_container"):
        return None if state.get("Running") else "author container stopped"
    if state.get("Running"):
        return "docker client exited while the author container kept running"
    if state.get("Error") or state.get("ExitCode") in {137, 143}:
        return (
            f"author container killed externally (exit {state.get('ExitCode')}"
            f"{', ' + state['Error'] if state.get('Error') else ''})"
        )
    return None


def remove_container(name):
    """Force-remove; True only once Docker confirms the container no longer exists."""
    docker = shutil.which("docker") or "docker"
    try:
        subprocess.run(
            [docker, "rm", "--force", name],
            check=False,
            capture_output=True,
            timeout=60,
        )
        gone = subprocess.run(
            [docker, "inspect", "--format", "{{.Id}}", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return gone.returncode != 0 and "no such" in gone.stderr.lower()


def ensure_removed(name, attempts=3, interval=10):
    """Remove an author container or raise: a survivor stops the campaign
    (exit 4); the restart removes every campaign container first."""
    for attempt in range(attempts):
        if remove_container(name):
            return
        if attempt + 1 < attempts:
            time.sleep(interval)
    raise DockerUnavailable(f"container {name} could not be removed")


def image_probe(image, command, timeout, mounts=()):
    name = "mriseqbench-probe-" + uuid.uuid4().hex
    args, image = container_args(image, name=name)
    if command[0] == "/opt/mriseqbench/simulate":
        args = configure_simulator_args(args)
    for source, target in mounts:
        args += mount(source, target)
    try:
        return docker_output(args[1:] + [image, *command], timeout)
    finally:
        remove_container(name)


def probe_author(image, arm, timeout, julia_heap_size_hint_mb):
    script = Path(__file__).with_name("arm_audit.py").resolve()
    text = image_probe(
        image,
        [
            "python3",
            "/mriseqbench-audit.py",
            json.dumps(arm.model_dump(mode="json")),
            *(
                [str(julia_heap_size_hint_mb)]
                if julia_heap_size_hint_mb is not None
                else []
            ),
        ],
        timeout,
        [(script, "/mriseqbench-audit.py")],
    )
    try:
        result = json.loads(text.splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise ValueError("author image audit did not return valid JSON") from exc
    if result.get("ok") is not True:
        raise ValueError(f"author image violates arm restrictions: {result}")
    return result


def resolve_runtime(experiment, arm):
    config = experiment.arm_runtime
    if (arm.simulator == "none") != (config.simulator_image is None):
        raise ValueError(
            "simulator_image must be supplied exactly for simulation-enabled arms"
        )
    author = pin_image(config.author_image)
    pi_version = None
    if experiment.agent.uses_model and experiment.agent.adapter == "pi":
        pi_version = image_probe(author, ["pi", "--version"], config.probe_timeout_s)
        if pi_version.strip() != experiment.agent.pi_version:
            raise ValueError(
                f"Pi version mismatch: expected {experiment.agent.pi_version}, found {pi_version}"
            )
    audit = probe_author(
        author, arm, config.probe_timeout_s, config.julia_heap_size_hint_mb
    )
    simulator = None
    capabilities = None
    if config.simulator_image:
        simulator = pin_image(config.simulator_image)
        text = image_probe(
            simulator,
            ["/opt/mriseqbench/simulate", "--probe"],
            config.probe_timeout_s,
        )
        try:
            capabilities = json.loads(text.splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                "simulator image did not return a valid capability probe"
            ) from exc
        if (
            capabilities.get("engine") != arm.simulator
            or capabilities.get("pulseq_input") is not True
            or capabilities.get("self_test") is not True
            or capabilities.get("backend") != "cuda"
            or capabilities.get("cpu_fallback") is not False
        ):
            raise ValueError(
                "simulator image must pass a CUDA Pulseq-input self-test for the selected engine; no CPU fallback"
            )
    return {
        "arm": arm.model_dump(mode="json"),
        "pi_version": pi_version,
        "author_memory_mb": config.author_memory_mb,
        "author_image": author,
        "simulator_image": simulator,
        "audit": audit,
        "simulator_probe": capabilities,
        "setup_timeout_s": config.probe_timeout_s,
        "julia_heap_size_hint_mb": config.julia_heap_size_hint_mb,
    }


@contextmanager
def launch_arm(runtime, command, workspace, control, env):
    if runtime["arm"].get("tool_interface") == "julia_mcp":
        from .arm_mcp import launch_mcp

        with launch_mcp(runtime, command, workspace, control, env) as launched:
            yield launched
        return
    name = "mriseqbench-" + uuid.uuid4().hex
    args, image = container_args(runtime["author_image"], name=name)
    args = with_memory(args, runtime["author_memory_mb"])
    args += mount(workspace, workspace, readonly=False)
    args += mount(control / "transport", "/run/mriseqbench", readonly=False)
    args += mount(
        Path(__file__).with_name("arm_gateway.py").resolve(), "/mriseqbench-gateway.py"
    )
    args += ["--workdir", str(workspace)]
    for key, value in env.items():
        args += ["--env", f"{key}={value}"]
    julia = runtime["arm"]["language"] == "julia"
    # Keep the container (and its exit/OOM state) until infrastructure_failure
    # has inspected it; finally removes it.
    args.remove("--rm")
    if julia:
        for filename in ("arm_julia_session.jl", "arm_julia_client.py"):
            args += mount(Path(__file__).with_name(filename).resolve(), "/" + filename)
    metadata = {
        "backend": "docker",
        "isolated": True,
        "internet": False,
        "container": name,
        "persistent_container": julia,
    }
    try:
        if julia:
            # Start and warm the author process before execute() starts the agent
            # timer. docker exec reuses precisely this container and its process.
            started = time.monotonic()
            timeout = runtime["setup_timeout_s"]
            docker_output(
                args[1:]
                + [
                    "--detach",
                    image,
                    "julia",
                    "--startup-file=no",
                    "--compiled-modules=existing",
                    *(
                        [f"--heap-size-hint={runtime['julia_heap_size_hint_mb']}M"]
                        if runtime["julia_heap_size_hint_mb"] is not None
                        else []
                    ),
                    "--project=/opt/author",
                    "/arm_julia_session.jl",
                    str(runtime["arm"]["pulse_designer"]).lower(),
                    str(runtime["arm"]["unitful"]).lower(),
                ],
                timeout=timeout,
            )
            ready = docker_output(
                [
                    "exec",
                    name,
                    "python3",
                    "/arm_julia_client.py",
                    "--ready",
                    "--timeout",
                    str(timeout),
                ],
                timeout=timeout + 5,
            )
            session = json.loads(ready)
            if session.get("mode") != "persistent" or not session.get("revise_version"):
                raise ValueError("Julia session did not confirm warm readiness")
            session["setup_elapsed_s"] = time.monotonic() - started
            metadata["julia_session"] = session
            write_json(control / "julia-session.json", session)
            args = [
                args[0],
                "exec",
                name,
                "python3",
                "/mriseqbench-gateway.py",
                json.dumps(command),
            ]
        else:
            args += [image, "python3", "/mriseqbench-gateway.py", json.dumps(command)]
        yield args, metadata
    finally:
        try:
            state = subprocess.run(
                [args[0], "inspect", "--format", "{{json .State}}", name],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if state.returncode == 0:
                write_json(
                    control / f"{'julia' if julia else 'author'}-container-state.json",
                    json.loads(state.stdout),
                )
            if julia:
                logs = subprocess.run(
                    [args[0], "logs", "--tail=100", name],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                (control / "julia-session.log").write_text(logs.stdout + logs.stderr)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass  # Diagnostics are best effort; removal below is not.
        finally:
            # Killing the Docker client does not reliably stop its container.
            ensure_removed(name)


def arm_environment(agent, root, workspace, **values):
    env = environment(agent, root, workspace, **values)
    # Use image runtimes only; never prepend the host interpreter/agent path.
    env["PATH"] = "/opt/venv/bin:/usr/local/julia/bin:/usr/local/bin:/usr/bin:/bin"
    env.update(
        JULIA_DEPOT_PATH="/opt/julia-depot",
        JULIA_LOAD_PATH="/opt/author:@stdlib",
        JULIA_PKG_OFFLINE="true",
        JULIA_HISTORY="/dev/null",
        PYTHONNOUSERSITE="1",
        PYTHONPATH="",
        PIP_NO_INDEX="1",
    )
    return env


def _log_tail(path, limit=4000):
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - limit))
        return stream.read(limit).decode("utf-8", errors="replace")


# Adapter/driver failures that no sequence can cause (CUDA, prewarm, devices).
SIMULATOR_INFRASTRUCTURE = re.compile(
    r"no CPU fallback|not prewarmed|cannot fall back to CPU|CUDA_ERROR|"
    r"CUDA driver|NVML|Required CUDA|libcuda",
    re.IGNORECASE,
)


class SimulatorSlots:
    """Campaign-wide GPU slots. A slot whose container removal is unconfirmed
    stays held while cleanup retries; if removal never succeeds the pool is
    broken: waiting and later simulations interrupt their trials and the
    campaign stops dispatch and exits resumably."""

    def __init__(self, count, retries=10, interval=30):
        self.free = threading.BoundedSemaphore(count)
        self.broken = threading.Event()
        self.retries, self.interval = retries, interval
        self.reapers = []

    def acquire(self, cancelled):
        while not (self.broken.is_set() or cancelled()):
            if self.free.acquire(timeout=1):
                if not (self.broken.is_set() or cancelled()):
                    return True
                self.free.release()
        return False

    def reap(self, name):
        def retry():
            for _ in range(self.retries):
                time.sleep(self.interval)
                if remove_container(name):
                    self.free.release()
                    return
            self.broken.set()

        reaper = threading.Thread(target=retry, daemon=True)
        self.reapers.append(reaper)
        reaper.start()

    def drain(self):
        """Wait for pending cleanup before the campaign exits (a daemon
        thread would die with it, abandoning the container)."""
        for reaper in self.reapers:
            reaper.join()


class Simulator:
    def __init__(
        self,
        runtime,
        config,
        workspace,
        control,
        public_case,
        slots=None,
        waiting=None,
        infrastructure=None,
    ):
        self.runtime, self.config = runtime, config
        self.workspace, self.control = Path(workspace), Path(control)
        self.public_case = public_case
        self.calls = []
        # Campaign-wide GPU slots; waiting() excludes queue time from the agent clock.
        self.slots = slots or SimulatorSlots(1)
        self.waiting = waiting or nullcontext
        # Called with a reason when the simulator itself failed (see classify).
        self.infrastructure = infrastructure or (lambda reason: None)
        self.infrastructure_failure = None
        self.closed = threading.Event()  # Set when the trial's mailbox shuts down.

    def fail(self, record, reason):
        self.infrastructure_failure = record["infrastructure_failure"] = reason
        write_json(self.control / "simulations.json", self.calls)
        self.infrastructure(reason)
        return 503, {"error": f"simulator infrastructure failure: {reason}", **record}

    @staticmethod
    def classify(execution, state, stderr):
        """Infrastructure reason for a failed run, or None for a participant
        error (invalid/unsupported sequence, its memory or time use) shown to
        the agent. A timeout stays the participant's: simulation cost grows
        with the submitted sequence, and the image passed its GPU self-test."""
        if execution["status"] == "launch_error":
            return f"docker client failed to launch: {execution.get('error')}"
        if execution.get("returncode") in {125, 126, 127}:
            return (
                f"docker could not run the simulator (exit {execution['returncode']})"
            )
        if state is None:
            return "simulator container lost before inspection"
        if state.get("Error"):
            return f"simulator container error: {state['Error']}"
        if (
            execution["status"] != "timeout"
            and not state.get("OOMKilled")
            and state.get("ExitCode") in {137, 143}
        ):
            return f"simulator container killed externally (exit {state['ExitCode']})"
        if DAEMON_DOWN.search(stderr):
            return "Docker daemon unavailable during simulation"
        if SIMULATOR_INFRASTRUCTURE.search(stderr):
            return "CUDA/simulator runtime failure: " + stderr.strip()[-500:]
        return None

    def __call__(self, body):
        if not body:
            return 400, {"error": "simulation requires a nonempty Pulseq file"}
        # Snapshot public inputs before accepting untrusted agent modifications.
        # A fresh container gets only these bytes, never trusted evaluator state.
        with tempfile.TemporaryDirectory(
            dir=self.control, prefix="simulation-"
        ) as temporary:
            base = Path(temporary)
            inputs, outputs = base / "input", base / "output"
            shutil.copytree(self.control / "simulation_inputs", inputs)
            outputs.mkdir()
            (inputs / "sequence.seq").write_bytes(body)
            write_json(inputs / "case.json", self.public_case)
            name = "mriseqbench-sim-" + uuid.uuid4().hex
            record = {
                "sha256": hashlib.sha256(body).hexdigest(),
                "engine": self.runtime["arm"]["simulator"],
            }
            try:  # Host Docker/GPU discovery; no participant input is involved.
                args, image = container_args(self.runtime["simulator_image"], name=name)
                args.remove("--rm")  # Inspect exit/OOM state before removal.
                args = configure_simulator_args(args)
            except Exception as exc:  # noqa: BLE001
                self.calls.append(record)
                return self.fail(record, f"simulator setup failed: {exc}")
            args += mount(inputs, "/input") + mount(outputs, "/output", readonly=False)
            args += [image, "/opt/mriseqbench/simulate", "/input", "/output"]
            queued = time.monotonic()
            with self.waiting():
                acquired = self.slots.acquire(self.closed.is_set)
            record["queue_wait_s"] = time.monotonic() - queued
            if not acquired:
                if self.closed.is_set():
                    return 503, {"error": "the trial ended while queued", **record}
                self.calls.append(record)
                return self.fail(record, "simulator slots lost to unremoved containers")
            state, removed = None, False
            try:
                # Closing the trial kills the client ("submitted"); the
                # container is still removed below before the slot frees.
                execution = execute(
                    args,
                    base,
                    self.config.simulation_timeout_s,
                    base / "simulator",
                    stop_when=self.closed,
                )
                if execution["status"] not in {"completed", "submitted"}:
                    try:
                        state = json.loads(
                            docker_output(
                                ["inspect", "--format", "{{json .State}}", name], 30
                            )
                        )
                    except (ValueError, DockerUnavailable):
                        state = None
            finally:
                # The slot frees only once the container (and its GPU context)
                # is confirmed gone; a killed client can leave it running.
                removed = remove_container(name)
                if removed:
                    self.slots.free.release()
                else:
                    self.slots.reap(name)
            record["execution"] = execution
            if execution["status"] != "completed":
                # Keep bounded diagnostics after the temporary container files
                # disappear. These are public simulator logs, never judge data.
                record["stderr"] = _log_tail(base / "simulator.stderr.log")
                record["stdout_tail"] = _log_tail(base / "simulator.stdout.log")
                record["container_state"] = state
            self.calls.append(record)
            write_json(self.control / "simulations.json", self.calls)
            if execution["status"] == "submitted":
                return 503, {"error": "the trial ended during simulation", **record}
            if not removed:
                return self.fail(
                    record,
                    f"simulator container {name} was not removed; slot held while cleanup retries",
                )
            if execution["status"] != "completed":
                if reason := self.classify(execution, state, record["stderr"]):
                    return self.fail(record, reason)
                return 422, {
                    "error": "simulator execution failed",
                    **record,
                }
            artifacts, total = {}, 0
            for path in outputs.rglob("*"):
                if path.is_symlink() or not (path.is_file() or path.is_dir()):
                    return 422, {"error": "simulator produced an invalid output file"}
                if path.is_file():
                    total += path.stat().st_size
                    if total > 64 * 1024 * 1024:
                        return 422, {"error": "simulator outputs exceed 64 MiB"}
                    artifacts[str(path.relative_to(outputs))] = base64.b64encode(
                        path.read_bytes()
                    ).decode()
            if not artifacts:
                return 422, {"error": "simulator produced no output artifacts"}
            execution_path = outputs / "simulation.json"
            device = (
                json.loads(execution_path.read_text())
                if execution_path.is_file()
                else {}
            )
            if (
                device.get("backend") != "cuda"
                or device.get("cpu_fallback") is not False
            ):
                return self.fail(
                    record, "simulator did not verify CUDA execution (no CPU results)"
                )
            record["device"] = device
            write_json(self.control / "simulations.json", self.calls)
            return 200, {
                "engine": self.runtime["arm"]["simulator"],
                "artifacts": artifacts,
                **record,
            }
