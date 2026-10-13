"""Image-pinned, offline arm execution; no host runtimes or package directories.

Authoring and simulation live in separate containers. The agent can invoke only
one fixed simulator adapter, never evaluate arbitrary code in its environment.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from ..io import write_json
from ..process import execute

IMAGE_ID = re.compile(r"sha256:[a-f0-9]{64}\Z")


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
        raise ValueError(
            f"arm runtime requires a working Docker daemon: {exc}"
        ) from exc
    if result.returncode:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"exit status {result.returncode}"
        )
        raise ValueError(f"arm runtime command failed: {detail[-2000:]}")
    return result.stdout.strip()


def pin_image(image):
    # Never pull automatically. Missing dependencies are configuration failures.
    value = docker_output(["image", "inspect", "--format", "{{.Id}}", image])
    if not IMAGE_ID.fullmatch(value):
        raise ValueError("Docker did not return an immutable image ID")
    return value


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
        "--no-healthcheck",
        "--env",
        "HOME=/tmp",
        "--env",
        "XDG_CACHE_HOME=/tmp/.cache",
        "--env",
        "PYTHONNOUSERSITE=1",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=512m",
        "--entrypoint",
        "",
    ]
    if name:
        args += ["--name", name]
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


def remove_container(name):
    subprocess.run(
        [shutil.which("docker") or "docker", "rm", "--force", name],
        check=False,
        capture_output=True,
        timeout=30,
    )


def image_probe(image, command, timeout, mounts=()):
    name = "mriseqbench-probe-" + uuid.uuid4().hex
    args, image = container_args(image, name=name)
    for source, target in mounts:
        args += mount(source, target)
    try:
        return docker_output(args[1:] + [image, *command], timeout)
    finally:
        remove_container(name)


def probe_author(image, arm, timeout, julia_heap_size_hint_mb=None):
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
    # Revalidate here too: Pydantic instances can have been mutated by callers.
    type(experiment).model_validate(experiment.model_dump(mode="json"))
    config = experiment.arm_runtime
    if config is None:
        raise ValueError("restricted arm requires arm_runtime")
    if (arm.simulator == "none") != (config.simulator_image is None):
        raise ValueError(
            "simulator_image must be supplied exactly for simulation-enabled arms"
        )
    author = pin_image(config.author_image)
    audit = (
        probe_author(
            author, arm, config.probe_timeout_s, config.julia_heap_size_hint_mb
        )
        if config.julia_heap_size_hint_mb is not None
        else probe_author(author, arm, config.probe_timeout_s)
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
        ):
            raise ValueError(
                "simulator image must pass a Pulseq-input self-test for the selected engine"
            )
    return {
        "arm": arm.model_dump(mode="json"),
        "author_image": author,
        "simulator_image": simulator,
        "audit": audit,
        "simulator_probe": capabilities,
        "setup_timeout_s": config.probe_timeout_s,
        "julia_heap_size_hint_mb": config.julia_heap_size_hint_mb,
    }


@contextmanager
def launch_arm(runtime, command, workspace, control, env):
    name = "mriseqbench-" + uuid.uuid4().hex
    args, image = container_args(runtime["author_image"], name=name)
    args += mount(workspace, workspace, readonly=False)
    args += mount(control / "transport", "/run/mriseqbench", readonly=False)
    args += mount(
        Path(__file__).with_name("arm_gateway.py").resolve(), "/mriseqbench-gateway.py"
    )
    args += ["--workdir", str(workspace)]
    for key, value in env.items():
        args += ["--env", f"{key}={value}"]
    julia = runtime["arm"]["language"] == "julia"
    if julia:
        # Keep a failed server available for diagnostics until finally removes it.
        args.remove("--rm")
        for filename in ("arm_julia_session.jl", "arm_julia_client.py"):
            args += mount(Path(__file__).with_name(filename).resolve(), "/" + filename)
    metadata = {
        "backend": "docker",
        "isolated": True,
        "internet": False,
        "arm": runtime["arm"],
        "author_image": image,
        "simulator_image": runtime["simulator_image"],
        "audit": runtime["audit"],
        "host_read_grants": [],
    }
    try:
        if julia:
            # Start and warm the author process before execute() starts the agent
            # timer. docker exec reuses precisely this container and its process.
            started = time.monotonic()
            timeout = runtime.get("setup_timeout_s", 120)
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
                        if runtime.get("julia_heap_size_hint_mb") is not None
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
            session["heap_size_hint_mb"] = runtime.get("julia_heap_size_hint_mb")
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
            if julia:
                state = subprocess.run(
                    [
                        shutil.which("docker") or "docker",
                        "inspect",
                        "--format",
                        "{{json .State}}",
                        name,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if state.returncode == 0:
                    write_json(
                        control / "julia-container-state.json", json.loads(state.stdout)
                    )
                logs = subprocess.run(
                    [shutil.which("docker") or "docker", "logs", "--tail=100", name],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                (control / "julia-session.log").write_text(logs.stdout + logs.stderr)
        finally:
            # Killing the Docker client does not reliably stop its container.
            remove_container(name)


def arm_environment(agent, root, workspace, **values):
    from .sandbox import environment

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


class Simulator:
    def __init__(self, runtime, config, workspace, control, public_case):
        self.runtime, self.config = runtime, config
        self.workspace, self.control = Path(workspace), Path(control)
        self.public_case = public_case
        self.calls = []

    def __call__(self, body):
        if self.runtime["arm"]["simulator"] == "none":
            return 403, {"error": "simulation is disabled for this arm"}
        if not body or len(body) > 16777216:
            return 400, {"error": "simulation requires a nonempty Pulseq file <=16 MiB"}
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
            args, image = container_args(self.runtime["simulator_image"], name=name)
            args += mount(inputs, "/input") + mount(outputs, "/output", readonly=False)
            args += [image, "/opt/mriseqbench/simulate", "/input", "/output"]
            try:
                execution = execute(
                    args, base, self.config.simulation_timeout_s, base / "simulator"
                )
            finally:
                remove_container(name)
            record = {
                "sha256": hashlib.sha256(body).hexdigest(),
                "engine": self.runtime["arm"]["simulator"],
                "execution": execution,
            }
            if execution["status"] != "completed":
                # Keep bounded diagnostics after the temporary container files
                # disappear. These are public simulator logs, never judge data.
                record["stderr"] = _log_tail(base / "simulator.stderr.log")
                record["stdout_tail"] = _log_tail(base / "simulator.stdout.log")
            self.calls.append(record)
            write_json(self.control / "simulations.json", self.calls)
            if execution["status"] != "completed":
                return 422, {
                    "error": "simulator execution failed",
                    **record,
                }
            import base64

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
            return 200, {
                "engine": self.runtime["arm"]["simulator"],
                "artifacts": artifacts,
                **record,
            }
