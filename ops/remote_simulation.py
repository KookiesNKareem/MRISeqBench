"""SHA-verified transport for the pinned author simulator; no model credentials."""

import hashlib
import json
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

IMAGE = "sha256:a333d8bd759abfa9b88c1dd65c48df761dd1b8e312ab80084059b823a95c6275"
_SHORT_TRANSPORT = threading.Lock()


def manifest(directory):
    result = {}
    for p in sorted(Path(directory).rglob("*")):
        if p.is_symlink():
            raise ValueError("Simulation transport rejects symlinks")
        if p.is_file() and p.name != "transfer-manifest.json":
            h = hashlib.sha256()
            with p.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    h.update(block)
            result[str(p.relative_to(directory))] = h.hexdigest()
    return result


def dispatch(
    host,
    inputs,
    outputs,
    receipt_directory,
    slot,
    *,
    cancelled=lambda: False,
    timeout=300,
    command=None,
    qualification_warmup=False,
):
    identity = uuid.uuid4().hex
    remote = "/opt/mriseqbench-sim/jobs/" + identity
    record = Path(receipt_directory) / identity
    record.mkdir(parents=True)
    ssh = [
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
        "-p",
        str(host["port"]),
        "-i",
        host["key"],
    ]
    if host.get("ssh_control_path"):
        ssh += [
            "-o", "ControlMaster=auto",
            "-o", "ControlPersist=120",
            "-o", "ControlPath=" + host["ssh_control_path"],
        ]
    transport = shlex.join(ssh)

    def call(argv, **kwargs):
        with _SHORT_TRANSPORT:
            return subprocess.run(
                ssh + [host["endpoint"], shlex.join(argv)],
                check=True,
                timeout=120,
                **kwargs,
            )

    def sync(source, target):
        # Eight long execution channels plus one short transfer channel fit
        # within OpenSSH's default session ceiling, including during returns.
        with _SHORT_TRANSPORT:
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--safe-links",
                    "--protect-args",
                    "-e",
                    transport,
                    "--",
                    source,
                    target,
                ],
                check=True,
                timeout=300,
            )

    call(["mkdir", "-p", remote + "/input"])
    input_manifest = manifest(inputs)
    sync(str(inputs) + "/", host["endpoint"] + ":" + remote + "/input/")
    spec = {"image": IMAGE, "input_manifest": input_manifest, "timeout": timeout}
    if command is not None:
        spec["command"] = command
    if qualification_warmup:
        assert command and (
            command == ["/opt/mriseqbench/simulate", "--probe"]
            or any(
                x in command[-1]
                for x in (
                    "Pkg.precompile()",
                    "Base.Precompilation.precompilepkgs()",
                )
            )
        )
        spec["qualification_warmup"] = True
    (record / "request.json").write_text(json.dumps(spec, indent=2) + "\n")
    started = time.monotonic()
    requested_at = datetime.now(UTC).isoformat()
    with (
        (record / "transport.stdout.log").open("w") as out,
        (record / "transport.stderr.log").open("w") as err,
    ):
        process = subprocess.Popen(
            ssh
            + [
                host["endpoint"],
                shlex.join(
                    [
                        "python3",
                        "/opt/mriseqbench-sim/run-image.py",
                        remote,
                        str(slot),
                    ]
                ),
            ],
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=err,
        )
        process.stdin.write((json.dumps(spec) + "\n").encode())
        process.stdin.flush()
        while process.poll() is None:
            if cancelled() or time.monotonic() - started > timeout + 180:
                process.stdin.close()  # Remote EOF kills and reaps its process group.
                break
            time.sleep(0.1)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        finally:
            if not process.stdin.closed:
                process.stdin.close()
    returned = record / "returned"
    returned.mkdir()
    sync(host["endpoint"] + ":" + remote + "/", str(returned) + "/")
    expected = json.loads((returned / "transfer-manifest.json").read_text())
    assert manifest(returned) == expected, "Remote simulation SHA256 mismatch"
    result = json.loads((returned / "execution.json").read_text())
    assert result["remote_process_group_reaped"]
    Path(outputs).mkdir(parents=True, exist_ok=True)
    for p in (returned / "output").rglob("*"):
        if p.is_file():
            dest = Path(outputs) / p.relative_to(returned / "output")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    # All inputs, arrays and logs are now retained and verified on the campaign host.
    call(["rm", "-rf", "--", remote])
    (record / "receipt.json").write_text(
        json.dumps(
            {
                "requested_at": requested_at,
                "image_id": IMAGE,
                "slot": slot,
                "SHA256_verified": True,
                "remote_process_group_reaped": True,
                "remote_scratch_removed": True,
                "remote_execution": result,
                "transport_and_execution_wall_s": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n"
    )
    return result, record


def install(arm_runtime, slots, host):
    """Install only in a newly started/reattached controller, before its jobs."""
    original_execute = arm_runtime.execute
    original_remove = arm_runtime.remove_container
    original_output = arm_runtime.docker_output
    states = {}

    def execute(argv, cwd, timeout_s, log_prefix, **kwargs):
        if IMAGE not in argv or "/opt/mriseqbench/simulate" not in argv:
            return original_execute(argv, cwd, timeout_s, log_prefix, **kwargs)
        mounts = {}
        for i, arg in enumerate(argv):
            if arg == "--mount":
                fields = dict(
                    x.split("=", 1) for x in argv[i + 1].split(",") if "=" in x
                )
                mounts[fields["target"]] = fields["source"]
        name = argv[argv.index("--name") + 1]
        holder = slots.holders[threading.get_ident()]
        stop = kwargs.get("stop_when")
        try:
            result, record = dispatch(
                holder.get("host", host),
                mounts["/input"],
                mounts["/output"],
                Path(cwd).parent / "remote-simulations",
                holder["slot"],
                cancelled=stop.is_set if stop else lambda: False,
                timeout=timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 -- unknown outcomes must retain the slot
            # An unknown transport outcome must not free/reuse a physical slot
            # or become a scientific sequence failure.
            states[name] = {
                "Running": True,
                "OOMKilled": False,
                "ExitCode": 125,
                "Error": "Remote simulation transport/cleanup: " + str(exc),
            }
            result = {"status": "launch_error", "returncode": None, "error": str(exc)}
            prefix = Path(log_prefix)
            prefix.with_suffix(".stdout.log").write_text("")
            prefix.with_suffix(".stderr.log").write_text(str(exc) + "\n")
            prefix.with_suffix(".execution.json").write_text(json.dumps(result) + "\n")
            return result
        states[name] = result["container_state"]
        prefix = Path(log_prefix)
        for source, suffix in [
            ("stdout.log", ".stdout.log"),
            ("stderr.log", ".stderr.log"),
        ]:
            shutil.copy2(record / "returned" / source, prefix.with_suffix(suffix))
        prefix.with_suffix(".execution.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return result

    def output(argv, timeout=120):
        if argv and argv[0] == "inspect" and argv[-1] in states:
            return json.dumps(states[argv[-1]])
        return original_output(argv, timeout)

    def remove(name, *args, **kwargs):
        if name in states:
            return not states[name]["Running"]
        return original_remove(name, *args, **kwargs)

    arm_runtime.execute, arm_runtime.docker_output, arm_runtime.remove_container = (
        execute,
        output,
        remove,
    )
