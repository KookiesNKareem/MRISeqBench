"""Explicit filesystem grants, workspace-only writes and loopback gateways.

macOS policy is derived from koma-agent-bench/harness/sandbox.py. Linux uses
unprivileged bubblewrap instead of altering host users, DNS or permissions.
"""

import json
import shutil
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

from ..process import expand

SYSTEM_READ_MAC = (
    "/System",
    "/usr",
    "/bin",
    "/sbin",
    "/Library/Frameworks",
    "/Library/Apple",
    "/Library/Developer",
    "/private/etc",
    "/private/var/db/timezone",
    "/dev",
    "/opt/homebrew",
)
SYSTEM_READ_LINUX = (
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/localtime",
)


def real(path):
    return Path(path).expanduser().resolve()


def inside(path, parent):
    return path == parent or parent in path.parents


def configured_path(value, root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def grants(config, command, root, workspace):
    paths = [real(sys.base_prefix), real(Path(sys.executable).parent.parent)]
    executable = (
        shutil.which(command[0]) if not Path(command[0]).is_absolute() else command[0]
    )
    if executable:
        paths += [real(executable), Path(executable).absolute()]
    for path in expand(config.read_paths, root, workspace):
        target = configured_path(path, root)
        if not target.exists():
            raise ValueError(f"sandbox read grant does not exist: {target}")
        paths.append(target)
    # Broad grants would expose other runs, protected evaluation state or credentials.
    for path in paths:
        if path.is_dir() and (
            inside(real(root), path) or inside(real(workspace.parent), path)
        ):
            raise ValueError(f"sandbox read grant is too broad: {path}")
        if inside(path, real(workspace.parent)) and not inside(path, real(workspace)):
            raise ValueError(f"sandbox read grant exposes protected run state: {path}")
    return sorted(set(paths), key=str)


def seatbelt_profile(
    workspace, read_paths=(), deny_paths=(), ports=(), protected_paths=()
):
    workspace = real(workspace)
    quote = lambda p: json.dumps(str(p), ensure_ascii=False)
    lines = [
        "(version 1)",
        "(allow default)",
        "(deny file-read-data)",
        "(deny file-write*)",
        "(deny network-outbound)",
        "(deny network-inbound)",
        "(deny network-bind)",
        "(deny mach-lookup)",
        "(deny process-info*)",
        "(deny signal)",
        "(allow signal (target same-sandbox))",
        "(allow process-info* (target same-sandbox))",
        '(allow file-read-data (literal "/"))',
        f"(allow file-write* (subpath {quote(workspace)}))",
        '(allow file-write* (literal "/dev/null") (literal "/dev/tty") (literal "/dev/dtracehelper"))',
    ]
    for path in protected_paths:
        lines.append(f"(deny file-read-data (subpath {quote(real(path))}))")
    for path in [workspace, *[real(p) for p in SYSTEM_READ_MAC], *read_paths]:
        predicate = "subpath" if Path(path).is_dir() else "literal"
        lines.append(f"(allow file-read-data ({predicate} {quote(path)}))")
    for path in deny_paths:
        lines.append(f"(deny file-read-data (subpath {quote(real(path))}))")
    for port in sorted(set(ports)):
        if not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("sandbox ports must be integers in 1..65535")
        lines.append(f'(allow network-outbound (remote ip "localhost:{port}"))')
    # Prevent Apple-event/UI delegation to a process outside the sandbox.
    for path in ("/usr/bin/osascript", "/usr/bin/open"):
        lines.append(f"(deny process-exec (literal {quote(path)}))")
    return "\n".join(lines) + "\n"


def environment(agent, root, workspace, **values):
    """Do not inherit API keys, cloud credentials, proxy variables or Python paths."""
    workspace = real(workspace)
    tmp = workspace / ".tmp"
    tmp.mkdir(exist_ok=True)
    env = {
        "HOME": str(workspace),
        "TMPDIR": str(tmp),
        "TMP": str(tmp),
        "TEMP": str(tmp),
        "PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TERM": "dumb",
        "XDG_CACHE_HOME": str(workspace / ".cache"),
        "XDG_CONFIG_HOME": str(workspace / ".config"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    for name, value in agent.environment.items():
        if name in ("HOME", "TMPDIR", "TMP", "TEMP"):
            raise ValueError(
                f"agent environment cannot override workspace variable {name}"
            )
        env[name] = expand([value], root, workspace, **values)[0]
    return env


@contextmanager
def launch(config, command, root, workspace, control_dir, ports=()):
    """Return a command and isolation metadata. Never silently use unsandboxed execution."""
    backend = config.backend
    if backend == "auto":
        backend = (
            "macos"
            if sys.platform == "darwin"
            else "bubblewrap"
            if sys.platform.startswith("linux")
            else "unavailable"
        )
    if backend == "none":
        yield command, {"backend": "none", "isolated": False}
        return
    read_paths = grants(config, command, root, workspace)
    deny_paths = [
        configured_path(p, root) for p in expand(config.deny_paths, root, workspace)
    ]
    allowed_ports = sorted({*config.network_ports, *ports})
    control_dir = real(control_dir)
    control_dir.mkdir(parents=True, exist_ok=True)
    if backend == "macos":
        executable = shutil.which("sandbox-exec")
        if sys.platform != "darwin" or not executable:
            raise ValueError(
                "macOS sandbox-exec is unavailable; isolation was not bypassed"
            )
        profile = control_dir / "agent.sb"
        profile.write_text(
            seatbelt_profile(
                workspace,
                read_paths,
                deny_paths,
                allowed_ports,
                [root, Path(workspace).parent],
            )
        )
        yield (
            [executable, "-f", str(profile), *command],
            {
                "backend": "macos",
                "isolated": True,
                "read_paths": [str(p) for p in read_paths],
                "network_ports": allowed_ports,
                "profile": str(profile),
            },
        )
        return
    if (
        backend != "bubblewrap"
        or not sys.platform.startswith("linux")
        or not shutil.which("bwrap")
    ):
        raise ValueError("bubblewrap is unavailable; isolation was not bypassed")
    with ExitStack() as stack:
        from .relay import HostRelay

        args = [
            shutil.which("bwrap"),
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
        ]
        for path in SYSTEM_READ_LINUX:
            if Path(path).exists():
                args += ["--ro-bind", path, path]
        # Hide catalogs/control even if they live beneath a public system mount.
        for path in (real(root), real(workspace.parent)):
            args += ["--tmpfs", str(path)]
        for path in read_paths:
            args += ["--ro-bind", str(path), str(path)]
        args += ["--bind", str(real(workspace)), str(real(workspace))]
        for path in deny_paths:
            if path.is_dir():
                args += ["--tmpfs", str(path)]
            elif path.exists():
                args += ["--ro-bind", "/dev/null", str(path)]
        # TCP inside the isolated network namespace reaches only these Unix relays.
        relay_pairs = []
        for port in allowed_ports:
            relay = stack.enter_context(HostRelay(port, control_dir))
            socket_path = f"/run/mriseqbench/port-{port}.sock"
            args += ["--ro-bind", str(relay.path), socket_path]
            relay_pairs.append([port, socket_path])
        args += ["--chdir", str(real(workspace))]
        if relay_pairs:
            helper = control_dir / "namespace_relay.py"
            helper.write_text(Path(__file__).with_name("relay.py").read_text())
            args += ["--ro-bind", str(helper), "/run/mriseqbench/relay.py"]
            args += [
                "--",
                sys.executable,
                "/run/mriseqbench/relay.py",
                json.dumps(relay_pairs),
                *command,
            ]
        else:
            args += ["--", *command]
        yield (
            args,
            {
                "backend": "bubblewrap",
                "isolated": True,
                "read_paths": [str(p) for p in read_paths],
                "network_ports": allowed_ports,
            },
        )
