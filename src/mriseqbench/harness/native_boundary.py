"""Focused native macOS boundary around the private judge (also wraps Julia)."""

import json
import re
import sys
import tempfile
from pathlib import Path


def isolate_judge(
    command, *, workspace, judge_roots, public_paths, feedback_port, timing_log=None
):
    """Keep author files/libraries readable; deny judge files and other local services.

    Wrap the whole agent and its persistent author process, not just shell tools.
    Trusted grading/simulation workers remain outside this boundary and exchange
    submission bytes and public results through the per-trial feedback endpoint.
    """
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise RuntimeError("Native judge isolation requires the macOS sandbox")
    if not 0 < feedback_port < 65536:
        raise ValueError("Invalid per-trial feedback port")
    workspace = Path(workspace).resolve()
    public = {workspace, *(Path(path).resolve() for path in public_paths)}
    roots = {Path(path).resolve() for path in judge_roots}
    if not roots or any(workspace == root for root in roots):
        raise ValueError("The author workspace must be separate from the judge")
    exceptions = " ".join(
        f"(require-not ({'subpath' if path.is_dir() else 'literal'} {json.dumps(str(path))}))"
        for path in sorted(public)
    )
    rules = ["(version 1)", "(allow default)", "(deny file-write*)"]
    rules.append(f"(allow file-write* (subpath {json.dumps(str(workspace))}))")
    # The standalone Codex engine persists session state, while authors may
    # write only their workspace. Never reopen managed source worktrees.
    codex_home = Path.home() / ".codex"
    rules.append(
        f"(allow file-write* (require-all (subpath {json.dumps(str(codex_home))}) "
        f"(require-not (subpath {json.dumps(str(codex_home / 'worktrees'))}))))"
    )
    rules.append('(allow file-write* (literal "/dev/null") (literal "/dev/ptmx") (regex #"^/dev/tty"))')
    if timing_log is not None:
        rules.append(f"(allow file-write* (literal {json.dumps(str(Path(timing_log).resolve()))}))")
    for root in sorted(roots):
        rules.append(
            f"(deny file-read* (require-all "
            f"(subpath {json.dumps(str(root))}) {exceptions}))"
        )
    # Native workers also put transient requests/signals in the OS temp folder.
    # Match the prefix so files created after agent startup stay private.
    for directory in {Path("/private/tmp"), Path(tempfile.gettempdir()).resolve()}:
        pattern = json.dumps("^" + re.escape(str(directory)) + "/mriseqbench-")
        rules.append(
            f"(deny file-read* file-write* (require-all (regex #{pattern}) {exceptions}))"
        )
    writable = f"(require-not (subpath {json.dumps(str(workspace))}))"
    if timing_log is not None:
        writable += (
            f" (require-not (literal {json.dumps(str(Path(timing_log).resolve()))}))"
        )
    # Public libraries and transport code remain read-only, including shared
    # package caches used by the trusted workers.
    for root in sorted(roots | public | {Path.home() / ".julia"}):
        if root == workspace:
            continue
        kind = "subpath" if root.is_dir() else "literal"
        rules.append(
            f"(deny file-write* (require-all ({kind} {json.dumps(str(root))}) {writable}))"
        )
    # Parent-directory metadata is needed to resolve allowed nested libraries.
    rules.append("(allow file-read-metadata)")
    # The dashboard exposes judge/reference previews; authors may contact only
    # their own feedback/simulation endpoint among loopback services.
    rules += [
        "(deny network-outbound (remote unix-socket))",
        # macOS resolves public hostnames through this system socket.
        '(allow network-outbound (remote unix-socket (literal "/private/var/run/mDNSResponder")))',
        '(allow network-outbound (remote unix-socket (literal "/var/run/mDNSResponder")))',
        '(deny network-outbound (remote ip "localhost:*"))',
        f'(allow network-outbound (remote ip "localhost:{feedback_port}"))',
    ]
    return ["/usr/bin/sandbox-exec", "-p", "\n".join(rules), *map(str, command)]
