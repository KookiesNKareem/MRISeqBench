"""Container-local client for the trial's persistent Julia process."""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

DIRECTORY = Path("/tmp/mriseqbench-julia")


def wait_for(path, deadline):
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("persistent Julia session did not respond")
        time.sleep(0.02)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("-e", dest="code")
    parser.add_argument("script", nargs="?")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    if not options.ready and (options.code is None) == (options.script is None):
        parser.error("supply either -e CODE or a Julia script")
    deadline = time.monotonic() + options.timeout
    try:
        wait_for(DIRECTORY / "ready.json", deadline)
        if options.ready:
            print((DIRECTORY / "ready.json").read_text(), end="")
            return 0
        request = Path(tempfile.mkdtemp(prefix="request-", dir=DIRECTORY))
        try:
            # JSON string/array literals are also TOML values. No Julia source
            # interpolation is used for paths, arguments or code.
            fields = {
                "mode": "eval" if options.code is not None else "file",
                "value": options.code if options.code is not None else options.script,
                "cwd": os.getcwd(),
                "args": options.args,
            }
            temporary = request / "request.tmp"
            temporary.write_text(
                "".join(
                    f"{key} = {json.dumps(value, ensure_ascii=False)}\n"
                    for key, value in fields.items()
                )
            )
            temporary.rename(request / "request.toml")
            wait_for(request / "done", deadline)
            sys.stdout.write((request / "stdout").read_text(errors="replace"))
            sys.stderr.write((request / "stderr").read_text(errors="replace"))
            return int((request / "done").read_text())
        finally:
            # On timeout the server may still be writing here; the enclosing
            # trial container is responsible for killing in-flight work.
            if (request / "done").exists():
                shutil.rmtree(request)
    except (OSError, TimeoutError) as exc:
        print(f"Julia session error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
