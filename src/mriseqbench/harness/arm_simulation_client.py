"""Standalone client for the selected arm's fixed simulator adapter."""

import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def main():
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "sequence.seq")
    if not source.is_file() or source.stat().st_size > 16777216:
        raise ValueError("expected a Pulseq file <=16 MiB")
    request = urllib.request.Request(
        "http://127.0.0.1:8765/simulate/simulate",
        data=source.read_bytes(),
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        text = exc.read(16384).decode("utf-8", errors="replace")
        try:
            detail = json.loads(text)
            text = detail.get("stderr") or detail.get("error") or text
        except (ValueError, AttributeError):
            pass
        raise ValueError(f"HTTP {exc.code}: {text}") from exc
    directory = Path("simulation").resolve()
    directory.mkdir(exist_ok=True)
    for filename, content in result.pop("artifacts").items():
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory) or path == directory:
            raise ValueError("invalid simulator artifact filename")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(content, validate=True))
    print(json.dumps({**result, "output": str(directory)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, urllib.error.URLError) as exc:
        print(f"simulation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
