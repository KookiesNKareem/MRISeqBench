"""Copied into the workspace; a standalone, stdlib-only feedback client."""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    config = json.loads((root / ".harness.json").read_text())
    kind = sys.argv[1]
    path = root / "sequence.seq"
    if not path.is_file():
        print("sequence.seq is missing", file=sys.stderr)
        return 1
    if path.stat().st_size > config["max_submission_bytes"]:
        print("sequence.seq exceeds the submission size limit", file=sys.stderr)
        return 1
    req = urllib.request.Request(
        f"http://127.0.0.1:{config['port']}/{kind}",
        data=path.read_bytes(),
        method="POST",
        headers={
            "X-MRISeqBench-Token": config["token"],
            "Content-Type": "application/octet-stream",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=config["timeout_s"] + 5) as response:
            result = json.loads(response.read())
        print(json.dumps(result, indent=2))
        return (
            0
            if result.get("status")
            in (
                "submitted",
                "preflight_passed",
                "passed",
                "smoke_passed",
            )
            else 1
        )
    except (urllib.error.URLError, ValueError) as exc:
        print(f"harness request failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
