"""Copied into the workspace; a standalone, stdlib-only feedback client."""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with temporary.open("x") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def present(result, receipt, full=False):
    evaluation = result.get("evaluation", result)
    # These are the public verdict's canonical check groups. Nested frame
    # reports and ideal/case aliases would duplicate checks already listed here.
    outcomes = [
        {**check, "check_group": group}
        for group in ("checks", "application_checks", "image_checks")
        for check in evaluation.get(group, [])
        if "passed" in check
    ]
    summary = {
        "status": result.get("status", "error"),
        "failed_checks": [c for c in outcomes if c["passed"] is False],
        "unresolved_checks": [c for c in outcomes if c["passed"] is None],
        "full_receipt": str(receipt),
    }
    for key in ("attempt", "submissions_remaining", "reason", "error", "runtime"):
        if key in result:
            summary[key] = result[key]
    if "evaluation" in result:
        summary["evaluation_status"] = evaluation.get("status")
    if full:
        summary.update(result)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kind", choices=["lint", "check", "submit", "status", "feedback"]
    )
    parser.add_argument(
        "--full", action="store_true", help="print the complete response (always saved)"
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    config = json.loads((root / ".harness.json").read_text())
    kind = args.kind
    query = kind in {"status", "feedback"}
    route = "receipt" if kind == "feedback" else kind
    body = None
    if not query:
        path = root / "sequence.seq"
        if not path.is_file():
            print("sequence.seq is missing", file=sys.stderr)
            return 1
        if path.stat().st_size > config["max_submission_bytes"]:
            print("sequence.seq exceeds the submission size limit", file=sys.stderr)
            return 1
        body = path.read_bytes()
    req = urllib.request.Request(
        f"http://127.0.0.1:{config['port']}{config.get('route_prefix', '')}/{route}",
        data=body,
        method="GET" if query else "POST",
        headers={
            "X-MRISeqBench-Token": config["token"],
            "Content-Type": "application/octet-stream",
        },
    )
    try:
        try:
            response = urllib.request.urlopen(
                req, timeout=None if kind == "submit" else config["timeout_s"] + 5
            )
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            result = json.loads(response.read())
        if not isinstance(result, dict):
            raise TypeError("feedback response must be an object")
        if kind == "status":
            output = result
        else:
            directory = root / "feedback"
            directory.mkdir(exist_ok=True)
            receipt = directory / f"{kind}-{uuid.uuid4().hex}.json"
            atomic_json(receipt, result)
            atomic_json(directory / f"latest-{kind}.json", result)
            output = present(result, receipt.relative_to(root), args.full)
        # Publish the full receipt before stdout can be cut off by a pipe.
        try:
            print(json.dumps(output, indent=2, allow_nan=False), flush=True)
        except BrokenPipeError:
            with open(os.devnull, "w") as sink:
                os.dup2(sink.fileno(), sys.stdout.fileno())
        return (
            0
            if result.get("status")
            in (
                "submitted",
                "retry_required",
                "preflight_passed",
                "lint_passed",
                "passed",
                "smoke_passed",
                "running",
                "preparing",
                "closed",
            )
            else 1
        )
    except (urllib.error.URLError, ValueError, TypeError, OSError) as exc:
        print(f"harness request failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
