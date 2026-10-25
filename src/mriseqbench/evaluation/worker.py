"""Killable host-side preflight/evaluation worker used by the feedback server."""

import argparse
import json
import resource
import sys
from pathlib import Path

from ..io import write_json
from ..models import Backend
from . import evaluate, preflight

PREFLIGHT_MEMORY = 4 * 1024**3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["lint", "preflight", "evaluation"], required=True)
    p.add_argument("--case", required=True, type=Path)
    p.add_argument("--submission", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--workspace", required=True, type=Path)
    p.add_argument("--backend", type=Path)
    args = p.parse_args()
    case = json.loads(args.case.read_text())
    if args.mode in ("lint", "preflight"):
        # Like a scanner's file loader, parsing is bounded; only the submitted
        # bytes can exhaust it (callers bound time the same way).
        try:
            resource.setrlimit(resource.RLIMIT_AS, (PREFLIGHT_MEMORY, PREFLIGHT_MEMORY))
        except (ValueError, OSError):
            if sys.platform.startswith("linux"):  # Campaign hosts must enforce it.
                raise
        try:
            checks = preflight(
                args.submission, case, hardware_limits=args.mode != "lint"
            )
        except MemoryError:
            checks = [
                {
                    "name": "preflight_memory_limit",
                    "passed": False,
                    "detail": "parsing the file exceeded the preflight memory limit",
                }
            ]
        result = {
            "case_id": case["case_id"],
            "status": ("lint_passed" if args.mode == "lint" else "preflight_passed")
            if all(c["passed"] for c in checks)
            else "failed",
            "checks": checks,
            "metrics": {},
        }
    else:
        backend = (
            Backend.model_validate(json.loads(args.backend.read_text()))
            if args.backend
            else None
        )
        result = evaluate(
            args.submission, case, args.root, args.workspace / "physical", backend
        )
    write_json(args.workspace / "result.json", result)


if __name__ == "__main__":
    main()
