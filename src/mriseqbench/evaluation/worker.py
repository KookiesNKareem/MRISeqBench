"""Killable host-side preflight/evaluation worker used by the feedback server."""

import argparse
import json
from pathlib import Path

from ..io import write_json
from ..models import Backend
from . import evaluate, preflight


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["preflight", "evaluation"], required=True)
    p.add_argument("--case", required=True, type=Path)
    p.add_argument("--submission", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--workspace", required=True, type=Path)
    p.add_argument("--backend", type=Path)
    args = p.parse_args()
    case = json.loads(args.case.read_text())
    if args.mode == "preflight":
        checks = preflight(args.submission, case)
        result = {
            "case_id": case["case_id"],
            "status": "preflight_passed"
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
