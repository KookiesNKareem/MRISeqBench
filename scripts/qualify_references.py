"""Build and GPU-replay the production references, retaining failed candidates.

Run one shard per CUDA device. Reuse needs matching inputs and the same or an
explicitly reviewed compatible source hash.
Nothing is promoted merely because a pulse was constructed or an image saved.
"""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import evaluate, s1_koma
from mriseqbench.io import write_json
from tests.fixtures import reference128_advanced, reference128_readouts

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--hardware", nargs="*")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--budget",
        type=float,
        help="Optional per-case limit in seconds; default runs to completion",
    )
    parser.add_argument("--reuse", type=Path, action="append", default=[])
    parser.add_argument(
        "--reuse-compatible-source",
        action="append",
        default=[],
        metavar="SHA256",
        help="Explicitly allow passed receipts from a reviewed prior source hash",
    )
    args = parser.parse_args()
    if any(
        len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
        for value in args.reuse_compatible_source
    ):
        parser.error("--reuse-compatible-source requires a lowercase SHA-256")
    if (
        not args.build_only
        and os.environ.get("MRISEQBENCH_KOMA_DEVICE", "cuda") != "cuda"
    ):
        raise RuntimeError("Reference qualification requires CUDA")
    catalog = Catalog(ROOT)
    cases = catalog.cases(catalog.get("suites", "full"))
    cases = [
        c
        for c in cases
        if (not args.tasks or c["task"]["id"] in args.tasks)
        and (not args.hardware or c["hardware"]["id"] in args.hardware)
    ]
    # Obtain 2D results first; volumes use the same full scientific criteria.
    cases.sort(
        key=lambda c: (
            len(c["task"]["sequence"]["matrix"]),
            c["task"]["id"],
            c["hardware"]["id"],
        )
    )
    source = {
        str(p.relative_to(ROOT)): sha(p)
        for folder in ("src", "benchmark", "tests/fixtures")
        for p in (ROOT / folder).rglob("*")
        if p.is_file() and p.suffix in {".py", ".jl", ".yaml", ".json", ".npz"}
    }
    for name in ("Project.toml", "Manifest.toml", "pyproject.toml", "uv.lock"):
        if (ROOT / name).is_file():
            source[name] = sha(ROOT / name)
    source_sha = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    for previous in args.out.glob("source-shard-*.json"):
        if json.loads(previous.read_text()) != source:
            raise RuntimeError(
                "Output was created with different source; use a new --out"
            )
    write_json(args.out / f"source-shard-{args.shard}.json", source)
    for case in cases[args.shard :: args.shards]:
        if any(
            completed_result(path, case, source_sha, args.reuse_compatible_source)
            for path in args.reuse
        ):
            continue
        directory = args.out / case["task"]["id"] / case["hardware"]["id"]
        directory.mkdir(parents=True, exist_ok=True)
        receipt = directory / "receipt.json"
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            if prior.get("source_sha256") == source_sha and prior.get("complete"):
                continue
            raise RuntimeError(
                f"Changed source or incomplete receipt: use a new output: {directory}"
            )
        start = time.monotonic()
        row = {
            "task": case["task"]["id"],
            "hardware": case["hardware"]["id"],
            "matrix": case["task"]["sequence"]["matrix"],
            "source_sha256": source_sha,
            "visual_review": "pending",
            "complete": False,
        }
        case_path = directory / "case.json"
        if case_path.exists() and json.loads(case_path.read_text()) != case:
            raise RuntimeError(f"Changed case: use a new output: {directory}")
        write_json(case_path, case)
        try:
            author = (
                reference128_readouts
                if case["sequence_level"] in {"S1", "S2"}
                else reference128_advanced
            )
            sequence = directory / "sequence.seq"
            if not sequence.exists():
                author.build(case).write(str(sequence))
            row["sequence_sha256"] = sha(sequence)
            row["duration_s"] = float(author_duration(sequence))
            if args.build_only:
                row["status"] = "constructed_only"
            else:
                report = evaluate(
                    sequence,
                    case,
                    ROOT,
                    directory / "evaluation",
                    time_budget_s=args.budget,
                )
                write_json(directory / "verdict.json", report)
                row.update(
                    status=report["status"],
                    metrics=report.get("metrics", {}),
                    numerically_converged=report.get("numerically_converged"),
                    failed_checks=[
                        c
                        for group in ("checks", "application_checks", "image_checks")
                        for c in report.get(group, [])
                        if c.get("passed") is not True
                    ],
                    achieved_images=[
                        str(p.relative_to(directory))
                        for p in directory.rglob("images.npz")
                    ],
                )
                row["complete"] = True
                if s1_koma._WORKER is not None:
                    row["native_execution"] = s1_koma._WORKER.execution
        except Exception:  # noqa: BLE001 - retain each independent candidate failure
            row.update(status="unresolved", error=traceback.format_exc(), complete=True)
        row["seconds"] = time.monotonic() - start
        write_json(receipt, row)
        print(
            json.dumps(
                {k: row.get(k) for k in ("task", "hardware", "status", "seconds")}
            ),
            flush=True,
        )
    if s1_koma._WORKER is not None:
        s1_koma._WORKER.close()


def author_duration(path):
    from pypulseq import Sequence

    sequence = Sequence()
    sequence.read(str(path))
    return sequence.duration()[0]


def completed_result(root, case, source_sha, compatible_sources=()):
    """Leave already checked pulses in their original result folders."""
    directory = root / case["task"]["id"] / case["hardware"]["id"]
    receipt = directory / "receipt.json"
    if (directory / "recovered-receipt.json").exists():
        receipt = directory / "recovered-receipt.json"
    if not receipt.exists():
        return False
    row = json.loads(receipt.read_text())
    previous = json.loads((directory / "case.json").read_text())
    return (
        row.get("complete") is True
        and row.get("status") in {"passed", "failed"}
        and (
            row.get("source_sha256") == source_sha
            or row.get("status") == "passed"
            and row.get("source_sha256") in compatible_sources
        )
        and previous["task"]["sequence"] == case["task"]["sequence"]
        and all(
            previous[key] == case[key]
            for key in ("object", "hardware", "physics", "evaluation")
        )
        and row.get("sequence_sha256") == sha(directory / "sequence.seq")
    )


if __name__ == "__main__":
    main()
