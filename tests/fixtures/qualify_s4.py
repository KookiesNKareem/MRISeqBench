"""Freeze and replay independent cardiac S4 controls without altering acceptance.

Run from the benchmark virtual environment with --output and optional --tasks /
--hardware subsets. Every result retains its actual calibrated/provisional state.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import evaluate
from mriseqbench.evaluation.bart import executable as bart_executable
from mriseqbench.evaluation.bart import provenance as bart_provenance
from tests.fixtures.s4_sequences import MUTATIONS, make_sequence

FAMILIES = ("adiabatic_inversion", "t2_prepared", "t1_mapping", "t2_mapping")
HARDWARE = ("low_field", "standard_1_5t", "high_performance")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(output):
    tool = {"executable": bart_executable(), "identity": bart_provenance()}
    if output.exists():
        raise ValueError(
            "Refusing to overwrite a frozen qualification; choose a new output directory"
        )
    source = output / "source"
    source.mkdir(parents=True)
    (source / "reconstruction_tool.json").write_text(json.dumps(tool, indent=2) + "\n")
    for folder in ("src", "benchmark", "schemas"):
        shutil.copytree(
            ROOT / folder, source / folder, ignore=shutil.ignore_patterns("__pycache__")
        )
    for name in ("Project.toml", "Manifest.toml", "pyproject.toml"):
        shutil.copy2(ROOT / name, source / name)
    fixtures = source / "tests/fixtures"
    fixtures.mkdir(parents=True)
    for name in ("s4_sequences.py", "qualify_s4.py"):
        shutil.copy2(ROOT / "tests/fixtures" / name, fixtures / name)
    manifest = {
        str(path.relative_to(source)): sha(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return source


def qualification(result):
    groups = [
        result.get("checks", []),
        result.get("application_checks", []),
        result.get("image_checks", []),
    ]
    checks = [item for group in groups for item in group]
    return bool(
        result.get("sequence_valid")
        and result.get("numerically_converged")
        and checks
        and all(item["passed"] is True for item in checks)
    )


def applicable_mutations(family, hardware=None):
    for mutation in MUTATIONS:
        if mutation == "narrow_preparation" and hardware != "high_performance":
            continue
        if mutation == "bad_refocusing" and family not in {"t2_prepared", "t2_mapping"}:
            continue
        if mutation == "constant_contrast" and family not in {
            "t1_mapping",
            "t2_mapping",
        }:
            continue
        yield mutation


def run(args):
    output = args.output.resolve()
    manifest = json.loads((output / "source_manifest.json").read_text())
    for name, expected in manifest.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen qualification source bytes changed: {name}")
    catalog = Catalog(ROOT)
    rows = []
    for family in args.tasks:
        for hardware in args.hardware:
            for mutation in (
                [None, *list(applicable_mutations(family, hardware))]
                if args.negatives
                else [None]
            ):
                name = f"{family}--{hardware}--{mutation or 'positive'}"
                directory = output / "cases" / name
                directory.mkdir(parents=True)
                case = catalog.resolve(
                    catalog.get("tasks", family), hardware_ref=hardware
                )
                case["evaluation"]["spatial_seed"] = args.seed
                if args.samples is not None:
                    case["evaluation"]["spatial_samples"] = args.samples
                if args.replicates is not None:
                    case["evaluation"]["spatial_replicates"] = args.replicates
                case_path = directory / "case.json"
                case_path.write_text(json.dumps(case, indent=2) + "\n")
                path = directory / "sequence.seq"
                make_sequence(case, mutation).write(str(path))
                started = time.time()
                try:
                    result = evaluate(path, case, ROOT, directory / "evaluation")
                except Exception as error:  # noqa: BLE001 -- Retain evaluator errors as evidence.
                    result = {
                        "status": "evaluator_error",
                        "type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                (directory / "result.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                qualified = qualification(result)
                entry = {
                    "name": name,
                    "task": family,
                    "hardware": hardware,
                    "mutation": mutation,
                    "case_sha256": sha(case_path),
                    "sequence_sha256": sha(path),
                    "result_sha256": sha(directory / "result.json"),
                    "status": result["status"],
                    "qualified_control": qualified if mutation is None else None,
                    "negative_rejected": result["status"] == "failed" and not qualified
                    if mutation
                    else None,
                    "elapsed_s": time.time() - started,
                }
                rows.append(entry)
                (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(json.dumps(entry), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument(
        "--hardware", nargs="+", choices=HARDWARE, default=list(HARDWARE)
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Diagnostic override; otherwise retain the public contract",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        help="Diagnostic override; otherwise retain the public contract",
    )
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--negatives", action="store_true")
    parser.add_argument("--frozen-run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.frozen_run:
        run(args)
        return
    source = freeze(args.output)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source / "src") + os.pathsep + str(source)
    environment["MRISEQBENCH_BART"] = json.loads(
        (source / "reconstruction_tool.json").read_text()
    )["executable"]
    child = [
        sys.executable,
        str(source / "tests/fixtures/qualify_s4.py"),
        "--output",
        str(args.output),
        "--tasks",
        *args.tasks,
        "--hardware",
        *args.hardware,
        "--seed",
        str(args.seed),
        "--frozen-run",
    ]
    if args.samples is not None:
        child.extend(["--samples", str(args.samples)])
    if args.replicates is not None:
        child.extend(["--replicates", str(args.replicates)])
    if args.negatives:
        child.append("--negatives")
    with (args.output / "qualification.log").open("w") as log:
        completed = subprocess.run(
            child,
            cwd=source,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
