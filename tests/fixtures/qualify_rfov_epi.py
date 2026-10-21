"""Frozen end-to-end qualification for the S3 2D RF reduced-FOV EPI task."""

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path

import numpy as np

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import evaluate
from mriseqbench.evaluation.s3 import mechanism_probes
from mriseqbench.evaluation.s3_sequence import inspect_sequence
from mriseqbench.harness.prepare import prepare_inputs, prompt
from mriseqbench.io import write_json
from tests.fixtures.rfov_epi_sequences import make_sequence

ROOT = Path(__file__).resolve().parents[2]
HARDWARE = ("low_field", "standard_1_5t", "high_performance")
EXPECTED = {
    "no_selection": {"two_dimensional_rf_encoding"},
    "one_dimensional_rf": {"two_dimensional_rf_encoding"},
    "wrong_polarity": {"single_shot_epi"},
    "full_fov": {"sampling_pattern"},
    "missing_line": {"sampling_pattern"},
    "shift_selection": {"spatial_selection", "rfov_inside_signal"},
    "wrong_receiver_phase": {"case_complex_image_nrmse", "case_phase_error_rad"},
}


def run(output, hardware=HARDWARE, mutations=(None, *EXPECTED)):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog = Catalog(ROOT)
    manifest = {}
    for pattern in (
        "src/mriseqbench/**/*.py",
        "src/mriseqbench/**/*.jl",
        "benchmark/**/*.yaml",
        "benchmark/objects/*.json",
        "tests/fixtures/*rfov_epi*.py",
        "Project.toml",
        "Manifest.toml",
    ):
        for path in ROOT.glob(pattern):
            manifest[str(path.relative_to(ROOT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    write_json(output / "source_manifest.json", manifest)
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for name in sorted(manifest):
            archive.add(ROOT / name, arcname=name, recursive=False)
    rows = []
    for hw in hardware:
        case = catalog.resolve(catalog.get("tasks", "rfov_epi"), hardware_ref=hw)
        for mutation in mutations:
            directory = output / f"{hw}--{mutation or 'positive'}"
            directory.mkdir()
            write_json(directory / "case.json", case)
            (directory / "PROMPT.md").write_text(prompt(case))
            prepare_inputs(directory, case, ROOT)
            submission = directory / "sequence.seq"
            make_sequence(case, mutation).write(str(submission))
            started = time.monotonic()
            result = evaluate(submission, case, ROOT, directory / "evaluation")
            write_json(directory / "verdict.json", result)
            failed = {
                c["name"]
                for key, checks in result.items()
                if key.endswith("checks") and isinstance(checks, list)
                for c in checks
                if not c["passed"]
            }
            expected = (
                result["status"] == "passed"
                if mutation is None
                else result["status"] == "failed" and bool(failed & EXPECTED[mutation])
            )
            if mutation == "one_dimensional_rf":
                acq = inspect_sequence(submission, case)
                checks, metrics, _response = mechanism_probes(
                    acq, case, ["water"], np.array([[1.0, 0.8, 0.08]])
                )
                write_json(
                    directory / "independent_axis_assay.json",
                    {"checks": checks, "metrics": metrics},
                )
                expected &= any(
                    c["name"] == "spatial_selection" and not c["passed"] for c in checks
                )
            row = {
                "hardware": hw,
                "control": mutation or "positive",
                "expected_outcome": bool(expected),
                "status": result["status"],
                "metrics": result["metrics"],
                "failed_checks": sorted(failed),
                "sequence_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
                "seconds": time.monotonic() - started,
            }
            rows.append(row)
            write_json(
                output / "summary.json",
                {
                    "all_expected": all(r["expected_outcome"] for r in rows),
                    "complete": len(rows) == len(hardware) * len(mutations),
                    "rows": rows,
                },
            )
            print(json.dumps(row), flush=True)
    return all(row["expected_outcome"] for row in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware", nargs="+", default=HARDWARE)
    parser.add_argument("--positives-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        0
        if run(
            args.output,
            args.hardware,
            (None,) if args.positives_only else (None, *EXPECTED),
        )
        else 1
    )
