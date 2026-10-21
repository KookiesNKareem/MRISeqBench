"""Create frozen AI excitation references and adversarial judge controls."""

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path

import numpy as np

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import evaluate
from mriseqbench.harness.prepare import prepare_inputs, prompt
from mriseqbench.io import write_json
from tests.fixtures.rf_excitation_oracle import independent_response
from tests.fixtures.rf_excitation_sequences import make_sequence

ROOT = Path(__file__).resolve().parents[2]
HARDWARE = ("low_field", "standard_1_5t", "high_performance")
MUTATIONS = (None, "uniform", "missing_i", "shift", "wrong_amplitude", "no_rf")


def run(output, hardware=HARDWARE, mutations=MUTATIONS):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog = Catalog(ROOT)
    manifest = {}
    for pattern in (
        "src/mriseqbench/**/*.py",
        "src/mriseqbench/**/*.jl",
        "benchmark/**/*.yaml",
        "benchmark/objects/*.json",
        "benchmark/references/rf_excitation_2d/target.npz",
        "tests/fixtures/*rf_excitation*.py",
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
        case = catalog.resolve(
            catalog.get("tasks", "rf_excitation_2d"), hardware_ref=hw
        )
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
            oracle = None
            if mutation is None and result.get("numerically_converged"):
                replay = max(
                    (directory / "evaluation/rf_excitation").glob("replay-*.npz")
                )
                with np.load(replay) as data:
                    indices = np.unique(
                        np.r_[
                            np.linspace(0, len(data["mxy"]) - 1, 7, dtype=int),
                            np.argmax(data["target"]),
                        ]
                    )
                    positions = data["positions_m"][indices]
                    native = data["mxy"][indices]
                independent = independent_response(
                    submission, case, positions, max_step=0.25e-6
                )
                fine = independent_response(
                    submission, case, positions, max_step=0.125e-6
                )
                oracle = {
                    "native_max_absolute_error": float(np.max(np.abs(native - fine))),
                    "oracle_refinement_error": float(
                        np.max(np.abs(independent - fine))
                    ),
                    "absolute_error_limit": 0.001,
                    "passed": bool(
                        np.max(np.abs(native - fine)) <= 0.001
                        and np.max(np.abs(independent - fine)) <= 1e-5
                    ),
                }
                np.savez_compressed(
                    directory / "independent_oracle.npz",
                    positions_m=positions,
                    native=native,
                    independent=fine,
                )
                write_json(directory / "independent_oracle.json", oracle)
            checks_pass = all(c["passed"] for c in result["checks"])
            expected = (
                checks_pass and result.get("numerically_converged") and oracle["passed"]
                if mutation is None
                else result["status"] == "failed"
            )
            row = {
                "hardware": hw,
                "control": mutation or "positive",
                "expected_outcome": bool(expected),
                "status": result["status"],
                "metrics": result["metrics"],
                "failed_checks": [
                    c["name"] for c in result["checks"] if not c["passed"]
                ],
                "sequence_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
                "seconds": time.monotonic() - started,
                "independent_oracle": oracle,
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
            args.output, args.hardware, (None,) if args.positives_only else MUTATIONS
        )
        else 1
    )
