"""Independent nominal-grid KomaMRI controls for the S1 P2/P3 image contract.

The candidate cutoffs are saved before simulation. Participant submissions are
not calibration inputs. All outcomes, including unexpected failures, are retained.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mriseqbench.catalog import Catalog

ROOT = Path(__file__).resolve().parents[2]
JOBS = [
    ("gre_t1w", "tissue_discs", "b0_smooth"),
    ("spin_echo_t2w", "tissue_discs", "b0_offset"),
    ("bssfp", "nested_ellipses", "b0_localized"),
    ("gre_t1w", "tissue_discs", "rigid_motion"),
    ("spin_echo_t2w", "tissue_discs", "fields_and_motion"),
    ("bssfp", "nested_ellipses", "b0_drift"),
]


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    catalog = Catalog(ROOT)
    loader = importlib.util.spec_from_file_location(
        "controls", ROOT / "tests/fixtures/s1_sequences.py"
    )
    controls = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(controls)
    policy = catalog.get("evaluators", "s1").model_dump(mode="json")
    (out / "frozen-policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    jobs = []
    for fam, obj, physics in JOBS:
        case = catalog.resolve(
            catalog.get("tasks", fam), catalog.get("physics", physics), obj
        )
        directory = out / f"{fam}--{physics}"
        directory.mkdir()
        sequence = directory / "sequence.seq"
        controls.make_sequence(case).write(str(sequence))
        (directory / "case.json").write_text(json.dumps(case, indent=2) + "\n")
        jobs.append((directory, sequence))

    def once(job):
        directory, sequence = job
        start = time.monotonic()
        with (directory / "worker.log").open("w") as log:
            execution = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mriseqbench.evaluation.worker",
                    "--mode",
                    "evaluation",
                    "--case",
                    str(directory / "case.json"),
                    "--submission",
                    str(sequence),
                    "--root",
                    str(ROOT),
                    "--workspace",
                    str(directory / "evaluation"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result_path = directory / "evaluation/result.json"
        if execution.returncode != 0 or not result_path.exists():
            raise RuntimeError(
                f"Calibration interrupted without a verdict: {directory}"
            )
        result = json.loads(result_path.read_text())
        row = {
            "control": directory.name,
            "status": result["status"],
            "seconds": time.monotonic() - start,
            "worker_returncode": execution.returncode,
            "submission_sha256": hashlib.sha256(sequence.read_bytes()).hexdigest(),
            "result": str(result_path),
            "metrics": result.get("metrics", {}),
            "reference": result.get("case_phase_reference"),
        }
        print(json.dumps(row), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(once, jobs))
    (out / "summary.json").write_text(
        json.dumps(
            {
                "records": records,
                "all_passed": all(r["status"] == "passed" for r in records),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    run(parser.parse_args().destination.resolve())
