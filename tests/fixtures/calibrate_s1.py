"""Reproducible nominal-grid S1 controls; exits nonzero for any unexpected verdict.

Run: uv run python tests/fixtures/calibrate_s1.py runs/s1-koma-calibration
This is release validation, not agent performance or an exhaustive suite run.
"""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import evaluate
from mriseqbench.evaluation.s1_sequence import FAMILIES

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.util.spec_from_file_location(
    "s1_controls", ROOT / "tests/fixtures/s1_sequences.py"
)
controls = importlib.util.module_from_spec(loader)
loader.loader.exec_module(controls)


def run(destination, families=None):
    catalog = Catalog(ROOT)
    objects = [
        (f, o)
        for f in (
            "gre_t1w",
            "gre_t2star",
            "spin_echo_t2w",
            "bssfp",
            "rich_geometry_gre",
        )
        for o in ("uniform_disc", "tissue_discs", "nested_ellipses")
    ]
    objects += [("volume_gre", o) for o in ("uniform_sphere", "volume_ellipsoids")]
    jobs = [
        (f, o, research, None, None) for f, o in objects for research in (False, True)
    ]
    jobs += [
        ("bssfp", "uniform_disc", False, h, None)
        for h in ("low_field", "standard_1_5t", "high_performance")
    ]
    jobs += [
        (
            f,
            "uniform_sphere" if f == "volume_gre" else "tissue_discs",
            False,
            None,
            mutation,
        )
        for f, mutation in (
            ("gre_t1w", "missing_line"),
            ("gre_t1w", "repeated_lines"),
            ("gre_t1w", "echo_outside_adc"),
            ("gre_t1w", "wrong_echo_time"),
            ("gre_t1w", "no_spoiling"),
            ("gre_t1w", "receiver_phase"),
            ("spin_echo_t2w", "no_refocusing"),
            ("spin_echo_t2w", "lying_refocusing_label"),
            ("spin_echo_t2w", "refocus_late"),
            ("spin_echo_t2w", "wrong_echo_time"),
            ("bssfp", "unbalanced"),
            ("bssfp", "truncated_final_tr"),
            ("bssfp", "receiver_phase"),
            ("volume_gre", "flat_volume"),
            ("rich_geometry_gre", "short_tr_coherence"),
        )
    ]
    if families:
        jobs = [job for job in jobs if job[0] in families]
    records = []
    destination.mkdir(parents=True, exist_ok=False)
    for family, obj, research, hardware, mutation in jobs:
        task = family + ("_research" if research else "")
        case = catalog.resolve(
            catalog.get("tasks", task),
            catalog.get("physics", "ideal"),
            obj,
            hardware_ref=hardware,
        )
        name = "--".join(x for x in (task, obj, hardware, mutation) if x)
        directory = destination / name
        directory.mkdir()
        submission = directory / "sequence.seq"
        controls.make_sequence(case, mutation).write(str(submission))
        start = time.monotonic()
        result = evaluate(submission, case, ROOT, directory / "evaluation")
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        expected = "failed" if mutation else "passed"
        record = {
            "control": name,
            "expected": expected,
            "actual": result["status"],
            "seconds": time.monotonic() - start,
            "sequence_valid": result.get("sequence_valid"),
            "ideal_imaging_success": result.get("ideal_imaging_success"),
            "case_imaging_success": result.get("imaging_success_under_case_physics"),
            "metrics": result.get("metrics"),
            "intervals": result.get("numerical_intervals"),
            "reason": result.get("reason"),
            "result": str((directory / "result.json").resolve()),
            "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
        }
        records.append(record)
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in ("control", "expected", "actual", "seconds", "reason")
                }
            ),
            flush=True,
        )
        (destination / "summary.json").write_text(
            json.dumps(
                {
                    "records": records,
                    "all_expected": all(r["expected"] == r["actual"] for r in records),
                },
                indent=2,
            )
            + "\n"
        )
    return all(r["expected"] == r["actual"] for r in records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--family",
        action="append",
        choices=sorted(FAMILIES),
        help="Evaluate only this family",
    )
    args = parser.parse_args()
    raise SystemExit(0 if run(args.destination, args.family) else 1)
