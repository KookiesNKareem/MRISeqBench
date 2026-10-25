"""GPU replay of S3 references, design variants and negative controls.

Each control is ``task:hardware:label[:mutation[:json-overrides]]``; the
sequence comes from the reference author with that mutation/options. Use
``--image-diagnostics`` for calibration: gate failures (mask, timing, probes)
are recorded, then forced to pass so the achieved-image metrics are still
measured. Such runs are calibration evidence, never verdicts.
"""

import argparse
import json
import time
import traceback
from pathlib import Path

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import evaluate, s1_koma, s3
from mriseqbench.io import write_json
from tests.fixtures import reference128_advanced

ROOT = Path(__file__).resolve().parents[1]
KEYS = (
    "case_complex_image_nrmse",
    "case_magnitude_image_nrmse",
    "case_phase_error_rad",
    "case_min_tissue_signal_fraction",
    "case_contrast_retention",
    "case_outside_object_energy_fraction",
    "selection_leakage_ratio",
    "diagnostic_signal_fraction",
)


def parse(spec):
    parts = spec.split(":", 4)
    task, hardware, label = parts[:3]
    mutation = parts[3] if len(parts) > 3 and parts[3] else None
    overrides = json.loads(parts[4]) if len(parts) > 4 else {}
    return task, hardware, label, mutation, overrides


def bypass(gates):
    """Record and force gate checks so image metrics are measured (calibration)."""
    inspect, probes = s3.inspect_sequence, s3.mechanism_probes

    def forced(checks, stage):
        for c in checks:
            if c.get("passed") is not True:
                gates.append(f"{stage}:{c['name']}")
                c["passed"] = True
        return checks

    def inspector(path, case):
        a = inspect(path, case)
        forced(a.checks, "sequence")
        return a

    def prober(acq, case, materials, properties):
        checks, diagnostic, response = probes(acq, case, materials, properties)
        return forced(checks, "probe"), diagnostic, response

    s3.inspect_sequence, s3.mechanism_probes = inspector, prober


def probe_only(sequence, case):
    from mriseqbench.evaluation.s1_phantom import phantom

    acq = s3.inspect_sequence(sequence, case)
    spins = phantom(case, ROOT, 4)
    checks, diagnostic, _ = s3.mechanism_probes(
        acq, case, spins.materials, spins.properties
    )
    return {
        "sequence_failed": [c["name"] for c in acq.checks if not c["passed"]],
        "probe_failed": [c["name"] for c in checks if not c["passed"]],
        **diagnostic,
        "probe_details": [c.get("detail") for c in checks],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("controls", nargs="*")
    parser.add_argument(
        "--file", type=Path, help="Control specs, one per line ('#' comments)"
    )
    parser.add_argument("--image-diagnostics", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    catalog = Catalog(ROOT)
    cases = {
        (c["task"]["id"], c["hardware"]["id"]): c
        for c in catalog.cases(catalog.get("suites", "full"))
    }
    gates = []
    if args.image_diagnostics:
        bypass(gates)
    specs = list(args.controls)
    if args.file:
        specs += [
            line.strip()
            for line in args.file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    for spec in specs:
        task, hardware, label, mutation, overrides = parse(spec)
        directory = args.out / task / hardware / label
        directory.mkdir(parents=True, exist_ok=True)
        start, row = time.monotonic(), {"task": task, "hardware": hardware}
        row.update(label=label, mutation=mutation, overrides=overrides)
        gates.clear()
        try:
            case = cases[task, hardware]
            sequence = directory / "sequence.seq"
            reference128_advanced.build(case, mutation, **overrides).write(
                str(sequence)
            )
            if args.probe_only:
                row.update(probe_only(sequence, case))
            else:
                report = evaluate(sequence, case, ROOT, directory / "evaluation")
                write_json(directory / "verdict.json", report)
                metrics = report.get("metrics", {})
                row.update(
                    status=report["status"],
                    gates_forced=list(gates),
                    failed=[
                        c["name"]
                        for group in ("checks", "application_checks", "image_checks")
                        for c in report.get(group, [])
                        if c.get("passed") is not True
                    ],
                    failure_kind=report.get("failure_kind"),
                    **{k: metrics[k] for k in KEYS if k in metrics},
                )
        except Exception:  # noqa: BLE001 - keep each control independent
            row.update(status="error", error=traceback.format_exc()[-2000:])
        row["seconds"] = round(time.monotonic() - start, 1)
        write_json(directory / "row.json", row)
        print(json.dumps(row, default=float), flush=True)
    if s1_koma._WORKER is not None:
        s1_koma._WORKER.close()


if __name__ == "__main__":
    main()
