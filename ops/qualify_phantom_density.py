"""Compare two fixed native phantom densities before admitting benchmark trials.

This is offline numerical qualification, not an additional candidate criterion.
It never changes contracts or starts model agents. Use known reference sequences
for each task/scanner; passing one control does not qualify the entire suite.
"""

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from mriseqbench.evaluation import bart
from mriseqbench.evaluation.evaluate import evaluate
from mriseqbench.evaluation.s1 import IMPLEMENTATION_SHA256
from mriseqbench.evaluation.s1_koma import versions


def image_paths(report):
    entries = [report, *report.get("frames", [])]
    paths = [
        Path(row["artifacts"]["images"])
        for row in entries
        if row.get("artifacts", {}).get("images")
    ]
    return list(dict.fromkeys(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--maximum-change",
        type=float,
        default=0.02,
        help="Offline image difference limit, not trial acceptance",
    )
    args = parser.parse_args()
    source = json.loads(args.case.read_text())
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "sequence_sha256": hashlib.sha256(args.sequence.read_bytes()).hexdigest(),
        "case_sha256": hashlib.sha256(args.case.read_bytes()).hexdigest(),
        "signal_processing": "native_koma",
        "implementation_sha256": IMPLEMENTATION_SHA256,
        "scope": "offline density comparison; SAR not rechecked",
        "slab_mm": source["evaluation"].get("phantom_slab_mm"),
        "maximum_image_change": args.maximum_change,
        "stages": [],
    }
    paths = []
    for multiplier, name in [(1, "base"), (2, "double-density")]:
        case = deepcopy(source)
        contract = case["evaluation"]
        if contract["id"] == "rf_excitation":
            contract["phantom_z_samples"] = multiplier * (
                contract.get("phantom_z_samples") or 32
            )
        else:
            if contract.get("spatial_mode") != "fixed":
                parser.error("case must select the fixed-phantom contract")
            contract["spatial_samples"] *= multiplier
            contract["spatial_max_samples"] = contract["spatial_samples"]
        folder = args.output / name
        folder.mkdir()
        (folder / "case.json").write_text(json.dumps(case, indent=2))
        print(f"Starting {name}: {contract}", flush=True)
        started = time.monotonic()
        try:
            if "runtime" not in metadata:
                metadata["bart"] = bart.provenance()
                metadata["runtime"] = versions()
            report = evaluate(
                args.sequence.resolve(), case, root, folder, feedback=True
            )
        except Exception as error:
            metadata.update(state="qualification_error", error=str(error))
            (args.output / "qualification.json").write_text(
                json.dumps(metadata, indent=2)
            )
            raise
        (folder / "result.json").write_text(json.dumps(report, indent=2))
        found = image_paths(report)
        if not found:
            metadata.update(
                state="control_rejected",
                failed_checks=[
                    c for c in report.get("checks", []) if c.get("passed") is False
                ],
            )
            (args.output / "qualification.json").write_text(
                json.dumps(metadata, indent=2)
            )
            return 1
        paths.append(found)
        metadata["stages"].append(
            {
                "name": name,
                "wall_s": time.monotonic() - started,
                "spatial_sampling": report.get("spatial_sampling"),
                "status": report["status"],
                "metrics": report.get("metrics"),
                "execution": report.get("grading_execution", report.get("execution")),
            }
        )
        print(f"Finished {name}", flush=True)
    if len(paths[0]) != len(paths[1]):
        raise ValueError("The two densities did not produce the same frame count")
    changes = []
    for base, fine in zip(*paths, strict=True):
        with np.load(base) as a, np.load(fine) as b:
            x, y = a["physical"], b["physical"]
            changes.append(float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30)))
    metadata.update(
        image_changes=changes,
        density_qualified=max(changes) <= args.maximum_change,
        state="qualified"
        if max(changes) <= args.maximum_change
        else "needs_more_spins",
    )
    (args.output / "qualification.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    return 0 if metadata["density_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
