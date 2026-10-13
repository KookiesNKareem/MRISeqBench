"""Counterfactual mapping assay: encoded relaxation changes, judge truth stays fixed.

This is a diagnostic of the map-error gate, not official sequence qualification.
Only the encoded native Bloch replay sees the +30% myocardial relaxation time.
The RF target, phase oracle, dictionary, anatomy and acceptance limits stay fixed.
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation import evaluate
from mriseqbench.evaluation.s1_koma import simulate
from tests.fixtures.qualify_s4 import freeze, sha
from tests.fixtures.s4_sequences import make_sequence


def run(output):
    for name, digest in json.loads(
        (output / "source_manifest.json").read_text()
    ).items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen diagnostic source changed: {name}")
    s4 = importlib.import_module("mriseqbench.evaluation.s4")
    original = s4.evaluate_s1
    catalog = Catalog(ROOT)
    receipts = []
    for family, axis in (("t1_mapping", 1), ("t2_mapping", 2)):
        case = catalog.resolve(catalog.get("tasks", family), hardware_ref="low_field")
        case["evaluation"].update(
            spatial_samples=4096, spatial_replicates=4, spatial_seed=20261019
        )
        directory = output / family
        directory.mkdir()
        case_path = directory / "case.json"
        case_path.write_text(json.dumps(case, indent=2) + "\n")
        sequence_path = directory / "sequence.seq"
        make_sequence(case).write(str(sequence_path))
        calls = []

        def biased_encoded(
            acquisition,
            spins,
            *args,
            axis=axis,
            calls=calls,
            simulator=simulate,
            **kwargs,
        ):
            properties = spins.properties.copy()
            myocardial = spins.materials.index("myocardium")
            baseline = float(properties[myocardial, axis])
            properties[myocardial, axis] *= 1.3
            calls.append({"judge_seconds": baseline, "encoded_seconds": baseline * 1.3})
            return simulator(
                acquisition, replace(spins, properties=properties), *args, **kwargs
            )

        def frame_evaluator(*args, bias=biased_encoded, **kwargs):
            # Preserve the production all-frame cache, if present. Changed
            # properties enter its identity before any native replay is reused.
            simulator = kwargs.pop("simulator", simulate)

            def encoded(*values, **steps):
                return bias(*values, simulator=simulator, **steps)

            return original(*args, simulator=encoded, **kwargs)

        s4.evaluate_s1 = frame_evaluator
        try:
            result = evaluate(sequence_path, case, ROOT, directory / "evaluation")
        except Exception as error:  # noqa: BLE001 -- Preserve failed diagnostics.
            result = {
                "status": "evaluator_error",
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        finally:
            s4.evaluate_s1 = original
        result_path = directory / "result.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        name = f"myocardial_t{axis}_relative_error"
        mapping = next(
            (
                item
                for item in result.get("application_checks", [])
                if item["name"] == name
            ),
            None,
        )
        receipt = {
            "task": family,
            "kind": "counterfactual_encoded_relaxation_diagnostic",
            "official_sequence_qualification": False,
            "encoded_relaxation_factor": 1.3,
            "judge_truth_modified": False,
            "encoded_simulator_requests": len(calls),
            "encoded_properties": calls[0] if calls else None,
            "case_sha256": sha(case_path),
            "sequence_sha256": sha(sequence_path),
            "result_sha256": sha(result_path),
            "status": result["status"],
            "mapping_check": mapping,
            "rejected_by_mapping_gate": mapping is not None
            and mapping["passed"] is False,
            "artifacts": {
                str(path.relative_to(output)): sha(path)
                for path in sorted(directory.rglob("*"))
                if path.is_file() and path.name in {"images.npz", "provenance.json"}
            },
        }
        receipts.append(receipt)
        (output / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.frozen_run:
        run(output)
        return
    source = freeze(output)
    name = "tests/fixtures/qualify_s4_mapping_bias.py"
    shutil.copy2(ROOT / name, source / name)
    manifest_path = output / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[name] = sha(source / name)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source / "src") + os.pathsep + str(source)
    environment["MRISEQBENCH_BART"] = json.loads(
        (source / "reconstruction_tool.json").read_text()
    )["executable"]
    with (output / "diagnostic.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(source / name),
                "--output",
                str(output),
                "--frozen-run",
            ],
            cwd=source,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
