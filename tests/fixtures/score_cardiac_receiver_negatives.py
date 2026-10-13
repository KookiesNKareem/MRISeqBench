"""Score exported receiver corruptions using exact acquisition-phase cache reuse.

Run with PYTHONPATH=<campaign>/source/src to use the frozen evaluator. Bloch
histories are reused only after verifying every RF/gradient/timing byte and ADC
parameter except receiver phase. Optional fresh native replay checks that identity.
"""

import argparse
import hashlib
import importlib
import json
import shutil
from pathlib import Path

import numpy as np
from score_cardiac_calibration import rotate_adc, score

import mriseqbench
from mriseqbench.evaluation import bart
from mriseqbench.evaluation.s1_image import expected_receiver_phase, material_masks
from mriseqbench.evaluation.s1_phantom import phantom
from mriseqbench.evaluation.s1_sampling import sampled_phantom


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def verify_export(source, negative):
    left, right = [
        Path(path).read_text().split("[SIGNATURE]")[0].splitlines()
        for path in (source, negative)
    ]
    assert len(left) == len(right)
    section, changes = None, 0
    for original, altered in zip(left, right, strict=True):
        if original.startswith("["):
            section = original
        if (
            section == "[ADC]"
            and original.strip()
            and not original.startswith(("#", "["))
        ):
            a, b = original.split(), altered.split()
            assert len(a) == len(b) == 9
            assert a[:7] == b[:7] and a[8:] == b[8:]
            assert abs(np.exp(-1j * (float(b[7]) - float(a[7]))) + 1j) < 1e-4
            changes += 1
        else:
            assert original == altered
    assert changes > 0
    return changes


def run(args):
    campaign, destination = args.campaign.resolve(), args.destination.resolve()
    frozen = campaign / "source"
    assert Path(mriseqbench.__file__).resolve().is_relative_to(frozen)
    selection = None
    if args.selection is not None:
        specification = json.loads(args.selection.read_text())
        assert Path(specification["campaign"]).resolve() == campaign
        assert specification["source_manifest_sha256"] == sha(
            campaign / "source-hashes.json"
        )
        selection = specification["positive_record_sha256"]
    destination.mkdir(parents=True, exist_ok=False)
    if args.selection is not None:
        shutil.copy2(args.selection, destination / "selection.json")
    shutil.copy2(__file__, destination / "source-scorer.py")
    shutil.copy2(
        Path(__file__).with_name("score_cardiac_calibration.py"),
        destination / "source-score-policy.py",
    )
    rows = []
    for path in sorted(campaign.glob("*/record.json")):
        native = json.loads(path.read_text())
        if selection is not None:
            if native["case_id"] not in selection:
                continue
            assert sha(path) == selection[native["case_id"]]
        if args.qualified_only and not (
            native.get("status") in {"uncalibrated", "passed"}
            and native.get("sequence_valid") is True
            and native.get("numerically_converged") is True
            and native.get("image_pass") is True
            and not native.get("failed_checks")
        ):
            continue
        directory = path.parent
        case = json.loads((directory / "case.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        assert sha(directory / "case.json") == native["case_sha256"]
        assert sha(directory / "sequence.seq") == native["sequence_sha256"]
        assert sha(directory / "result.json") == native["result_sha256"]
        if "images" not in result.get("artifacts", {}):
            continue
        identity = json.loads(Path(result["artifacts"]["provenance"]).read_text())[
            "reconstruction_provenance"
        ]
        assert bart.provenance()["binary_sha256"] == identity["binary_sha256"]
        image_path = Path(result["artifacts"]["images"])
        assert sha(image_path) == native["artifact_sha256"]["images"]
        with np.load(image_path) as data:
            stored = {key: data[key] for key in data.files}
        out = destination / case["case_id"]
        out.mkdir()
        exported = out / "receiver-rotation.seq"
        rotate_adc(directory / "sequence.seq", exported)
        changed = verify_export(directory / "sequence.seq", exported)
        module = importlib.import_module(
            f"mriseqbench.evaluation.{case['sequence_level'].lower()}"
        )
        inspector = importlib.import_module(
            f"mriseqbench.evaluation.{case['sequence_level'].lower()}_sequence"
        ).inspect_sequence
        positive, negative = [
            inspector(sequence, case)
            for sequence in (directory / "sequence.seq", exported)
        ]
        for name in (
            "adc",
            "k",
            "indices",
            "offset",
            "excitation",
            "te",
            "tr",
            "flips",
        ):
            np.testing.assert_array_equal(
                getattr(positive, name), getattr(negative, name)
            )
        np.testing.assert_array_equal(
            expected_receiver_phase(positive), expected_receiver_phase(negative)
        )
        factor = np.exp(-1j * (negative.adc_phase - positive.adc_phase))
        assert np.max(abs(factor + 1j)) < 1e-4
        # The reference/sampled reference retain the prescribed receiver phase.
        # Apply the mutation to native acquisition, then rebuild the full control
        # variate exactly as evaluate_s1; rotating saved images is not equivalent.
        actual = stored["koma_physical_raw_replicates"] * factor
        negative_raw = (
            stored["reference_raw"]
            + actual
            - stored["sampled_reference_raw_replicates"]
        )
        matrix = tuple(case["task"]["sequence"]["matrix"])
        images = np.asarray(
            [module.reconstruct(raw, negative, matrix) for raw in negative_raw]
        )
        masks = material_masks(phantom(case, frozen, 4), case["task"]["sequence"])
        if case["task"]["id"] == "reduced_fov":
            region = module.selection_region(negative, matrix)
            masks = {key: value & region for key, value in masks.items()}
        verdict = score(images, stored, case, masks)
        artifacts = out / "negative-images.npz"
        np.savez_compressed(
            artifacts,
            native_phase_mutated_raw=actual,
            control_variate_raw=negative_raw,
            physical_replicates=images,
        )
        row = {
            "case_id": case["case_id"],
            "scope": "exported ADC-phase corruption; exact native acquisition cache reuse, fresh BART reconstruction and case-image scoring",
            "positive_record_sha256": sha(path),
            "positive_image_pass": native["image_pass"],
            "sequence_sha256": sha(exported),
            "artifact_sha256": sha(artifacts),
            "changed_adc_events": changed,
            "receiver_factor_error": float(np.max(abs(factor + 1j))),
            "export_legal": all(check["passed"] for check in negative.checks),
            "image_verdict": verdict,
        }
        if case["case_id"] in args.fresh_native_case:
            provenance = json.loads(Path(result["artifacts"]["provenance"]).read_text())
            samples = provenance["spatial_samples"]
            progress = json.loads(
                (
                    Path(result["artifacts"]["sequence"]).parent / "progress.json"
                ).read_text()
            )
            step = progress["temporal_refinement"][-1]
            spins = sampled_phantom(case, frozen, samples, 0)
            simulator = (
                module.simulate_channels
                if case["sequence_level"] == "S3"
                else module.simulate
            )
            fresh = simulator(
                negative,
                spins,
                case["physics"]["effects"],
                step_s=step["step_s"],
                rf_step_s=step["rf_step_s"],
                integration_method="magnus4"
                if provenance["encoded_integrator"].endswith("BlochMagnus4")
                else "bloch",
            )
            error = float(np.linalg.norm(fresh - actual[0]) / np.linalg.norm(actual[0]))
            np.save(out / "fresh-native-negative-raw.npy", fresh)
            row["fresh_native_crosscheck"] = {
                "raw_relative_error": error,
                "passed": error < 1e-9,
                "integrator": provenance["encoded_integrator"],
                "sha256": sha(out / "fresh-native-negative-raw.npy"),
                "replicate": 0,
                "samples": samples,
            }
        write(out / "record.json", row)
        rows.append(row)
        print(
            json.dumps(
                {
                    "case_id": row["case_id"],
                    "rejected": verdict["passed"] is False,
                    "fresh_native": row.get("fresh_native_crosscheck"),
                }
            ),
            flush=True,
        )
    if selection is not None:
        assert {row["case_id"] for row in rows} == set(selection)
    write(
        destination / "results.json",
        {
            "records": rows,
            "rejected": sum(row["image_verdict"]["passed"] is False for row in rows),
            "accepted": sum(row["image_verdict"]["passed"] is True for row in rows),
            "unresolved": sum(row["image_verdict"]["passed"] is None for row in rows),
            "fresh_native_checks": sum(
                "fresh_native_crosscheck" in row for row in rows
            ),
            "fresh_native_failures": sum(
                not row["fresh_native_crosscheck"]["passed"]
                for row in rows
                if "fresh_native_crosscheck" in row
            ),
            "bart": bart.provenance(),
            "selection_sha256": sha(destination / "selection.json")
            if selection is not None
            else None,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--fresh-native-case", action="append", default=[])
    parser.add_argument(
        "--qualified-only",
        action="store_true",
        help="Score negatives only for frozen controls that qualify",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        help="Immutable case-ID/positive-record-hash selection manifest",
    )
    run(parser.parse_args())
