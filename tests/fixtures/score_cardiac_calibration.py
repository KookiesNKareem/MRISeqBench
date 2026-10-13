"""SHA-verified cardiac stress controls; receiver rotations are derived linear stress controls."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from mriseqbench.evaluation.s1 import resolve_decision, score_precision
from mriseqbench.evaluation.s1_case import case_bounds, case_image_scores
from mriseqbench.evaluation.s1_image import material_masks
from mriseqbench.evaluation.s1_phantom import phantom


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotate_adc(source, destination):
    """Change only Pulseq-1.5 ADC receiver phase; leave RF/gradient/timing bytes intact."""
    lines = Path(source).read_text().split("[SIGNATURE]")[0].splitlines()
    section = None
    for index, line in enumerate(lines):
        if line.startswith("["):
            section = line
        elif section == "[ADC]" and line.strip() and not line.startswith("#"):
            fields = line.split()
            assert len(fields) == 9, "Expected Pulseq 1.5 ADC columns"
            fields[7] = format(float(fields[7]) % (2 * np.pi) + np.pi / 2, ".16g")
            lines[index] = " ".join(fields)
    Path(destination).write_text("\n".join(lines) + "\n")


def score(images, stored, case, masks):
    metrics, intervals, _ = case_image_scores(
        images,
        stored["reference"],
        stored["object_support"],
        stored["case_reference_phasor"],
        stored["case_reference_phase_valid"],
        masks,
    )
    policy = case["evaluation"]
    precise, errors = score_precision(metrics, intervals, policy["image_precision"])
    passed, basis = resolve_decision(
        case_bounds(policy["case_image_criteria"]["static"]),
        metrics,
        intervals,
        precise,
    )
    return {
        "passed": passed,
        "basis": basis,
        "metrics": metrics,
        "intervals": intervals,
        "precision_errors": errors,
    }


def run(args):
    records = []
    for campaign in args.campaign:
        frozen = campaign / "source"
        for record_path in sorted(campaign.glob("*/record.json")):
            native = json.loads(record_path.read_text())
            if not native["image_pass"]:
                continue
            directory = record_path.parent
            case = json.loads((directory / "case.json").read_text())
            assert sha(directory / "case.json") == native["case_sha256"]
            assert sha(directory / "sequence.seq") == native["sequence_sha256"]
            assert sha(directory / "result.json") == native["result_sha256"]
            result = json.loads((directory / "result.json").read_text())
            image_path = Path(result["artifacts"]["images"])
            assert sha(image_path) == native["artifact_sha256"]["images"]
            with np.load(image_path) as data:
                stored = {key: data[key] for key in data.files}
            images = stored["physical_replicates"]
            masks = material_masks(phantom(case, frozen, 4), case["task"]["sequence"])
            original = score(images, stored, case, masks)
            assert original["passed"] is True
            negative_path = directory / "receiver-rotation-negative-v2.seq"
            assert not negative_path.exists(), "Never overwrite a frozen negative"
            rotate_adc(directory / "sequence.seq", negative_path)
            inspect = importlib.import_module(
                f"mriseqbench.evaluation.{case['sequence_level'].lower()}_sequence"
            ).inspect_sequence
            positive, negative = (
                inspect(directory / "sequence.seq", case),
                inspect(negative_path, case),
            )
            assert np.array_equal(positive.adc, negative.adc)
            assert np.array_equal(positive.k, negative.k)
            factor = np.exp(-1j * (negative.adc_phase - positive.adc_phase))
            assert np.max(abs(factor + 1j)) < 1e-4
            # The exported receiver change is audited separately (Pulseq parsing
            # rounds phases). These exact 90-degree image rotations are derived
            # stress controls, not fresh negative encoded Bloch submissions.
            transforms = {
                "derived_receiver_rotation": -1j * images,
                "blank": np.zeros_like(images),
                "half_fov_ghost": np.roll(images, images.shape[1] // 2, axis=1),
            }
            for tissue, mask in masks.items():
                if mask.any():
                    transforms[f"dropout_{tissue}"] = images * np.where(mask, 0.05, 1)
            corruptions = {
                name: score(value, stored, case, masks)
                for name, value in transforms.items()
            }
            records.append(
                {
                    "case_id": case["case_id"],
                    "record_sha256": sha(record_path),
                    "original": original,
                    "corruptions": corruptions,
                    "negative_sha256": sha(negative_path),
                    "receiver_factor_error": float(np.max(abs(factor + 1j))),
                    "scope": "derived image stress controls; separately exported receiver phase audited within parser rounding tolerance, not scored as fresh native replay",
                }
            )
    output = {
        "records": records,
        "rejected": sum(
            row["passed"] is False for r in records for row in r["corruptions"].values()
        ),
        "accepted": sum(
            row["passed"] is True for r in records for row in r["corruptions"].values()
        ),
        "unresolved": sum(
            row["passed"] is None for r in records for row in r["corruptions"].values()
        ),
    }
    assert not args.out.exists()
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps({key: output[key] for key in ("rejected", "accepted", "unresolved")})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args())
