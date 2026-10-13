"""Rescore SHA-verified S3 replays and declared image stress transforms.

Receiver rotation, blanking, displacement and ROI dropout are linear transforms
of retained replicates. Contrast erasure is a nonlinear transform of the scored
mean; its jackknife applies that transform independently to leave-one-out means.
These are derived controls, not fresh encoded Bloch submissions.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.s1 import resolve_decision, score_precision
from mriseqbench.evaluation.s1_case import (
    case_bounds,
    case_image_scores,
    image_policy,
    uses_case_image,
)
from mriseqbench.evaluation.s1_image import material_masks
from mriseqbench.evaluation.s1_phantom import phantom
from mriseqbench.evaluation.s1_sampling import image_scores, interval

ROOT = Path(__file__).resolve().parents[2]


def score(images, stored, case, policy, *, masks=None):
    physical = uses_case_image(case)
    if physical:
        if masks is None:
            masks = material_masks(phantom(case, ROOT, 4), case["task"]["sequence"])
            masks = {
                name: mask & stored["object_support"] for name, mask in masks.items()
            }
        masks = {name: mask & stored["object_support"] for name, mask in masks.items()}
        metrics, intervals, _ = case_image_scores(
            images,
            stored["reference"],
            stored["object_support"],
            stored["case_reference_phasor"],
            stored["case_reference_phase_valid"],
            masks,
        )
        bounds, prefix = (
            case_bounds(
                policy["case_image_criteria"].get(
                    image_policy(case), policy["case_image_criteria"].get("static")
                )
            ),
            "",
        )
    else:
        metrics, intervals = image_scores(
            images, stored["reference"], stored["object_support"]
        )
        bounds, prefix = policy["metrics"], "ideal_"
    precise, errors = score_precision(metrics, intervals, policy["image_precision"])
    passed, basis = resolve_decision(bounds, metrics, intervals, precise, prefix)
    if physical:
        energy = np.abs(stored["reference"]) ** 2
        coverage = float(
            energy[stored["case_reference_phase_valid"]].sum()
            / max(float(energy.sum()), 1e-30)
        )
        if coverage < 0.5:
            passed, basis = False, "phase_reference_coverage"
    return {
        "passed": passed,
        "basis": basis,
        "metrics": metrics,
        "intervals": intervals,
        "precision_errors": errors,
    }


def score_image_transform(images, stored, case, policy, transform, *, masks=None):
    """Jackknife a nonlinear transform of the scored mean image.

    T(mean(I)) and mean(T(I)) are different estimands. Transform the full and
    leave-one-out means separately; never reinterpret replicate phase noise as
    tissue contrast introduced by the intended contrast-erasure operation.
    """
    n = len(images)
    total = images.sum(axis=0)
    if masks is None and uses_case_image(case):
        masks = material_masks(phantom(case, ROOT, 4), case["task"]["sequence"])
        masks = {name: mask & stored["object_support"] for name, mask in masks.items()}

    def at(image):
        value = transform(image)
        return score(
            np.broadcast_to(value, (n, *value.shape)), stored, case, policy, masks=masks
        )

    full = at(total / n)
    leave_one_out = [at((total - image) / (n - 1)) for image in images]
    metrics, intervals = {}, {}
    derived = {
        "complex_image_nrmse": "normalized_squared_error",
        "magnitude_image_nrmse": "normalized_squared_magnitude_error",
        "phase_error_rad": "normalized_squared_phase_error",
    }
    for name, value in full["metrics"].items():
        if name in derived:
            continue
        loo = np.array([r["metrics"][name] for r in leave_one_out])
        estimate = float(n * value - (n - 1) * loo.mean())
        metrics[name] = estimate
        intervals[name] = list(interval(estimate, loo))
    for name, squared in derived.items():
        if squared in metrics:
            metrics[name] = float(np.sqrt(max(0.0, metrics[squared])))
            intervals[name] = [float(np.sqrt(max(0.0, x))) for x in intervals[squared]]
    physical = uses_case_image(case)
    bounds = (
        case_bounds(
            policy["case_image_criteria"].get(
                image_policy(case), policy["case_image_criteria"].get("static")
            )
        )
        if physical
        else policy["metrics"]
    )
    precise, errors = score_precision(metrics, intervals, policy["image_precision"])
    passed, basis = resolve_decision(
        bounds, metrics, intervals, precise, "" if physical else "ideal_"
    )
    if full["basis"] == "phase_reference_coverage":
        passed, basis = False, full["basis"]
    return {
        "passed": passed,
        "basis": basis,
        "metrics": metrics,
        "intervals": intervals,
        "precision_errors": errors,
        "uncertainty_method": "jackknife of transform applied to full and leave-one-out mean images",
    }


def erase_contrast(image, masks):
    result = image.copy()
    common = np.mean([np.abs(image[m]).mean() for m in masks.values()])
    for mask in masks.values():
        result[mask] = common * np.exp(1j * np.angle(image[mask]))
    means = [np.abs(result[m]).mean() for m in masks.values()]
    assert np.allclose(means, means[0], rtol=1e-12, atol=1e-14)
    return result


def run(args):
    policy = Catalog(ROOT).get("evaluators", "s3").model_dump(mode="json")
    rows = []
    for directory in args.directories:
        directory = directory.resolve()
        for case_path in sorted(directory.glob("*/case.json")):
            case = json.loads(case_path.read_text())
            result_path = case_path.parent / "evaluation/result.json"
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text())
            record_path = case_path.parent / "record.json"
            record = json.loads(record_path.read_text())
            assert (
                hashlib.sha256(result_path.read_bytes()).hexdigest()
                == record["result_sha256"]
            )
            assert (
                hashlib.sha256(case_path.read_bytes()).hexdigest()
                == record["case_sha256"]
            )
            assert (
                hashlib.sha256(
                    (case_path.parent / "sequence.seq").read_bytes()
                ).hexdigest()
                == record["sequence_sha256"]
            )
            if "images" not in result.get("artifacts", {}):
                rows.append(
                    {
                        "control": case_path.parent.name,
                        "source_result": str(result_path),
                        "status": result["status"],
                        "reason": result.get("reason"),
                        "failed_checks": [
                            c["name"]
                            for c in result.get("checks", [])
                            if not c["passed"]
                        ],
                    }
                )
                continue
            image_path = Path(result["artifacts"]["images"])
            if "artifact_sha256" in record:
                assert (
                    hashlib.sha256(image_path.read_bytes()).hexdigest()
                    == record["artifact_sha256"]["images"]
                )
            stored = np.load(image_path)
            physical = uses_case_image(case)
            images = stored["physical_replicates" if physical else "ideal_replicates"]
            native = score(images, stored, case, policy)
            phase_converged = (
                result.get("case_phase_reference") is None
                or result["case_phase_reference"]["converged"]
            )
            converged = (
                result["metrics"]["temporal_nrmse"] <= 0.01
                and phase_converged
                and native["passed"] is not None
            )
            corruptions = {
                "receiver_90deg": 1j * images,
                "blank": 0 * images,
                "displaced_image": np.roll(images, images.shape[1] // 3, axis=1),
            }
            if physical:
                masks = material_masks(phantom(case, ROOT, 4), case["task"]["sequence"])
                masks = {
                    name: mask & stored["object_support"]
                    for name, mask in masks.items()
                }
                eligible = {
                    k: m
                    for k, m in masks.items()
                    if m.any()
                    and np.abs(stored["reference"])[m].mean()
                    > 0.05 * np.abs(stored["reference"]).max()
                }
                if eligible:
                    mask = next(iter(eligible.values()))
                    dropout = images.copy()
                    dropout[:, mask] *= 0.1
                    corruptions["tissue_dropout"] = dropout
                if len(eligible) > 1:
                    corruptions["erased_contrast"] = score_image_transform(
                        images,
                        stored,
                        case,
                        policy,
                        lambda image, masks=eligible: erase_contrast(image, masks),
                    )
            rows.append(
                {
                    "control": case_path.parent.name,
                    "case_level": image_policy(case),
                    "source_status": result["status"],
                    "source_result": str(result_path),
                    "source_result_sha256": hashlib.sha256(
                        result_path.read_bytes()
                    ).hexdigest(),
                    "images_sha256": hashlib.sha256(
                        image_path.read_bytes()
                    ).hexdigest(),
                    "submission_sha256": hashlib.sha256(
                        (case_path.parent / "sequence.seq").read_bytes()
                    ).hexdigest(),
                    "numerically_converged": converged,
                    "native": native,
                    "corruptions": {
                        name: value
                        if name == "erased_contrast"
                        else score(value, stored, case, policy)
                        for name, value in corruptions.items()
                    },
                }
            )
    args.output.write_text(
        json.dumps(
            {
                "method": "retained-replay rescore and derived image corruptions; no fresh encoded simulation",
                "policy": policy,
                "contrast_erasure_method": "Erase contrast in the scored mean image, with the same transform applied independently to each leave-one-out mean for jackknife uncertainty.",
                "records": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "controls": len(rows),
                "resolved": sum(r.get("numerically_converged", False) for r in rows),
                "passes": sum(
                    r.get("numerically_converged", False)
                    and r["native"]["passed"] is True
                    for r in rows
                ),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
