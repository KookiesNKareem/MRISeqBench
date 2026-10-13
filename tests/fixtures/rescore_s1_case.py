"""Apply s1@2 to verified stored KomaMRI/MRIReco replicates from a frozen pilot.

Runs new KomaMRI RF phase-reference simulations. It does not claim to rerun the
encoded acquisition. Old reports and submission bytes are never modified.
"""

import argparse
import hashlib
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mriseqbench.catalog import Catalog
from mriseqbench.evaluation.s1 import (
    IMPLEMENTATION_SHA256,
    resolve_decision,
    score_precision,
)
from mriseqbench.evaluation.s1_case import (
    case_bounds,
    case_image_scores,
    phase_reference,
)
from mriseqbench.evaluation.s1_image import material_masks
from mriseqbench.evaluation.s1_koma import versions
from mriseqbench.evaluation.s1_phantom import phantom
from mriseqbench.evaluation.s1_sequence import inspect_sequence
from mriseqbench.harness.grading import grade_frozen
from mriseqbench.io import write_json
from mriseqbench.process import execute

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retry_unresolved(row, directory):
    """Resume a fresh adaptive encoded replay when saved replicates are insufficient."""
    if row.get("status") in {"passed", "failed"}:
        return row
    case = json.loads((directory / "case.json").read_text())
    out = directory / "adaptive-replay"
    sequence = Path(row["source_result"]).parent / "sequence.seq"
    fresh = grade_frozen(sequence, case, ROOT, out, None, execute)
    if "provenance" in fresh.get("artifacts", {}):
        provenance = json.loads(Path(fresh["artifacts"]["provenance"]).read_text())
        assert (
            provenance["submission_sha256"] == row["submission_sha256"] == sha(sequence)
        )
    write_json(directory / "initial-result.json", row)
    updated = {
        **row,
        "status": fresh["status"],
        "image_verdict": fresh.get("image_verdict", "not_evaluated"),
        "metrics": {
            k.removeprefix("case_"): v
            for k, v in fresh.get("metrics", {}).items()
            if k.startswith("case_")
        },
        "numerical_intervals": fresh.get("numerical_intervals", {}).get("case", {}),
        "numerically_converged": fresh.get("numerically_converged", False),
        "sequence_valid": fresh.get("sequence_valid"),
        "ideal_imaging_success": fresh.get("ideal_imaging_success"),
        "imaging_success_under_case_physics": fresh.get(
            "imaging_success_under_case_physics"
        ),
        "case_phase_reference": fresh.get("case_phase_reference"),
        "case_image_checks": fresh.get("case_image_checks", []),
        "score_precision_met": fresh.get("score_precision_met", {}).get("case"),
        "decision_basis": fresh.get("decision_basis", {}).get("case"),
        "numerical_precision_errors": fresh.get("numerical_precision", {}).get(
            "case_errors", {}
        ),
        "encoded_temporal_nrmse": fresh.get("metrics", {}).get("temporal_nrmse"),
        "adaptive_replay_result": str(out / "result.json"),
        "final_evaluation_mode": "fresh full encoded KomaMRI/MRIReco replay to numerical convergence",
        "implementation_sha256": provenance["implementation_sha256"]
        if "provenance" in fresh.get("artifacts", {})
        else row["implementation_sha256"],
    }
    updated.pop("error_kind", None)
    if "error_kind" in fresh:
        updated["error_kind"] = fresh["error_kind"]
    write_json(directory / "result.json", updated)
    return updated


def run(pilot, out):
    out.mkdir(parents=True, exist_ok=False)
    policy = Catalog(ROOT).get("evaluators", "s1").model_dump(mode="json")
    write_json(out / "frozen-policy.json", policy)
    verification = json.loads((pilot / "verification.json").read_text())
    manifest = json.loads((pilot / "source-manifest.json").read_text())
    archive = pilot / "source-snapshot.zip"
    assert sha(archive) == verification["source_archive_sha256"]
    with zipfile.ZipFile(archive) as files:
        for name, digest in manifest.items():
            assert hashlib.sha256(files.read(name)).hexdigest() == digest, name
    for name in ("Manifest.toml", "benchmark/objects/shapes.json"):
        assert sha(ROOT / name) == manifest[name], name
    current_versions = versions()

    jobs = []
    for job in sorted(pilot.glob("p[23]_*")):
        attempts = sorted((job / "control/submissions").glob("attempt-*"))
        jobs.append((job.name, attempts[-1], True))
        if job.name == "p3_spin_echo":
            jobs.append((job.name + "--preceding-attempt-002", attempts[-2], False))

    def once(job):
        name, attempt, final = job
        directory = out / name
        directory.mkdir()
        old = json.loads((attempt / "result.json").read_text())
        case = json.loads((attempt / "case.json").read_text())
        original_case_sha256 = hashlib.sha256(
            json.dumps(case, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        receipt = json.loads((attempt / "receipt.json").read_text())
        sequence = attempt / "sequence.seq"
        digest = sha(sequence)
        assert digest == receipt["sha256"]
        row = {
            "job": name,
            "final_submission": final,
            "attempt": receipt["attempt"],
            "submission_sha256": digest,
            "original_status": old["status"],
            "source_result": str(attempt / "result.json"),
            "source_result_sha256": sha(attempt / "result.json"),
            "evaluation_contract": "s1@2",
            "image_acceptance_basis": "case_physics",
            "implementation_sha256": IMPLEMENTATION_SHA256,
        }
        case["evaluation"] = policy
        case["task"]["evaluation"] = "s1@2"
        write_json(directory / "case.json", case)
        artifacts = old.get("artifacts", {})
        if "images" not in artifacts:
            # Keep preflight/sequence failures as failures, with no image claim.
            assert old["status"] == "failed"
            row.update(
                status="failed", image_verdict="not_evaluated", checks=old["checks"]
            )
            write_json(directory / "result.json", row)
            return row
        provenance = json.loads(Path(artifacts["provenance"]).read_text())
        assert provenance["submission_sha256"] == digest
        assert provenance["case_sha256"] == original_case_sha256
        assert (
            provenance["implementation_sha256"]
            == verification["judge_implementation_sha256"]
        )
        assert provenance["koma_versions"] == current_versions
        assert provenance["shapes_sha256"] == manifest["benchmark/objects/shapes.json"]
        assert provenance["julia_manifest_sha256"] == manifest["Manifest.toml"]
        verified = next(
            r
            for r in verification["receipts"]
            if r["job_id"] == name.split("--")[0] and r["attempt"] == receipt["attempt"]
        )
        assert any(
            r["images_sha256"] == sha(artifacts["images"])
            for r in verified["verified_replays"]
        )
        data = np.load(artifacts["images"])
        reference = data["reference"]
        acq = inspect_sequence(sequence, case)
        spins = phantom(case, ROOT, 4)
        masks = material_masks(spins, case["task"]["sequence"])
        matrix = reference.shape
        fov = np.asarray(case["task"]["sequence"]["fov_mm"]) / 1000
        loc = np.floor((spins.positions[:, : len(matrix)] / fov + 0.5) * matrix).astype(
            int
        )
        support = np.zeros(matrix, bool)
        support[tuple(loc.T)] = True
        phasor, valid, info = phase_reference(acq, case, ROOT, reference)
        metrics, intervals, regions = case_image_scores(
            data["physical_replicates"], reference, support, phasor, valid, masks
        )
        precise, errors = score_precision(metrics, intervals, policy["image_precision"])
        bounds = case_bounds(policy["case_image_criteria"][case["object_level"]])
        passed, basis = resolve_decision(bounds, metrics, intervals, precise)
        converged = (
            passed is not None
            and info["converged"]
            and old["metrics"]["temporal_nrmse"] <= 0.01
        )
        row.update(
            status=("passed" if passed and old["sequence_valid"] else "failed")
            if converged
            else None,
            image_verdict=("passed" if passed else "failed")
            if converged
            else "not_evaluated",
            metrics=metrics,
            numerical_intervals=intervals,
            score_precision_met=precise,
            numerical_precision_errors=errors,
            decision_basis=basis,
            numerically_converged=converged,
            imaging_success_under_case_physics=passed if converged else None,
            ideal_imaging_success=old.get("ideal_imaging_success"),
            sequence_valid=old["sequence_valid"],
            encoded_temporal_nrmse=old["metrics"]["temporal_nrmse"],
            case_phase_reference=info,
            case_scoring_regions=regions,
            case_image_checks=[
                {
                    **bound,
                    "value": metrics[bound["name"]],
                    "passed": resolve_decision([bound], metrics, intervals, precise)[0]
                    if converged
                    else None,
                }
                for bound in bounds
            ],
            provenance={
                "mode": "stored encoded KomaMRI/MRIReco replicates plus fresh KomaMRI phase reference",
                "encoded_images": artifacts["images"],
                "encoded_images_sha256": sha(artifacts["images"]),
                "original_provenance_sha256": sha(artifacts["provenance"]),
                "koma_versions": current_versions,
                "original_implementation_sha256": provenance["implementation_sha256"],
            },
        )
        np.savez_compressed(
            directory / "phase-reference.npz",
            reference=reference,
            case_reference=reference * phasor,
            phasor=phasor,
            phase_valid=valid,
            object_support=support,
        )
        if not converged:
            write_json(directory / "offline-diagnostics.json", row)
            return retry_unresolved(row, directory)
        write_json(directory / "result.json", row)
        print(
            json.dumps({"job": name, "status": row["status"], "metrics": metrics}),
            flush=True,
        )
        return row

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(once, jobs))
    rows = [retry_unresolved(row, out / row["job"]) for row in rows]
    write_json(
        out / "summary.json",
        {
            "records": rows,
            "source_files_verified": len(manifest),
            "original_archive_sha256": sha(archive),
            "final_passes": sum(
                r["status"] == "passed" for r in rows if r["final_submission"]
            ),
            "final_count": sum(r["final_submission"] for r in rows),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    run(args.pilot.resolve(), args.destination.resolve())
