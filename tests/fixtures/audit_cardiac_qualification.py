"""Audit merged, immutable cardiac qualification evidence against the live suite.

Campaign order gives explicit precedence to later successful repaired controls.
Original failures remain in the certificate. This never rewrites native results,
marks a provisional result passed, or changes evaluator calibration flags.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from mriseqbench.catalog import Catalog
from tests.fixtures.audit_s4_qualification import expected_negatives

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Changed evidence: {path}")
    return str(Path(path).resolve())


def contained_file(root, relative):
    """A sealed relative path cannot traverse or alias its evidence directory."""
    root, relative = Path(root), Path(relative)
    if root.is_symlink() or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Evidence path escapes or aliases its root: {relative}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Evidence path escapes its root: {relative}")
    parent = root
    for part in relative.parts:
        parent = parent / part
        if parent.is_symlink():
            raise ValueError(f"Symlinked evidence path: {relative}")
    if not path.is_file():
        raise ValueError(f"Missing sealed evidence: {path}")
    return path


def sealed_native_artifacts(directory, row, report):
    paths, hashes = report.get("artifacts"), row.get("artifact_sha256")
    if (
        not isinstance(paths, dict)
        or not isinstance(hashes, dict)
        or not {"images", "provenance", "sequence"} <= paths.keys()
        or set(paths) != set(hashes)
    ):
        raise ValueError(f"Missing native artifact hash coverage: {directory}")
    for key, artifact in paths.items():
        path = Path(artifact)
        if not path.is_absolute():
            path = directory / path
        try:
            relative = path.relative_to(directory)
        except ValueError as exc:
            raise ValueError(f"Unrelated native artifact: {path}") from exc
        verified(contained_file(directory, relative), hashes[key])


def explicit_failure(report):
    return any(
        item.get("passed") is False
        for group in ("checks", "image_checks", "application_checks")
        for item in report.get(group, [])
    ) or any(explicit_failure(frame) for frame in report.get("frames", []))


def qualified(report, *, frames_required=False):
    checks = report.get("checks", []) + report.get("image_checks", [])
    if "frames" in report or frames_required:
        frames = report.get("frames")
        if not frames or not all(qualified(frame) for frame in frames):
            return False
        checks += report.get("application_checks", [])
        if not report.get("application_checks"):
            return False
    return bool(
        report.get("status") in {"uncalibrated", "passed"}
        and report.get("sequence_valid") is True
        and report.get("numerically_converged") is True
        and report.get("checks")
        and report.get("image_checks")
        and all(item["passed"] is True for item in checks)
    )


def sealed_s4_artifacts(
    campaign, directory, row, report, case, *, complete_frames=True
):
    sealed_path = campaign / "evidence_manifest.json"
    sealed = json.loads(sealed_path.read_text())
    record = sealed["cases"][row["name"]]
    for filename, name, key in (
        ("case.json", "case", "case_sha256"),
        ("sequence.seq", "sequence", "sequence_sha256"),
        ("result.json", "result", "result_sha256"),
    ):
        entry = record[name]
        if (campaign / entry["path"]).resolve() != (
            directory / filename
        ).resolve() or entry["sha256"] != row[key]:
            raise ValueError(f"Unrelated sealed S4 {name}: {directory}")
        verified(campaign / entry["path"], entry["sha256"])
    expected = 3 if case["task"]["id"] in {"t1_mapping", "t2_mapping"} else 1
    frames, receipts = report.get("frames", []), record["frames"]
    if len(receipts) != len(frames) or (complete_frames and len(frames) != expected):
        raise ValueError(f"Missing S4 frame evidence: {directory}")
    for index, (frame, receipt) in enumerate(zip(frames, receipts, strict=True)):
        paths = frame.get("artifacts", {})
        if (
            receipt["index"] != index
            or set(paths) != set(receipt["artifacts"])
            or not (
                {"images", "provenance", "sequence"}
                if complete_frames
                else {"sequence"}
            )
            <= paths.keys()
        ):
            raise ValueError(f"Missing S4 frame artifact: {directory}/{index}")
        for key, entry in receipt["artifacts"].items():
            path = campaign / entry["path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(directory.resolve())
                or path.resolve() != Path(paths[key]).resolve()
            ):
                raise ValueError(
                    f"Unrelated sealed S4 frame artifact: {directory}/{index}/{key}"
                )
            verified(path, entry["sha256"])
    return sha(sealed_path)


def selected_s4_negatives(candidates, positives):
    selected = {}
    for key, candidate in candidates:
        positive = positives.get(candidate["case_id"])
        if (
            positive is not None
            and candidate["base_sequence_sha256"] == positive["sequence_sha256"]
        ):
            selected[key] = candidate
    return selected


def verify_receiver_export(positive, negative):
    left, right = [
        Path(path).read_text().split("[SIGNATURE]")[0].splitlines()
        for path in (positive, negative)
    ]
    if len(left) != len(right):
        raise ValueError("Receiver negative changes sequence structure")
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
            if len(a) != 9 or len(b) != 9 or a[:7] != b[:7] or a[8:] != b[8:]:
                raise ValueError("Receiver negative changes ADC acquisition parameters")
            delta = float(b[7]) - float(a[7])
            if (
                not math.isfinite(delta)
                or abs(complex(math.cos(delta), -math.sin(delta)) + 1j) >= 1e-4
            ):
                raise ValueError("Receiver negative has incorrect ADC phase")
            changes += 1
        elif original != altered:
            raise ValueError("Receiver negative changes RF, gradient or timing bytes")
    if not changes:
        raise ValueError("Receiver negative has no ADC phase mutation")
    return changes


def verify_receiver_negative(directory, row, positive):
    verified(directory / "receiver-rotation.seq", row["sequence_sha256"])
    verified(directory / "negative-images.npz", row["artifact_sha256"])
    verify_receiver_export(positive["sequence"], directory / "receiver-rotation.seq")
    crosscheck = row.get("fresh_native_crosscheck")
    if crosscheck is not None:
        error = crosscheck.get("raw_relative_error")
        if (
            crosscheck.get("passed") is not True
            or not isinstance(error, (float, int))
            or not math.isfinite(error)
            or error >= 1e-9
        ):
            raise ValueError(f"Failed fresh native receiver crosscheck: {directory}")
        verified(directory / "fresh-native-negative-raw.npy", crosscheck["sha256"])


def contract(case, current, *, report=None, mutation=None):
    exceptions = []
    for key in (
        "case_id",
        "sequence_level",
        "image_policy",
        "task",
        "object",
        "physics",
        "hardware",
    ):
        if case[key] != current[key]:
            old_task = json.loads(json.dumps(case["task"]))
            capabilities = old_task["sequence"]["capabilities"]
            if "spoiled_gre" in capabilities:
                capabilities[capabilities.index("spoiled_gre")] = "spin_echo"
            failed_checks = (
                []
                if report is None
                else (report.get("checks", []) + report.get("image_checks", []))
            )
            historical_failure = (
                key == "task"
                and case["task"]["id"] == "reduced_fov"
                and old_task == current["task"]
                and mutation is None
                and report is not None
                and report.get("status") in {"failed", "uncalibrated"}
                and not qualified(report)
                and any(item.get("passed") is False for item in failed_checks)
            )
            if not historical_failure:
                raise ValueError(f"Current case differs: {case['case_id']}/{key}")
            exceptions.append("failed_reduced_fov_capability_metadata_only")
    # Independent seeds are intentional. Calibration is an evidence-dependent
    # publication flag; it cannot determine whether a scientific control passes.
    ignored = {"spatial_seed", "calibrated"}
    for key, value in current["evaluation"].items():
        if key in ignored:
            continue
        if key == "mapping_relative_error_precision" and key not in case["evaluation"]:
            # Verify the new numerical rule from retained estimate CIs below;
            # no acceptance threshold or native verdict is rewritten.
            continue
        if case["evaluation"].get(key) != value:
            raise ValueError(f"Current evaluation differs: {case['case_id']}/{key}")
    return exceptions


REQUIRED_SCIENTIFIC_SOURCES = {
    "Project.toml",
    "Manifest.toml",
    "src/mriseqbench/evaluation/bart.py",
    "src/mriseqbench/evaluation/s1.py",
    "src/mriseqbench/evaluation/s1_koma.py",
    "src/mriseqbench/evaluation/s1_koma.jl",
    "src/mriseqbench/evaluation/s1_anatomy.py",
    "benchmark/objects/anatomy.json",
    "benchmark/objects/phantoms.jl",
    "benchmark/physics/effects.jl",
}


def source_manifest(campaign):
    name = (
        "source-hashes.json"
        if (campaign / "source-hashes.json").exists()
        else "source_manifest.json"
    )
    path = campaign / name
    declared = json.loads(path.read_text())
    source = campaign / "source"
    actual = {
        str(file.relative_to(source))
        for file in source.rglob("*")
        if file.is_file() and "__pycache__" not in file.parts
    }
    if (
        not isinstance(declared, dict)
        or not REQUIRED_SCIENTIFIC_SOURCES <= declared.keys()
        or set(declared) != actual
    ):
        raise ValueError(f"Incomplete scientific source manifest: {campaign}")
    for relative, digest in declared.items():
        verified(contained_file(source, relative), digest)
    return {"path": str(path), "sha256": sha(path)}


def audit(campaigns, receiver_campaigns, root=ROOT, complete=False):
    catalog = Catalog(root)
    current = {
        case["case_id"]: case for case in catalog.cases(catalog.get("suites", "full"))
    }
    positives, attempts, negative_candidates, sources = {}, [], [], []
    binary = None
    for campaign in map(Path, campaigns):
        campaign = campaign.resolve()
        source = source_manifest(campaign)
        sources.append(source)
        is_s4 = (campaign / "source_manifest.json").exists()
        if is_s4:
            path = campaign / "results.json"
            rows = json.loads(path.read_text()) if path.exists() else []
            entries = [(campaign / "cases" / row["name"], row) for row in rows]
        else:
            entries = [
                (path.parent, json.loads(path.read_text()))
                for path in sorted(campaign.glob("*/record.json"))
            ]
        base_controls = (
            {row["name"]: row for _, row in entries if row["mutation"] is None}
            if is_s4
            else {}
        )
        for directory, row in entries:
            for filename, key in (
                ("case.json", "case_sha256"),
                ("sequence.seq", "sequence_sha256"),
                ("result.json", "result_sha256"),
            ):
                verified(directory / filename, row[key])
            case = json.loads((directory / "case.json").read_text())
            identity = case["case_id"]
            if identity not in current:
                raise ValueError(f"Unexpected qualification case: {identity}")
            report = json.loads((directory / "result.json").read_text())
            if report["status"] != row["status"]:
                raise ValueError(f"Changed verdict: {directory}")
            exceptions = contract(
                case, current[identity], report=report, mutation=row["mutation"]
            )
            if row["mutation"] is not None:
                if is_s4:
                    key = f"{case['task']['id']}--{case['hardware']['id']}--{row['mutation']}"
                    if key not in expected_negatives():
                        raise ValueError(f"Unexpected S4 negative: {key}")
                    if report["status"] == "failed" and not qualified(
                        report, frames_required=True
                    ):
                        if not explicit_failure(report):
                            raise ValueError(
                                f"S4 negative has no explicit failed check: {directory}"
                            )
                        seal = sealed_s4_artifacts(
                            campaign,
                            directory,
                            row,
                            report,
                            case,
                            complete_frames=False,
                        )
                        base = base_controls.get(
                            f"{case['task']['id']}--{case['hardware']['id']}--positive"
                        )
                        if base is None:
                            raise ValueError(
                                f"Missing base S4 positive for negative: {directory}"
                            )
                        negative_candidates.append(
                            (
                                key,
                                {
                                    "case_id": identity,
                                    "base_sequence_sha256": base["sequence_sha256"],
                                    "source_manifest_sha256": source["sha256"],
                                    "evidence_manifest_sha256": seal,
                                    "result": str(directory / "result.json"),
                                    "sha256": row["result_sha256"],
                                },
                            )
                        )
                continue
            passed = qualified(report, frames_required=case["sequence_level"] == "S4")
            if passed and case["task"]["id"] in {"t1_mapping", "t2_mapping"}:
                axis = "t1" if case["task"]["id"] == "t1_mapping" else "t2"
                mapping = next(
                    item
                    for item in report["application_checks"]
                    if item["name"] == f"myocardial_{axis}_relative_error"
                )["detail"]
                lo, hi = mapping["numerical_interval"]
                point = mapping["relative_error"]
                policy = current[identity]["evaluation"]
                precise = (
                    max(abs(point - lo), abs(hi - point))
                    <= policy["mapping_relative_error_precision"]
                )
                if (
                    not all(math.isfinite(value) for value in (lo, hi, point))
                    or point < 0
                    or not lo <= point <= hi
                ):
                    raise ValueError(f"Invalid mapping numerical interval: {directory}")
                if not (
                    (precise and point <= policy["mapping_relative_error_max"])
                    or hi <= policy["mapping_relative_error_max"]
                ):
                    raise ValueError(
                        f"Current mapping numerical decision unresolved: {directory}"
                    )
            receipt = {
                "case_id": identity,
                "qualified_control": passed,
                "status": report["status"],
                "case": str(directory / "case.json"),
                "case_sha256": row["case_sha256"],
                "sequence": str(directory / "sequence.seq"),
                "sequence_sha256": row["sequence_sha256"],
                "result": str(directory / "result.json"),
                "result_sha256": row["result_sha256"],
                "record_sha256": sha(directory / "record.json") if not is_s4 else None,
                "source_manifest_sha256": source["sha256"],
                "contract_exceptions": exceptions,
            }
            attempts.append(receipt)
            if not passed:
                continue
            reports = [report, *report.get("frames", [])]
            for item in reports:
                provenance = item.get("reconstruction_provenance", {})
                if (
                    provenance.get("engine") != "BART"
                    or not provenance.get("commands")
                    or not provenance.get("version")
                ):
                    raise ValueError(f"Missing BART execution evidence: {directory}")
                digest = provenance.get("binary_sha256")
                if not isinstance(digest, str) or len(digest) != 64:
                    raise ValueError(f"Missing BART binary identity: {directory}")
                if binary is not None and binary != digest:
                    raise ValueError("Qualification mixes BART binaries")
                binary = digest
                for path in item.get("artifacts", {}).values():
                    if not Path(path).is_file():
                        raise ValueError(f"Missing native evidence: {path}")
            if not is_s4:
                sealed_native_artifacts(directory, row, report)
            else:
                receipt["evidence_manifest_sha256"] = sealed_s4_artifacts(
                    campaign, directory, row, report, case
                )
            positives[identity] = receipt
    negative_s4 = selected_s4_negatives(negative_candidates, positives)
    receiver = {}
    for campaign in map(Path, receiver_campaigns):
        summary = campaign / "results.json"
        if complete and not summary.is_file():
            raise ValueError(f"Receiver campaign summary is not sealed: {campaign}")
        if summary.exists():
            identity = json.loads(summary.read_text()).get("bart", {})
            if identity.get("engine") != "BART" or (
                binary is not None and identity.get("binary_sha256") != binary
            ):
                raise ValueError(f"Receiver negatives mix BART binaries: {campaign}")
        for path in sorted(campaign.glob("*/record.json")):
            row = json.loads(path.read_text())
            identity = row["case_id"]
            if identity not in positives:
                continue
            if row["positive_record_sha256"] != positives[identity]["record_sha256"]:
                continue  # Negative for a superseded positive is retained separately.
            verify_receiver_negative(path.parent, row, positives[identity])
            if row["export_legal"] is True and row["image_verdict"]["passed"] is False:
                receiver[identity] = {
                    "record": str(path.resolve()),
                    "sha256": sha(path),
                }
    required_receiver = {
        identity for identity, case in current.items() if case["sequence_level"] != "S4"
    }
    missing = sorted(set(current) - positives.keys())
    missing_receiver = sorted(required_receiver - receiver.keys())
    missing_s4 = sorted(expected_negatives() - negative_s4.keys())
    done = not (missing or missing_receiver or missing_s4)
    result = {
        "complete": done,
        "expected_positive_count": len(current),
        "qualified_positive_count": len(positives),
        "missing_positive_cases": missing,
        "missing_receiver_negative_cases": missing_receiver,
        "missing_s4_negative_cases": missing_s4,
        "bart_binary_sha256": binary,
        "source_manifests": sources,
        "selected_positives": positives,
        "all_positive_attempts": attempts,
        "receiver_negatives": receiver,
        "s4_negatives": negative_s4,
        "scope": "Fixed live task and numeric contracts; independent seeds; pinned campaign sources. Provisional native statuses are retained. Source compatibility is not inferred from equal outcomes.",
    }
    if complete and not done:
        raise ValueError(
            f"Incomplete qualification: {len(missing)} positives, {len(missing_receiver)} receiver negatives, {len(missing_s4)} S4 negatives missing"
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", nargs="+", type=Path)
    parser.add_argument("--receiver-negative", action="append", default=[], type=Path)
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.campaign, args.receiver_negative, complete=args.complete)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "complete",
                    "qualified_positive_count",
                    "expected_positive_count",
                )
            }
        )
    )
