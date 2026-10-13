"""Seal and verify S4 source, submitted bytes, results and per-frame evidence.

This companion never modifies frozen source, sequence, result or native images.
A running campaign may be audited incrementally; --complete requires all cases.
"""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

FAMILIES = ("adiabatic_inversion", "t2_prepared", "t1_mapping", "t2_mapping")
HARDWARE = ("low_field", "standard_1_5t", "high_performance")
MUTATIONS = (
    "missing_trigger",
    "wrong_trigger_channel",
    "late_adc",
    "tiny_flip",
    "missing_preparation",
    "wrong_preparation_time",
    "bad_refocusing",
    "unbalanced",
    "no_phase_cycle",
    "receiver_phase",
    "constant_contrast",
    "narrow_preparation",
)


def expected_negatives(families=FAMILIES, hardware_profiles=HARDWARE):
    return {
        f"{family}--{hardware}--{mutation}"
        for family in families
        for hardware in hardware_profiles
        for mutation in MUTATIONS
        if (mutation != "narrow_preparation" or hardware == "high_performance")
        and (mutation != "bad_refocusing" or family in {"t2_prepared", "t2_mapping"})
        and (mutation != "constant_contrast" or family in {"t1_mapping", "t2_mapping"})
    }


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(path, root):
    path = Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Unexpected artifact location: {path}")
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"Artifact changed while reading: {path}")
    return {
        "path": str(path.relative_to(root)),
        "sha256": digest,
        "bytes": after.st_size,
    }


def audit(root, complete=False, families=FAMILIES, hardware_profiles=HARDWARE):
    root = root.resolve()
    source = root / "source"
    source_manifest = json.loads((root / "source_manifest.json").read_text())
    for name, digest in source_manifest.items():
        if sha(source / name) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    public = yaml.safe_load((source / "benchmark/evaluators/s4.yaml").read_text())
    bart = public["reconstruction"].startswith("bart_")
    tool = (
        json.loads((source / "reconstruction_tool.json").read_text()) if bart else None
    )
    if bart and sha(tool["executable"]) != tool["identity"]["binary_sha256"]:
        raise ValueError("The BART executable changed during qualification")
    rows_path = root / "results.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    if len({row["name"] for row in rows}) != len(rows):
        raise ValueError("Duplicate submitted case names in qualification results")
    sealed_path = root / "evidence_manifest.json"
    prior = json.loads(sealed_path.read_text()) if sealed_path.exists() else {}
    entries = {}
    for row in rows:
        name = row["name"]
        directory = root / "cases" / name
        for filename, key in [
            ("case.json", "case_sha256"),
            ("sequence.seq", "sequence_sha256"),
            ("result.json", "result_sha256"),
        ]:
            if sha(directory / filename) != row[key]:
                raise ValueError(f"Frozen case evidence changed: {name}/{filename}")
        case = json.loads((directory / "case.json").read_text())
        if case["evaluation"]["reconstruction"] != public["reconstruction"]:
            raise ValueError(f"Reconstruction contract changed: {name}")
        contract_keys = (
            "spatial_samples",
            "spatial_replicates",
            "spatial_max_samples",
            "image_precision",
            "case_image_criteria",
            "preparation_effect_fraction_min",
            "mapping_relative_error_max",
            "intrinsic_tissue_response",
        )
        if public.get("mapping_relative_error_precision") is not None:
            contract_keys += ("mapping_relative_error_precision",)
        for key in contract_keys:
            if case["evaluation"][key] != public[key]:
                raise ValueError(
                    f"Official qualification contract changed: {name}/{key}"
                )
        report = json.loads((directory / "result.json").read_text())
        if report.get("status") != row["status"]:
            raise ValueError(f"Result status does not match receipt: {name}")
        if (
            bart
            and report.get("frames")
            and any("images" in f.get("artifacts", {}) for f in report["frames"])
        ):
            reports = [report, *report["frames"]]
            for item in reports:
                if item is not report and "images" not in item.get("artifacts", {}):
                    continue
                provenance = item.get("reconstruction_provenance", {})
                for key in ("engine", "version", "binary_sha256", "source_commit"):
                    if provenance.get(key) != tool["identity"].get(key):
                        raise ValueError(f"BART provenance changed: {name}/{key}")
                if not provenance.get("commands"):
                    raise ValueError(f"No actual BART reconstruction commands: {name}")
        record = {
            "case": artifact(directory / "case.json", root),
            "sequence": artifact(directory / "sequence.seq", root),
            "result": artifact(directory / "result.json", root),
            "frames": [],
        }
        for index, frame in enumerate(report.get("frames", [])):
            record["frames"].append(
                {
                    "index": index,
                    "artifacts": {
                        key: artifact(path, root)
                        for key, path in frame.get("artifacts", {}).items()
                    },
                }
            )
        if name in prior.get("cases", {}) and prior["cases"][name] != record:
            raise ValueError(f"Previously sealed S4 evidence changed: {name}")
        entries[name] = record
    expected_positive = {
        f"{family}--{hardware}--positive"
        for family in families
        for hardware in hardware_profiles
    }
    positive = [row for row in rows if row["mutation"] is None]
    negative = [row for row in rows if row["mutation"] is not None]
    actual_positive = {row["name"] for row in positive}
    actual_negative = {row["name"] for row in negative}
    expected_negative = expected_negatives(families, hardware_profiles)
    if (
        not actual_positive <= expected_positive
        or not actual_negative <= expected_negative
    ):
        raise ValueError("Unexpected family, hardware or corruption in qualification")
    if complete and (
        actual_positive != expected_positive or actual_negative != expected_negative
    ):
        raise ValueError(
            f"Incomplete official S4 qualification: require {len(expected_positive)} "
            f"positives and {len(expected_negative)} exported negatives for the declared scope"
        )
    result = {
        "source_manifest_sha256": sha(root / "source_manifest.json"),
        "auditor_sha256": sha(__file__),
        "cases": entries,
        "scope": {"tasks": list(families), "hardware": list(hardware_profiles)},
        "expected_positive_count": len(expected_positive),
        "expected_negative_count": len(expected_negative),
        "complete": actual_positive == expected_positive
        and actual_negative == expected_negative,
        "positive_count": len(positive),
        "qualified_positive_count": sum(
            row["qualified_control"] is True for row in positive
        ),
        "unqualified_positive_count": sum(
            row["qualified_control"] is not True for row in positive
        ),
        "negative_count": len(negative),
        "rejected_negative_count": sum(
            row["negative_rejected"] is True for row in negative
        ),
        "unrejected_negative_count": sum(
            row["negative_rejected"] is not True for row in negative
        ),
        "numerically_unresolved_count": sum(
            row["status"] == "numerically_unresolved" for row in rows
        ),
        "evaluator_error_count": sum(
            row["status"] == "evaluator_error" for row in rows
        ),
        "applicable_negatives": [
            {
                "task": row["task"],
                "hardware": row["hardware"],
                "mutation": row["mutation"],
            }
            for row in negative
        ],
    }
    sealed_path.write_text(json.dumps(result, indent=2) + "\n")
    return {
        key: value
        for key, value in result.items()
        if key not in {"cases", "applicable_negatives"}
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument(
        "--hardware", nargs="+", choices=HARDWARE, default=list(HARDWARE)
    )
    args = parser.parse_args()
    print(
        json.dumps(audit(args.run, args.complete, args.tasks, args.hardware), indent=2)
    )


if __name__ == "__main__":
    main()
