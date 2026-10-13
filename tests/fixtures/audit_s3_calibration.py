"""Seal a completed S3 qualification after independent receipt/export verification."""

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pypulseq as pp

from mriseqbench.catalog import Catalog

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_compatibility(source, hashes):
    """Require the actual score/physics dependencies to match qualified code."""
    drift = [
        p
        for p, digest in hashes.items()
        if not (ROOT / p).exists() or sha(ROOT / p) != digest
    ]
    relevant = [
        p
        for p in hashes
        if p.startswith(
            (
                "src/mriseqbench/evaluation/",
                "src/mriseqbench/phantoms/",
                "benchmark/objects/",
            )
        )
    ]
    allowed = []
    for rel in sorted(set(drift) & set(relevant)):
        if rel == "src/mriseqbench/evaluation/s2.py":
            # S3 imports these helpers, not S2's own version-dispatch entrypoint.
            names = {
                "reference_signal",
                "reference_index",
                "reference_indices",
                "phase_reference",
                "mechanism_probes",
                "canonical_response",
            }

            def nodes(path, names=names):
                return {
                    n.name: ast.dump(n, include_attributes=False)
                    for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name in names
                }

            assert nodes(source / rel) == nodes(ROOT / rel), (
                "A shared S2 helper changed after qualification"
            )
            allowed.append(
                {
                    "path": rel,
                    "reason": "all six S3-imported helper bodies are identical; S2-only changes",
                }
            )
        elif rel == "src/mriseqbench/evaluation/s2_sequence.py":

            def refocusing(path):
                return [
                    ast.dump(node, include_attributes=False)
                    for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "REFOCUSED"
                        for target in node.targets
                    )
                ]

            assert refocusing(source / rel) == refocusing(ROOT / rel), (
                "Shared RF classification changed"
            )
            allowed.append(
                {
                    "path": rel,
                    "reason": "S2-only waveform checks; the shared REFOCUSED family set is identical",
                }
            )
        elif rel.endswith("/worker.py"):
            # CLI/worker dispatch is separately tested; it does not implement scores.
            allowed.append(
                {
                    "path": rel,
                    "reason": "worker dispatch; release integration tests required",
                }
            )
        elif rel == "src/mriseqbench/evaluation/s3_sequence.py":

            def rest(path):
                tree = ast.parse(path.read_text())
                tree.body = [
                    node
                    for node in tree.body
                    if not (
                        isinstance(node, ast.FunctionDef)
                        and node.name == "research_pattern"
                    )
                ]
                return ast.dump(tree, include_attributes=False)

            assert rest(source / rel) == rest(ROOT / rel), (
                "S3 changed beyond research-domain checks"
            )
            frozen = load_module(
                "mriseqbench.evaluation.s3_sequence_qualified", source / rel
            )
            from mriseqbench.evaluation.s3_sequence import inspect_sequence

            # The sole release change rejects unsupported reduced-FOV lattices.
            # Prove that all qualified inputs retain identical parsed waveforms,
            # measured geometry, mask identities and checks before reusing data.
            verified = []
            for row in read(source.parent / "plan.json"):
                directory = Path(row["directory"])
                case = read(directory / "case.json")
                before = frozen.inspect_sequence(directory / "sequence.seq", case)
                after = inspect_sequence(directory / "sequence.seq", case)
                assert (
                    before.checks == after.checks
                    and before.measurements == after.measurements
                ), row["control"]
                for name in ("k", "adc", "indices", "offset", "te", "excitation"):
                    if hasattr(before, name):
                        np.testing.assert_array_equal(
                            getattr(before, name), getattr(after, name)
                        )
                verified.append(row["control"])
            allowed.append(
                {
                    "path": rel,
                    "reason": "Reduced-FOV research domain guard tightened; every qualification input has byte-identical parsed arrays, measurements and checks; all other syntax is identical",
                    "verified_controls": verified,
                }
            )
        else:
            raise AssertionError(
                f"Scoring/physics source changed after qualification: {rel}"
            )
    return {"live_source_drift": drift, "audited_non_scoring_drift": allowed}


def run(args):
    destination = args.directory.resolve()
    source = destination / "source"
    hashes = read(destination / "source-hashes.json")
    assert all(sha(source / rel) == digest for rel, digest in hashes.items()), (
        "Frozen source changed"
    )
    plan = read(destination / "plan.json")
    campaign = read(destination / "campaign.json")
    complete = None if args.partial else read(destination / "complete.json")
    assert len(plan) == campaign["controls"]
    if complete is not None:
        assert len(plan) == complete["completed"]
        assert complete["source_hashes_verified"] is True
    assert sha(destination / "plan.json") == campaign["plan_sha256"]
    assert sha(destination / "source-hashes.json") == campaign["source_hashes_sha256"]
    if args.partial:
        plan = [r for r in plan if (Path(r["directory"]) / "record.json").exists()]
    controls = load_module(
        "s3_frozen_controls", source / "tests/fixtures/s3_sequences.py"
    )
    frozen_catalog = Catalog(source)
    policy = frozen_catalog.get("evaluators", "s3").model_dump(mode="json")
    live_catalog = Catalog(ROOT)
    live_policy = live_catalog.get("evaluators", "s3").model_dump(mode="json")
    assert {k: v for k, v in policy.items() if k != "calibrated"} == {
        k: v for k, v in live_policy.items() if k != "calibrated"
    }, "Quality policy changed after qualification"
    assay = read(args.corruptions)
    assert {k: v for k, v in assay["policy"].items() if k != "calibrated"} == {
        k: v for k, v in policy.items() if k != "calibrated"
    }
    scored = {row["source_result"]: row for row in assay["records"]}
    if args.partial:
        # Audit exactly the sealed assay snapshot while other controls continue.
        plan = [
            row
            for row in plan
            if str(Path(row["directory"]) / "evaluation/result.json") in scored
        ]
    rows, exports, versions, python_versions = [], [], [], []
    with tempfile.TemporaryDirectory(prefix="s3-export-audit-") as temp:
        for row in plan:
            directory = Path(row["directory"])
            case_path, sequence_path = (
                directory / "case.json",
                directory / "sequence.seq",
            )
            assert (
                sha(case_path) == row["case_sha256"]
                and sha(sequence_path) == row["sequence_sha256"]
            )
            record = read(directory / "record.json")
            assert (
                record["case_sha256"] == row["case_sha256"]
                and record["sequence_sha256"] == row["sequence_sha256"]
            )
            result_path = Path(record["result"])
            assert sha(result_path) == record["result_sha256"]
            result, case = read(result_path), read(case_path)
            for name, path in result.get("artifacts", {}).items():
                assert sha(Path(path)) == record["artifact_sha256"][name]
            # Resolve metadata afresh from the frozen catalog, including its exact
            # nominal matrices, object definitions, physical effects and hardware.
            expected = frozen_catalog.resolve(
                frozen_catalog.get("tasks", case["task"]["id"]),
                frozen_catalog.get("physics", case["physics"]["id"]),
                case["object"]["id"],
                seed=case["seed"],
                hardware_ref=case["hardware"]["id"],
            )
            expected["evaluation"].update(
                spatial_seed=campaign["seed"],
                spatial_samples=campaign["initial_samples"],
                spatial_replicates=campaign["replicates"],
            )
            assert expected == case, f"Frozen catalog/input mismatch: {row['control']}"
            live_case = live_catalog.resolve(
                live_catalog.get("tasks", case["task"]["id"]),
                live_catalog.get("physics", case["physics"]["id"]),
                case["object"]["id"],
                seed=case["seed"],
                hardware_ref=case["hardware"]["id"],
            )
            live_case["evaluation"].update(
                spatial_seed=campaign["seed"],
                spatial_samples=campaign["initial_samples"],
                spatial_replicates=campaign["replicates"],
                calibrated=case["evaluation"]["calibrated"],
            )
            assert live_case == case, f"Release catalog/input drift: {row['control']}"
            assert case["task"]["sequence"]["matrix"] in ([32, 32], [32, 32, 32])
            mutation = (
                row["control"].split("--")[-1] if "negative" in row["role"] else None
            )
            recreated = Path(temp) / "recreated.seq"
            controls.make_sequence(case, mutation, **row["control_options"]).write(
                str(recreated)
            )
            assert sha(recreated) == row["sequence_sha256"], (
                f"Sequence regeneration mismatch: {row['control']}"
            )
            # A second Pulseq trajectory implementation independently reads the
            # exported bytes; its RF markers are appropriate for authored controls.
            sequence = pp.Sequence(pp.Opts(B0=case["hardware"]["field_strength_T"]))
            sequence.read(str(sequence_path))
            timing, errors = sequence.check_timing()
            assert timing, errors
            k_adc, _, _, _, adc_times = sequence.calculate_kspace()
            export = {
                "control": row["control"],
                "sequence_sha256": row["sequence_sha256"],
                "regenerated_sha256": sha(recreated),
                "adc_samples": len(adc_times),
                "duration_s": float(sequence.duration()[0]),
                "independent_trajectory_sha256": hashlib.sha256(
                    np.ascontiguousarray(k_adc).tobytes()
                ).hexdigest(),
            }
            candidate_pass = False
            if "images" in result.get("artifacts", {}):
                assert (
                    result["sequence_valid"] is True
                    and result["numerically_converged"] is True
                )
                with np.load(
                    result["artifacts"]["images"], allow_pickle=False
                ) as stored:
                    for key in stored.files:
                        assert np.isfinite(stored[key]).all(), (row["control"], key)
                    delta = float(np.max(abs(k_adc.T - stored["k_cycles_per_m"])))
                    assert delta < 1e-4, (row["control"], delta)
                    assert np.max(abs(adc_times - stored["adc_times_s"])) < 1e-9
                    export["max_independent_trajectory_error_cycles_per_m"] = delta
                rescore = scored[str(result_path)]
                assert rescore["numerically_converged"] is True
                candidate_pass = rescore["native"]["passed"]
                assert isinstance(candidate_pass, bool)
                assert candidate_pass == all(
                    c["passed"] is True for c in result["image_checks"]
                )
                provenance = read(Path(result["artifacts"]["provenance"]))
                versions.append(provenance["koma_versions"])
                python_versions.append(provenance["versions"])
            else:
                assert result["status"] == "failed" and any(
                    not c["passed"] for c in result["checks"]
                )
            assert record.get("error_kind") is None
            rows.append(
                {
                    **record,
                    "candidate_pass": candidate_pass,
                    "case": case,
                    "source_report": result,
                }
            )
            exports.append(export)
    assert all(r["candidate_pass"] for r in rows if r["role"] == "ideal_positive"), (
        "An ideal positive failed qualification"
    )
    assert all(not r["candidate_pass"] for r in rows if "negative" in r["role"]), (
        "A submitted negative was accepted"
    )
    stress = {}
    for row in assay["records"]:
        if not row.get("native", {}).get("passed") or not row.get(
            "numerically_converged"
        ):
            continue
        for name, score in row["corruptions"].items():
            counts = stress.setdefault(
                name, {"controls": 0, "rejected": 0, "accepted": 0, "unresolved": 0}
            )
            counts["controls"] += 1
            counts[
                "rejected"
                if score["passed"] is False
                else "accepted"
                if score["passed"] is True
                else "unresolved"
            ] += 1
    assert stress and all(
        c["accepted"] == c["unresolved"] == 0 for c in stress.values()
    ), "An image stress control was accepted or unresolved"
    assert all(version == versions[0] for version in versions), (
        "Software versions differ across qualification workers"
    )
    assert all(version == python_versions[0] for version in python_versions), (
        "Python dependency versions differ across qualification workers"
    )
    counts = {
        role: {
            "controls": sum(r["role"] == role for r in rows),
            "passed": sum(r["role"] == role and r["candidate_pass"] for r in rows),
            "failed": sum(r["role"] == role and not r["candidate_pass"] for r in rows),
        }
        for role in sorted({r["role"] for r in rows})
    }
    coverage = {
        "families": dict(
            Counter(r["case"]["task"]["id"].removesuffix("_research") for r in rows)
        ),
        "levels": dict(Counter(r["case"]["object_level"] for r in rows)),
        "hardware": dict(Counter(r["case"]["hardware"]["id"] for r in rows)),
        "physics": dict(Counter(r["case"]["physics"]["id"] for r in rows)),
        "objects": dict(Counter(r["case"]["object"]["id"] for r in rows)),
    }
    assert args.partial or (
        len(coverage["families"]) == 9
        and len(coverage["objects"]) == 5
        and len(coverage["hardware"]) == 4
        and set(coverage["levels"]) == {"P0", "P1", "P2", "P3"}
    )
    compatibility = source_compatibility(source, hashes)
    replay_reuse = None
    reuse_path = destination / "native-replay-reuse.json"
    if reuse_path.exists():
        replay_reuse = read(reuse_path)
        migrations = replay_reuse.get("migrations", [replay_reuse])
        for migration in migrations:
            prior = Path(migration["source"])
            prior_hashes = read(prior / "source-hashes.json")
            assert all(
                sha(prior / "source" / rel) == digest
                for rel, digest in prior_hashes.items()
            )
            for rel, digest in migration["native_dependencies"].items():
                assert hashes[rel] == prior_hashes[rel] == digest
            native_change = migration["conditional_native_change"]
            if native_change is not None:
                assert (
                    hashes[native_change["path"]] == native_change["destination_sha256"]
                )
                assert (
                    prior_hashes[native_change["path"]]
                    == native_change["source_sha256"]
                )
                assert native_change["normalization_verified"] is True
            for change in migration.get("orchestration_equivalence", []):
                assert hashes[change["path"]] == change["destination_sha256"]
                assert prior_hashes[change["path"]] == change["source_sha256"]
                assert change["native_call_equivalence_verified"] is True
            for receipt in migration["records"]:
                for signal in receipt["signals"]:
                    assert native_change is None or not any(
                        e["kind"] == "motion" and e["model"] == "rigid_keyframes"
                        for e in signal["effects"]
                    )
                    path = Path(signal["source"])
                    assert sha(path) == signal["source_sha256"]
                    if signal["source_array"] is None:
                        assert signal["source_sha256"] == signal["destination_sha256"]
                    else:
                        name, index = signal["source_array"].rstrip("]").split("[")
                        with np.load(path, allow_pickle=False) as stored:
                            value = stored[name][int(index)]
                        stream = io.BytesIO()
                        np.save(stream, value, allow_pickle=False)
                        assert (
                            hashlib.sha256(stream.getvalue()).hexdigest()
                            == signal["destination_sha256"]
                        )
    cross_control_reuse = None
    cross_control_path = destination / "cross_control_ideal_reuse.json"
    if cross_control_path.exists():
        verifier = load_module(
            "s3_ideal_reuse_audit", ROOT / "tests/fixtures/reuse_s3_ideal_replays.py"
        )
        cross_control_reuse = verifier.verify(destination, read(cross_control_path))
        cross_control_reuse["receipt_sha256"] = sha(cross_control_path)
        cross_control_reuse["verifier_sha256"] = sha(
            ROOT / "tests/fixtures/reuse_s3_ideal_replays.py"
        )
        cross_control_reuse["receipt"] = read(cross_control_path)
    document = {
        "qualified": not args.partial,
        "partial": args.partial,
        "evaluation_contract": "s3@1",
        "campaign": campaign,
        "policy": live_policy,
        "scope": "nominal-grid family/object/prompt controls, every family on four hardware profiles, every compatible registered physics profile, submitted structural/receiver negatives and derived image stress assays; not an exhaustive 800-case trial or scanner calibration",
        "qualification_directory": str(destination.relative_to(ROOT)),
        "complete_sha256": None if args.partial else sha(destination / "complete.json"),
        "source_hashes": hashes,
        "roles": counts,
        "coverage": coverage,
        "stress_controls": stress,
        "versions": versions[0],
        "python_versions": python_versions[0],
        "audit_code_sha256": sha(Path(__file__)),
        "corruption_assay_sha256": sha(args.corruptions),
        "source_compatibility": compatibility,
        "native_replay_reuse": replay_reuse,
        "cross_control_ideal_reuse": cross_control_reuse,
        "independent_exports": exports,
        "records": rows,
    }
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "qualified": not args.partial,
                "controls": len(rows),
                "roles": counts,
                "stress": stress,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--corruptions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Audit completed controls without claiming qualification",
    )
    run(parser.parse_args())
