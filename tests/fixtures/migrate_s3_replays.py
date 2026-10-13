"""Reuse verified native signals after the S3 lattice branch-cut correction.

No verdict, reconstructed image, reference or numerical interval is reused.
Native simulation dependencies and submitted inputs must be byte-identical.
Every imported signal has an immutable source receipt; grading runs afresh.
"""

import argparse
import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from mriseqbench.evaluation.s1_sampling import sampled_phantom
from mriseqbench.evaluation.s3_sequence import inspect_sequence


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def implementation(root):
    return hashlib.sha256(
        b"".join(
            p.name.encode() + p.read_bytes()
            for p in sorted((root / "src/mriseqbench/evaluation").glob("s[123]*.*"))
        )
    ).hexdigest()


def key(identity, spins, effects, steps):
    digest = hashlib.sha256(identity.encode())
    digest.update(
        json.dumps(
            [effects, steps, spins.materials, spins.source_sha256], sort_keys=True
        ).encode()
    )
    for values in (spins.positions, spins.labels, spins.weight, spins.properties):
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest() + ".npy"


def run(args):
    old, new = args.source.resolve(), args.destination.resolve()
    assert old != new and not (new / "complete.json").exists()
    old_hashes, new_hashes = (
        read(old / "source-hashes.json"),
        read(new / "source-hashes.json"),
    )
    for root, hashes in ((old, old_hashes), (new, new_hashes)):
        assert all(
            sha(root / "source" / rel) == digest for rel, digest in hashes.items()
        )
    dependencies = {
        rel: digest
        for rel, digest in old_hashes.items()
        if rel.startswith(("src/mriseqbench/", "benchmark/"))
        and rel
        not in {
            "src/mriseqbench/evaluation/s3_sequence.py",
            "src/mriseqbench/evaluation/s1_koma.jl",
            "src/mriseqbench/evaluation/s1.py",
            "src/mriseqbench/evaluation/s3.py",
        }
    }
    assert all(new_hashes.get(rel) == digest for rel, digest in dependencies.items()), (
        "Native dynamics, receive maps, quadrature, data, policy or other scoring dependencies changed"
    )
    assert old_hashes["Project.toml"] == new_hashes["Project.toml"]
    assert old_hashes["Manifest.toml"] == new_hashes["Manifest.toml"]
    orchestration = []
    for rel, names in {
        "src/mriseqbench/evaluation/s3.py": {
            "receive_terms",
            "receive_maps",
            "signal_shape",
            "simulate_channels",
        },
        "src/mriseqbench/evaluation/s1.py": None,
    }.items():
        before_tree = ast.parse((old / "source" / rel).read_text())
        after_tree = ast.parse((new / "source" / rel).read_text())
        ignored = (
            {"evaluate_s3", "temporal_image_error"}
            if names is not None
            else {"evaluate_s1"}
        )

        def remainder(tree, ignored=ignored):
            return [
                ast.dump(n, include_attributes=False)
                for n in tree.body
                if not (isinstance(n, ast.FunctionDef) and n.name in ignored)
            ]

        assert remainder(before_tree) == remainder(after_tree), (
            "Simulation constants or shared helper bodies changed"
        )
        if names is not None:

            def selected(tree, names=names):
                return [
                    ast.dump(n, include_attributes=False)
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name in names
                ]

            assert selected(before_tree) == selected(after_tree), (
                "Receive dynamics changed"
            )
        else:
            native_names = {
                "simulator",
                "simulate",
                "ReplayCache",
                "sampled_phantom",
                "replay",
            }

            def calls(tree, native_names=native_names):
                return [
                    ast.dump(n, include_attributes=False)
                    for n in ast.walk(tree)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id in native_names
                ]

            assert calls(before_tree) == calls(after_tree), (
                "Native simulation call arguments changed"
            )
        orchestration.append(
            {
                "path": rel,
                "source_sha256": old_hashes[rel],
                "destination_sha256": new_hashes[rel],
                "native_call_equivalence_verified": True,
            }
        )
    native_rel = "src/mriseqbench/evaluation/s1_koma.jl"
    native_before = (old / "source" / native_rel).read_text()
    native_after = (new / "source" / native_rel).read_text()
    native_changed = native_before != native_after
    start = native_after.index("function retain_keyframe_jumps(")
    end = native_after.index("function discretize_case(", start)
    normalized = native_after[:start] + native_after[end:]
    normalized = (
        normalized.replace(
            "function discretize_case(seq, params, motion, has_dynamic, effects=Any[])",
            "function discretize_case(seq, params, motion, has_dynamic)",
        )
        .replace("    seqd = retain_keyframe_jumps(seq,seqd,effects)\n", "")
        .replace(
            "discretize_case(seq,params,obj.motion,has_dynamic,c.effects)",
            "discretize_case(seq,params,obj.motion,has_dynamic)",
        )
    )
    assert not native_changed or normalized == native_before, (
        "Unexpected native implementation change"
    )
    old_implementation, new_implementation = (
        implementation(old / "source"),
        implementation(new / "source"),
    )
    old_plan = {r["control"]: r for r in read(old / "plan.json")}
    receipts = []
    for row in read(new / "plan.json"):
        prior = old_plan[row["control"]]
        before, after = Path(prior["directory"]), Path(row["directory"])
        assert prior["case_sha256"] == row["case_sha256"] == sha(after / "case.json")
        assert (
            prior["sequence_sha256"]
            == row["sequence_sha256"]
            == sha(after / "sequence.seq")
        )
        assert sha(before / "case.json") == prior["case_sha256"]
        assert sha(before / "sequence.seq") == prior["sequence_sha256"]
        case = read(after / "case.json")
        acq = inspect_sequence(after / "sequence.seq", case)
        if not all(c["passed"] for c in acq.checks):
            continue
        old_cache, new_cache = (
            before / "evaluation/physical/s3/replays",
            after / "evaluation/physical/s3/replays",
        )
        new_cache.mkdir(parents=True, exist_ok=True)
        suffix = row["sequence_sha256"] + json.dumps(
            {"case": case, "encoded_integrator": "bloch"}, sort_keys=True
        )
        old_identity, new_identity = (
            old_implementation + suffix,
            new_implementation + suffix,
        )
        effects = case["physics"]["effects"]
        variants = [[], effects] if effects else [[]]
        chunk_size = min(case["evaluation"]["spatial_max_samples"], 32768)
        imported = []

        def save(
            values,
            spins,
            effect,
            steps,
            source_path,
            source_key=None,
            *,
            new_cache=new_cache,
            new_identity=new_identity,
            imported=imported,
        ):
            if native_changed and any(
                e["kind"] == "motion" and e["model"] == "rigid_keyframes"
                for e in effect
            ):
                return  # Corrected jump integration requires a fresh native replay.
            assert np.isfinite(values).all()
            target = new_cache / key(new_identity, spins, effect, steps)
            if target.exists():
                np.testing.assert_array_equal(
                    np.load(target, allow_pickle=False), values
                )
                return
            with target.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            imported.append(
                {
                    "source": str(source_path),
                    "source_sha256": sha(source_path),
                    "source_array": source_key,
                    "destination": str(target),
                    "destination_sha256": sha(target),
                    "samples": len(spins.positions),
                    "effects": effect,
                    "steps": steps,
                }
            )

        record_path = before / "record.json"
        if record_path.exists():
            record = read(record_path)
            result_path = Path(record["result"])
            assert sha(result_path) == record["result_sha256"]
            result = read(result_path)
            assert all(
                sha(Path(path)) == record["artifact_sha256"][name]
                for name, path in result.get("artifacts", {}).items()
            )
            if "images" in result.get("artifacts", {}):
                image_path = Path(result["artifacts"]["images"])
                provenance = read(Path(result["artifacts"]["provenance"]))
                steps = {
                    "step_s": provenance["dynamic_step_s"][0],
                    "rf_step_s": provenance["rf_step_s"][0],
                }
                limit = max(64, int(chunk_size * min(1, steps["rf_step_s"] / 20e-6)))
                with np.load(image_path, allow_pickle=False) as images:
                    np.testing.assert_allclose(
                        acq.k, images["k_cycles_per_m"], rtol=0, atol=0
                    )
                    for replicate in range(provenance["spatial_replicates"]):
                        spins = sampled_phantom(
                            case,
                            new / "source",
                            provenance["spatial_samples"],
                            replicate,
                        )
                        spins = replace(
                            spins,
                            weight=np.broadcast_to(
                                spins.weight, (len(spins.positions),)
                            ),
                        )
                        if len(spins.positions) > limit:
                            continue  # A retained sum cannot recover individual chunk signals.
                        for effect in variants:
                            name = (
                                "koma_physical_raw_replicates"
                                if effect
                                else "koma_ideal_raw_replicates"
                            )
                            save(
                                images[name][replicate],
                                spins,
                                effect,
                                steps,
                                image_path,
                                f"{name}[{replicate}]",
                            )
        progress_path = before / "evaluation/physical/s3/progress.json"
        if old_cache.is_dir() and progress_path.exists():
            progress = read(progress_path)
            stages = progress.get("spatial_refinement", [])
            counts = {r["samples"] for r in stages}
            counts.add(
                progress.get("current_samples")
                or (
                    2 * stages[-1]["samples"]
                    if stages
                    else case["evaluation"]["spatial_samples"]
                )
            )
            steps_list = [{"step_s": 0.001, "rf_step_s": 20e-6}]
            for temporal in progress.get("temporal_refinement", []):
                base = {k: temporal[k] for k in ("step_s", "rf_step_s")}
                steps_list.extend([base, {k: v / 2 for k, v in base.items()}])
            for count in sorted(counts):
                for replicate in range(case["evaluation"]["spatial_replicates"]):
                    full = sampled_phantom(case, new / "source", count, replicate)
                    weights = np.broadcast_to(full.weight, (len(full.positions),))
                    for steps in steps_list:
                        size = max(
                            64, int(chunk_size * min(1, steps["rf_step_s"] / 20e-6))
                        )
                        for start in range(0, len(full.positions), size):
                            part = replace(
                                full,
                                positions=full.positions[start : start + size],
                                labels=full.labels[start : start + size],
                                weight=weights[start : start + size],
                            )
                            for effect in variants:
                                path = old_cache / key(
                                    old_identity, part, effect, steps
                                )
                                if path.exists():
                                    save(
                                        np.load(path, allow_pickle=False),
                                        part,
                                        effect,
                                        steps,
                                        path,
                                    )
        if imported:
            receipts.append(
                {
                    "control": row["control"],
                    "case_sha256": row["case_sha256"],
                    "sequence_sha256": row["sequence_sha256"],
                    "signals": imported,
                }
            )
    document = {
        "method": "SHA-verified retained native signals with byte-identical native simulation dependencies and submitted inputs; no verdict, reconstructed image or reference reuse",
        "source": str(old),
        "destination": str(new),
        "source_implementation_sha256": old_implementation,
        "destination_implementation_sha256": new_implementation,
        "native_dependencies": dependencies,
        "conditional_native_change": {
            "path": native_rel,
            "source_sha256": old_hashes[native_rel],
            "destination_sha256": new_hashes[native_rel],
            "normalization_verified": True,
            "reuse_excludes": "all physical rigid_keyframes effects; ideal signals remain unchanged",
        }
        if native_changed
        else None,
        "orchestration_equivalence": orchestration,
        "records": receipts,
        "migration_code_sha256": sha(Path(__file__)),
    }
    receipt_path = new / "native-replay-reuse.json"
    existing = read(receipt_path) if receipt_path.exists() else {"migrations": []}
    assert "migrations" in existing
    existing["migrations"].append(document)
    receipt_path.write_text(json.dumps(existing, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "controls": len(receipts),
                "signals": sum(len(r["signals"]) for r in receipts),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    run(parser.parse_args())
