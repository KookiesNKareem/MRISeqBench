"""Nominal-grid S2 qualification, with frozen inputs and bounded isolated workers.

Run each phase into a new directory. No participant sequences are calibration
inputs. Physics controls characterize robustness; a legal acquisition is not
automatically a positive image-quality control under every perturbation.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mriseqbench.catalog import Catalog
from mriseqbench.process import execute

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = (
    "epi_single_shot",
    "epi_multishot",
    "radial_readout",
    "spiral_gre",
    "rosette_gre",
    "concentric_rings",
    "propeller_tse",
    "ute_radial",
    "tse_t2w",
    "grase",
    "stack_of_stars",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jobs(phase):
    if phase == "qualification_all":
        for component in ("ideal", "physics_all", "hardware", "negative"):
            yield from jobs(component)
        return
    if phase == "physics_all":
        catalog = Catalog(ROOT)
        for fam in FAMILIES:
            obj = "volume_ellipsoids" if fam == "stack_of_stars" else "tissue_discs"
            for path in sorted((ROOT / "benchmark/physics").glob("*.yaml")):
                if path.stem == "ideal":
                    continue
                try:
                    catalog.resolve(
                        catalog.get("tasks", fam),
                        catalog.get("physics", path.stem),
                        obj,
                    )
                except ValueError:
                    continue
                yield fam, obj, path.stem, False, "standard", None
        return
    if phase == "qualification":
        for component in ("ideal", "physics", "profiles", "hardware", "negative"):
            yield from jobs(component)
        return
    for family in FAMILIES:
        volume = family == "stack_of_stars"
        objects = (
            ("uniform_sphere", "volume_ellipsoids")
            if volume
            else ("uniform_disc", "tissue_discs", "nested_ellipses")
        )
        representative = objects[1]
        if phase == "pilot":
            yield family, representative, "ideal", False, "standard", None
        elif phase == "ideal":
            for obj in objects:
                for research in (False, True):
                    yield family, obj, "ideal", research, "standard", None
        elif phase == "physics":
            profiles = (
                ("b0_smooth_3d", "rigid_motion_3d", "fields_and_motion_3d")
                if volume
                else ("b0_smooth", "rigid_motion", "fields_and_motion")
            )
            for physics in profiles:
                yield family, representative, physics, False, "standard", None
        elif phase == "hardware" and family in {
            "epi_single_shot",
            "radial_readout",
            "tse_t2w",
        }:
            for hardware in ("low_field", "standard_1_5t", "high_performance"):
                yield family, representative, "ideal", False, hardware, None
    if phase == "profiles":
        for physics in (
            "b0_offset",
            "b0_localized",
            "b0_drift",
            "rigid_steps",
            "deformation",
            "dynamic_tissue",
        ):
            yield "radial_readout", "tissue_discs", physics, False, "standard", None
    if phase == "negative":
        for family, mutation in (
            ("epi_single_shot", "no_alternation"),
            ("epi_multishot", "missing_line"),
            ("radial_readout", "repeated_spokes"),
            ("radial_readout", "wrong_fov"),
            ("tse_t2w", "no_refocusing"),
            ("tse_t2w", "lying_refocusing_label"),
            ("propeller_tse", "repeated_blades"),
        ):
            yield family, "tissue_discs", "ideal", False, "standard", mutation


def sources():
    paths = [
        ROOT / p for p in ("Project.toml", "Manifest.toml", "pyproject.toml", "uv.lock")
    ]
    paths += sorted((ROOT / "src/mriseqbench").rglob("*.py"))
    paths += sorted((ROOT / "src/mriseqbench").rglob("*.jl"))
    paths += sorted((ROOT / "benchmark").rglob("*.yaml"))
    paths += sorted((ROOT / "benchmark").rglob("*.json"))
    paths += sorted((ROOT / "benchmark").rglob("*.jl"))
    paths += [
        Path(__file__),
        ROOT / "tests/fixtures/s2_sequences.py",
        ROOT / "tests/fixtures/s1_sequences.py",
        ROOT / "tests/fixtures/score_s2_calibration.py",
    ]
    paths += [
        ROOT / "tests/evaluation" / name
        for name in (
            "test_s2.py",
            "test_s2_corruptions.py",
            "test_s1_case.py",
            "test_koma_engine.py",
            "test_registered_physics.py",
        )
    ]
    return paths


def run(args):
    out = args.destination.resolve()
    records = []
    worker_offset = 0
    if args.resume:
        hashes = json.loads((out / "source-hashes.json").read_text())
        assert all(
            sha(out / "source" / rel) == digest for rel, digest in hashes.items()
        ), "Frozen source changed"
        plan = json.loads((out / "plan.json").read_text())
        pending = []
        for row in plan:
            directory = Path(row["directory"])
            assert sha(directory / "case.json") == row["case_sha256"]
            assert sha(directory / "sequence.seq") == row["sequence_sha256"]
            record_path = directory / "record.json"
            if not record_path.exists():
                pending.append(row)
                continue
            record = json.loads(record_path.read_text())
            assert record["case_sha256"] == row["case_sha256"]
            assert record["sequence_sha256"] == row["sequence_sha256"]
            result_path = Path(record["result"])
            assert sha(result_path) == record["result_sha256"]
            result = json.loads(result_path.read_text())
            assert all(
                sha(Path(path)) == record["artifact_sha256"][name]
                for name, path in result.get("artifacts", {}).items()
            )
            assert record["status"] in {"passed", "failed"}
            records.append(record)
        plan = pending
        worker_offset = 1 + max(
            (
                int(p.name.removeprefix("worker-"))
                for p in out.glob("worker-*")
                if p.is_dir()
            ),
            default=-1,
        )
        print(json.dumps({"resumed": len(records), "pending": len(plan)}), flush=True)
    else:
        out.mkdir(parents=True, exist_ok=False)
        hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources()}
        for rel in hashes:
            target = out / "source" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, target)
        (out / "source-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
        catalog = Catalog(ROOT)
        spec = importlib.util.spec_from_file_location(
            "controls", ROOT / "tests/fixtures/s2_sequences.py"
        )
        controls = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(controls)
        plan = []
        for fam, obj, physics, research, hardware, mutation in jobs(args.phase):
            if args.family and fam not in args.family:
                continue
            task = fam + ("_research" if research else "")
            case = catalog.resolve(
                catalog.get("tasks", task),
                catalog.get("physics", physics),
                obj,
                hardware_ref=hardware,
            )
            case["evaluation"]["spatial_seed"] = args.seed
            if args.initial_samples is not None:
                case["evaluation"]["spatial_samples"] = args.initial_samples
                case["evaluation"]["spatial_max_samples"] = max(
                    args.initial_samples, case["evaluation"]["spatial_max_samples"]
                )
            if (
                args.volume_samples is not None
                and len(case["task"]["sequence"]["matrix"]) == 3
                and physics != "ideal"
            ):
                case["evaluation"]["spatial_samples"] = args.volume_samples
            options = (
                controls.physics_control_options(case)
                if args.robust_physics and physics != "ideal"
                else {}
            )
            name = "--".join(x for x in (task, obj, physics, hardware, mutation) if x)
            directory = out / name
            directory.mkdir()
            controls.make_sequence(case, mutation, **options).write(
                str(directory / "sequence.seq")
            )
            (directory / "case.json").write_text(json.dumps(case, indent=2) + "\n")
            if mutation is None:
                from mriseqbench.evaluation.preflight import preflight
                from mriseqbench.evaluation.s2_sequence import inspect_sequence

                checks = preflight(directory / "sequence.seq", case)
                checks += inspect_sequence(directory / "sequence.seq", case).checks
                assert all(c["passed"] for c in checks), (
                    name,
                    [c for c in checks if not c["passed"]],
                )
            plan.append(
                {
                    "control": name,
                    "role": "structural_negative"
                    if mutation
                    else "ideal_positive"
                    if physics == "ideal"
                    else "physics_characterization",
                    "control_options": options,
                    "sequence_sha256": sha(directory / "sequence.seq"),
                    "case_sha256": sha(directory / "case.json"),
                    "directory": str(directory),
                }
            )
        (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    if args.plan_only:
        print(json.dumps({"planned": len(plan), "destination": str(out)}), flush=True)
        return

    def once(batch_number, batch):
        directory = out / f"worker-{worker_offset + batch_number:03d}"
        directory.mkdir()
        batch_path = directory / "batch.json"
        batch_path.write_text(json.dumps(batch, indent=2) + "\n")
        execution = execute(
            [
                sys.executable,
                str(
                    Path(__file__)
                    if args.resume
                    else out / "source/tests/fixtures/calibrate_s2.py"
                ),
                "--batch",
                str(batch_path),
                "--time-budget",
                str(args.time_budget),
                *(["--source-root", str(out / "source")] if args.resume else []),
            ],
            ROOT,
            None,
            directory / "worker",
            env={**os.environ, "PYTHONPATH": str(out / "source/src")},
        )
        records = []
        for row in batch:
            record_path = Path(row["directory"]) / "record.json"
            if not record_path.exists():
                raise RuntimeError(
                    f"Calibration interrupted without a verdict: {row['directory']}; {execution}"
                )
            record = json.loads(record_path.read_text())
            records.append({**record, "execution": execution})
        return records

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(once, j, plan[i : i + args.batch_size])
            for j, i in enumerate(range(0, len(plan), args.batch_size))
        ]
        for future in as_completed(futures):
            batch_records = future.result()
            records.extend(batch_records)
            for record in batch_records:
                print(
                    json.dumps(
                        {
                            k: record[k]
                            for k in (
                                "control",
                                "status",
                                "seconds",
                                "failed_checks",
                                "reason",
                            )
                        }
                    ),
                    flush=True,
                )
            (out / "summary.json").write_text(
                json.dumps({"phase": args.phase, "records": records}, indent=2) + "\n"
            )
    assert all(sha(out / "source" / rel) == digest for rel, digest in hashes.items()), (
        "Frozen source changed during calibration"
    )
    (out / "complete.json").write_text(
        json.dumps(
            {
                "completed": len(records),
                "source_hashes_verified": True,
                "execution_source": str(out / "source"),
                "resume_driver_sha256": sha(Path(__file__)) if args.resume else None,
                "live_source_drift": [
                    rel
                    for rel, digest in hashes.items()
                    if not (ROOT / rel).exists() or sha(ROOT / rel) != digest
                ],
            },
            indent=2,
        )
        + "\n"
    )


def run_batch(args):
    from mriseqbench.evaluation import evaluate

    for row in json.loads(args.batch.read_text()):
        directory = Path(row["directory"])
        started = time.monotonic()
        case = json.loads((directory / "case.json").read_text())
        result_path = directory / "evaluation/result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result = evaluate(
            directory / "sequence.seq",
            case,
            args.source_root or ROOT,
            result_path.parent / "physical",
            time_budget_s=args.time_budget,
        )
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        record = {
            **row,
            "seconds": time.monotonic() - started,
            "result": str(result_path),
            "result_sha256": sha(result_path),
            "status": result["status"],
            "error_kind": result.get("error_kind"),
            "reason": result.get("reason"),
            "metrics": result.get("metrics", {}),
            "numerical_intervals": result.get("numerical_intervals"),
            "numerically_converged": result.get("numerically_converged"),
            "image_checks": result.get("image_checks"),
            "artifact_sha256": {
                name: sha(Path(path))
                for name, path in result.get("artifacts", {}).items()
            },
            "failed_checks": [
                c["name"] for c in result.get("checks", []) if not c["passed"]
            ],
        }
        assert sha(directory / "sequence.seq") == row["sequence_sha256"]
        assert sha(directory / "case.json") == row["case_sha256"]
        (directory / "record.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            json.dumps(
                {k: record[k] for k in ("control", "status", "seconds", "reason")}
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument("--batch", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and resume an existing frozen plan",
    )
    parser.add_argument(
        "--phase",
        choices=(
            "pilot",
            "ideal",
            "physics",
            "profiles",
            "hardware",
            "negative",
            "qualification",
            "physics_all",
            "qualification_all",
        ),
        default="pilot",
    )
    parser.add_argument("--family", action="append", choices=FAMILIES)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--initial-samples", type=int, choices=(4096, 8192, 16384, 32768, 65536)
    )
    parser.add_argument("--robust-physics", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Freeze and author the full plan before replay",
    )
    parser.add_argument(
        "--volume-samples", type=int, choices=(8192, 16384, 32768, 65536)
    )
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--batch-size", type=int, choices=range(1, 65), default=16)
    parser.add_argument("--time-budget", type=int, default=900)
    args = parser.parse_args()
    if args.batch:
        run_batch(args)
    elif args.destination:
        run(args)
    else:
        parser.error("destination is required")
