"""Freeze and qualify current cardiac acquisitions; pilot results are not calibration."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def controls(root, level):
    spec = importlib.util.spec_from_file_location(
        f"cardiac_{level}", Path(root) / f"tests/fixtures/{level.lower()}_sequences.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options_for(case, module):
    level, family = case["sequence_level"], case["task"]["id"]
    cardiac = (
        case["object"]["phantom"].startswith("cardiac_")
        and case.get("image_policy") == "static"
    )
    field = case["hardware"]["field_strength_T"]
    # 0.32 ms cannot carry a 256-sample readout; 0.64 ms (2.5 us dwell) is the
    # shortest raster-valid flat top under the 80 mT/m limit.
    flat = 0.00064 if case["task"]["sequence"]["matrix"][0] >= 256 else 0.00032
    if level == "S1":
        options = {"readout_flat": flat}
        if cardiac and family == "spin_echo_t2w" and field == 3:
            options.update(water_selective_order=2, water_selective_refocusing_order=2)
        return options
    if level == "S3":
        options = {"readout_flat": flat}
        if cardiac and family == "reduced_fov":
            options.update(
                selective_rf_duration=0.0096,
                reduced_refocusing="composite",
                reduced_refocusing_duration=0.0008,
                reduced_flip_deg=90,
                preparations=1,
                repetition_time=None
                if case["task"]["sequence"].get("tr_ms")
                else 0.352,
                reduced_fat_saturation=True,
                reduced_fat_saturation_pulses=2,
                reduced_fat_saturation_duration=0.008,
            )
        return options
    options = module.physics_control_options(case)
    if family == "radial_readout" and case["task"]["sequence"].get("te_ms"):
        options["radial_te"] = case["task"]["sequence"]["te_ms"]["target"] / 1000
    if family not in {"ute_radial", "spiral_gre"}:
        options["cartesian_flat"] = flat
    if family == "grase":
        options.update(grase_offset=0.0004, ref_duration=0.0006)
    if flat > 0.00032:
        # 256 controls: blips in the readout fall ramp keep the 0.64 ms flat top
        # inside each TE; GRASE gradient echoes sit outside that longer readout.
        if family.startswith("epi"):
            options.update(epi_spacing=0.00118, overlap_blips=True)
        if family == "epi_single_shot":
            options["epi_tr"] = 0.45
        if family == "grase":
            options["grase_offset"] = 0.0013
        if family == "spiral_gre":
            options.update(curved_duration=0.016, spiral_adc_oversampling=5)
    if case["hardware"]["id"] == "low_field":
        if family.startswith("epi"):
            options["epi_spacing"] = 0.0012
        if family == "grase":
            options["grase_offset"] = 0.0006
        if family == "spiral_gre":
            options["curved_duration"] = 0.0112
        if family == "ute_radial":
            options["ute_pre_duration"] = 0.00014
    if cardiac:
        if family.startswith("epi"):
            options["water_selective_order"] = 2
        if family == "spiral_gre":
            options["water_selective_order"] = 1 if field == 0.55 else 2
        if field == 3 and family in {"grase", "propeller_tse", "tse_t2w"}:
            options.update(water_selective_order=2, water_selective_refocusing_order=2)
        if field == 3 and family == "grase":
            options["grase_spin_echo_spacing"] = 0.008 if flat > 0.00032 else 0.006
        if family == "tse_t2w" and field in {1.5, 3}:
            options["centric_tse"] = True
            if field == 1.5:
                options["ref_duration"] = 0.0006
    return options


def worker(args):
    from mriseqbench.evaluation import evaluate

    for row in json.loads(args.worker.read_text()):
        directory = Path(row["directory"])
        case = json.loads((directory / "case.json").read_text())
        start = time.monotonic()
        try:
            result = evaluate(
                directory / "sequence.seq",
                case,
                args.source_root,
                directory / "evaluation",
                time_budget_s=args.time_budget,
            )
        except Exception as exc:  # noqa: BLE001 - preserve each isolated scientific failure
            result = {"status": "evaluator_error", "reason": repr(exc)}
        result_path = directory / "result.json"
        write(result_path, result)
        record = {
            **row,
            "seconds": time.monotonic() - start,
            "status": result["status"],
            "reason": result.get("reason"),
            "numerically_converged": result.get("numerically_converged"),
            "sequence_valid": result.get("sequence_valid"),
            "image_pass": bool(result.get("image_checks"))
            and all(
                check["passed"] is True for check in result.get("image_checks", [])
            ),
            "failed_checks": [
                check for check in result.get("checks", []) if not check["passed"]
            ],
            "failed_image_checks": [
                check
                for check in result.get("image_checks", [])
                if check["passed"] is not True
            ],
            "metrics": result.get("metrics"),
            "result_sha256": digest(result_path),
            "artifact_sha256": {
                key: digest(path) for key, path in result.get("artifacts", {}).items()
            },
        }
        assert digest(directory / "case.json") == row["case_sha256"]
        assert digest(directory / "sequence.seq") == row["sequence_sha256"]
        write(directory / "record.json", record)
        print(
            json.dumps(
                {
                    key: record[key]
                    for key in ("case_id", "status", "image_pass", "seconds", "reason")
                }
            ),
            flush=True,
        )


def run(args):
    from mriseqbench.catalog import Catalog
    from mriseqbench.evaluation import bart
    from mriseqbench.evaluation.preflight import preflight
    from mriseqbench.evaluation.sar import evaluate_sar

    out = args.destination.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / "source"
    frozen.mkdir()
    for name in ("src", "benchmark"):
        shutil.copytree(
            ROOT / name, frozen / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    for name in ("Project.toml", "Manifest.toml", "pyproject.toml", "uv.lock"):
        shutil.copyfile(ROOT / name, frozen / name)
    fixture = frozen / "tests/fixtures"
    fixture.mkdir(parents=True)
    for name in (
        "calibrate_cardiac.py",
        "s1_sequences.py",
        "s2_sequences.py",
        "s3_sequences.py",
        "score_cardiac_calibration.py",
        "score_cardiac_receiver_negatives.py",
        "audit_cardiac_water_excitation.py",
    ):
        shutil.copyfile(ROOT / "tests/fixtures" / name, fixture / name)
    independent = frozen / "tests/evaluation"
    independent.mkdir()
    for name in ("test_bart_reconstruction_gold.py", "test_cardiac_rf_reference.py"):
        shutil.copyfile(ROOT / "tests/evaluation" / name, independent / name)
    binary = frozen / "tools/bart"
    binary.parent.mkdir()
    shutil.copy2(bart.executable(), binary)
    write(out / "bart-provenance.json", bart.provenance())
    hashes = {
        str(path.relative_to(frozen)): digest(path)
        for path in frozen.rglob("*")
        if path.is_file()
    }
    write(out / "source-hashes.json", hashes)
    catalog = Catalog(frozen)
    authors = {level: controls(frozen, level) for level in ("S1", "S2", "S3")}
    scorer_spec = importlib.util.spec_from_file_location(
        "frozen_cardiac_scorer", fixture / "score_cardiac_calibration.py"
    )
    scorer = importlib.util.module_from_spec(scorer_spec)
    scorer_spec.loader.exec_module(scorer)
    cases = [
        case
        for case in catalog.cases(catalog.get("suites", "full"))
        if case["sequence_level"] in authors
    ]
    priority = {"gre_t1w": 0, "epi_single_shot": 1, "epi_multishot": 2}
    cases.sort(
        key=lambda case: (
            len(case["task"]["sequence"]["matrix"]),
            priority.get(case["task"]["id"], 3),
            case["task"]["id"],
            case["hardware"]["field_strength_T"],
        )
    )
    plan = []
    for case in cases:
        if args.family and case["task"]["id"] not in args.family:
            continue
        if args.hardware and case["hardware"]["id"] not in args.hardware:
            continue
        case["evaluation"].update(
            spatial_samples=args.samples,
            spatial_replicates=args.replicates,
            spatial_seed=args.seed,
        )
        case["evaluation"]["spatial_max_samples"] = max(
            args.samples, case["evaluation"]["spatial_max_samples"]
        )
        directory = out / case["case_id"]
        directory.mkdir()
        write(directory / "case.json", case)
        module = authors[case["sequence_level"]]
        options = options_for(case, module)
        if args.spin_echo_water and case["task"]["id"] == "spin_echo_t2w":
            options.update(water_selective_order=2, water_selective_refocusing_order=2)
        if args.water_excitation and case["task"]["id"] in {
            "epi_single_shot",
            "epi_multishot",
        }:
            options["water_selective_order"] = 2
        if args.water_excitation and case["task"]["id"] == "spiral_gre":
            options["water_selective_order"] = (
                1 if case["hardware"]["id"] == "low_field" else 2
            )
        if (
            args.grase_water
            and case["task"]["id"] == "grase"
            and case["hardware"]["field_strength_T"] == 3
        ):
            options["water_selective_order"] = 2
            if args.grase_water == "both":
                options["water_selective_refocusing_order"] = 2
        if (
            args.grase_spin_echo_spacing_ms is not None
            and case["task"]["id"] == "grase"
            and case["hardware"]["field_strength_T"] == 3
        ):
            options["grase_spin_echo_spacing"] = args.grase_spin_echo_spacing_ms / 1000
        if (
            args.propeller_water
            and case["task"]["id"] == "propeller_tse"
            and case["hardware"]["field_strength_T"] == 3
        ):
            options.update(water_selective_order=2, water_selective_refocusing_order=2)
        if case["task"]["id"] == "tse_t2w":
            if args.tse_centric:
                options["centric_tse"] = True
            if args.tse_ref_duration_ms is not None:
                options["ref_duration"] = args.tse_ref_duration_ms / 1000
            if args.tse_water:
                options.update(
                    water_selective_order=2, water_selective_refocusing_order=2
                )
        try:
            mutation = (
                ("missing_line" if case["sequence_level"] == "S1" else "wrong_fov")
                if args.structural_negatives
                else None
            )
            module.make_sequence(case, mutation, **options).write(
                str(directory / "sequence.seq")
            )
            if args.receiver_negatives:
                original = directory / "sequence-unrotated.seq"
                (directory / "sequence.seq").rename(original)
                scorer.rotate_adc(original, directory / "sequence.seq")
                mutation = "global_receiver_rotation_90deg"
            inspector = importlib.import_module(
                f"mriseqbench.evaluation.{case['sequence_level'].lower()}_sequence"
            )
            checks = preflight(directory / "sequence.seq", case) + evaluate_sar(
                directory / "sequence.seq", case["hardware"]
            )
            checks += inspector.inspect_sequence(
                directory / "sequence.seq", case
            ).checks
            write(directory / "export-audit.json", checks)
            row = {
                "case_id": case["case_id"],
                "directory": str(directory),
                "options": options,
                "mutation": mutation,
                "case_sha256": digest(directory / "case.json"),
                "sequence_sha256": digest(directory / "sequence.seq"),
                "legal": all(check["passed"] for check in checks),
            }
            if not row["legal"]:
                print(
                    json.dumps(
                        {
                            "illegal": case["case_id"],
                            "checks": [c for c in checks if not c["passed"]],
                        }
                    ),
                    flush=True,
                )
            plan.append(row)
        except Exception as exc:  # noqa: BLE001 - preserve each isolated scientific failure
            write(
                directory / "author-error.json",
                {"reason": repr(exc), "options": options},
            )
            print(
                json.dumps({"author_error": case["case_id"], "reason": repr(exc)}),
                flush=True,
            )
    write(out / "plan.json", plan)
    write(
        out / "campaign.json",
        {
            "role": "structural_negatives"
            if args.structural_negatives
            else "receiver_negatives"
            if args.receiver_negatives
            else args.role,
            "samples": args.samples,
            "replicates": args.replicates,
            "seed": args.seed,
            "eligible_case_count": len(cases),
            "case_count": sum(
                not args.family or case["task"]["id"] in args.family
                for case in cases
                if not args.hardware or case["hardware"]["id"] in args.hardware
            ),
            "wall_limit_s": args.wall_limit,
            "bart_binary_sha256": digest(binary),
            "authored": len(plan),
            "plan_sha256": digest(out / "plan.json"),
        },
    )
    if args.plan_only:
        return
    batches = [plan[index :: args.workers] for index in range(args.workers)]

    def launch(index, batch):
        batch_path = out / f"batch-{index}.json"
        write(batch_path, batch)
        if not args.wall_limit:
            # Official campaigns retain one Python/Julia process per batch.
            # The frozen cases and independent per-case records are unchanged.
            with (out / f"worker-{index}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(fixture / "calibrate_cardiac.py"),
                        "--worker",
                        str(batch_path),
                        "--source-root",
                        str(frozen),
                        "--time-budget",
                        str(args.time_budget),
                    ],
                    cwd=ROOT,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(frozen / "src"),
                        "MRISEQBENCH_BART": str(binary),
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                    },
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            return
        for row in batch:
            directory = Path(row["directory"])
            single = directory / "worker-input.json"
            write(single, [row])
            with (directory / "worker.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(fixture / "calibrate_cardiac.py"),
                        "--worker",
                        str(single),
                        "--source-root",
                        str(frozen),
                        "--time-budget",
                        str(args.time_budget),
                    ],
                    cwd=ROOT,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(frozen / "src"),
                        "MRISEQBENCH_BART": str(binary),
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                    },
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    status = process.wait(timeout=args.wall_limit or None)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    result = {
                        "status": "numerically_unresolved",
                        "reason": "External pilot wall limit reached; partial numerical evidence retained",
                        "wall_limit_s": args.wall_limit,
                    }
                    write(directory / "result.json", result)
                    write(
                        directory / "record.json",
                        {
                            **row,
                            **result,
                            "image_pass": False,
                            "result_sha256": digest(directory / "result.json"),
                        },
                    )
                    continue
                if status:
                    raise RuntimeError(
                        f"Case worker {row['case_id']} exited {status}; see {directory / 'worker.log'}"
                    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed(
            [pool.submit(launch, i, batch) for i, batch in enumerate(batches)]
        ):
            future.result()
    records = [
        json.loads((Path(row["directory"]) / "record.json").read_text()) for row in plan
    ]
    write(out / "summary.json", {"role": args.role, "records": records})
    assert all(digest(frozen / rel) == value for rel, value in hashes.items())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--family", action="append")
    parser.add_argument("--hardware", action="append")
    parser.add_argument("--role", choices=["pilot", "qualification"], default="pilot")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--time-budget", type=float, default=180)
    parser.add_argument(
        "--wall-limit",
        type=float,
        default=0,
        help="External per-case wall limit in seconds; zero disables it",
    )
    parser.add_argument("--plan-only", action="store_true")
    negatives = parser.add_mutually_exclusive_group()
    negatives.add_argument("--structural-negatives", action="store_true")
    negatives.add_argument("--receiver-negatives", action="store_true")
    parser.add_argument("--water-excitation", action="store_true")
    parser.add_argument("--grase-water", choices=["excitation", "both"])
    parser.add_argument("--grase-spin-echo-spacing-ms", type=float)
    parser.add_argument("--propeller-water", action="store_true")
    parser.add_argument("--tse-centric", action="store_true")
    parser.add_argument("--tse-ref-duration-ms", type=float)
    parser.add_argument("--tse-water", action="store_true")
    parser.add_argument("--spin-echo-water", action="store_true")
    args = parser.parse_args()
    if args.role == "qualification" and (args.samples < 32768 or args.replicates < 8):
        parser.error(
            "Official qualification requires at least 32768 samples and eight replicates"
        )
    if args.worker:
        worker(args)
    else:
        run(args)
