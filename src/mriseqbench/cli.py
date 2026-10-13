import argparse
import json
import sys
from pathlib import Path

from .catalog import Catalog, load_yaml
from .evaluation import evaluate
from .harness.prepare import prompt
from .harness.runner import run
from .io import write_json
from .models import MODELS, SEQUENCE_CATEGORIES, ArmRuntime, Backend, Experiment


def parser():
    p = argparse.ArgumentParser(prog="mriseqbench")
    p.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="catalog root (default: current directory)",
    )
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate every manifest and reference")
    commands.add_parser("arms", help="list versioned authoring/simulator arms")
    commands.add_parser(
        "axes", help="show sequence capabilities, cardiac anatomy and hardware profiles"
    )
    doctor = commands.add_parser(
        "doctor", help="probe native sandbox access restrictions"
    )
    doctor.add_argument(
        "--sandbox", choices=["auto", "macos", "bubblewrap"], default="auto"
    )
    doctor.add_argument("--arm", help="audit a restricted arm's local Docker images")
    doctor.add_argument("--author-image")
    doctor.add_argument("--simulator-image")
    schema = commands.add_parser(
        "schema", help="write JSON schemas for editor completion"
    )
    schema.add_argument("--out", type=Path, default=Path("schemas"))
    listing = commands.add_parser("list", help="show explicitly selected suite cases")
    listing.add_argument("--suite", default="core")
    describe = commands.add_parser(
        "describe", help="render agent instructions from the resolved contract"
    )
    describe.add_argument("task")
    describe.add_argument("--hardware", help="override the task's hardware reference")
    materialization = commands.add_parser(
        "materialize", help="build a KomaMRI phantom and companion fields"
    )
    materialization.add_argument("--case", required=True)
    materialization.add_argument("--out", type=Path, required=True)
    materialization.add_argument("--seed", type=int, default=42)
    materialization.add_argument("--timeout-s", type=float, default=600)
    evaluation = commands.add_parser(
        "evaluate", help="evaluate a submission independently of an agent"
    )
    evaluation.add_argument("submission", type=Path)
    evaluation.add_argument("--case", required=True)
    evaluation.add_argument(
        "--backend", type=Path, help="YAML file containing command and timeout_s"
    )
    evaluation.add_argument(
        "--out", type=Path, required=True, help="new evaluation directory"
    )
    running = commands.add_parser(
        "run", help="run an experiment's agent and evaluator separately"
    )
    running.add_argument("--experiment", type=Path, required=True)
    running.add_argument("--arm", help="override the experiment's arm reference")
    running.add_argument(
        "--author-image", help="local Docker authoring image (required with --arm)"
    )
    running.add_argument(
        "--simulator-image", help="local fixed-adapter simulator image"
    )
    running.add_argument(
        "--out", type=Path, help="new run directory (default: runs/<unique id>)"
    )
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            if not args.arm and (args.author_image or args.simulator_image):
                raise ValueError("doctor image options require --arm")
            if args.arm:
                from .harness.arm_runtime import resolve_runtime

                arm = Catalog(args.root).get("arms", args.arm)
                if not args.author_image:
                    raise ValueError("doctor --arm requires --author-image")
                experiment = Experiment(
                    schema_version=1,
                    id="arm_doctor",
                    version=1,
                    suite="smoke@1",
                    arm=f"{arm.id}@{arm.version}",
                    arm_runtime=ArmRuntime(
                        author_image=args.author_image,
                        simulator_image=args.simulator_image,
                    ),
                    agent={"command": ["true"], "sandbox": {"backend": "docker"}},
                )
                print(json.dumps(resolve_runtime(experiment, arm), indent=2))
                return 0
            from .harness.doctor import doctor

            result = doctor(args.root.resolve(), args.sandbox)
            print(json.dumps(result, indent=2))
            return 0 if result["ok"] else 1
        if args.command == "schema":
            args.out.mkdir(parents=True, exist_ok=True)
            for kind, model in MODELS.items():
                write_json(args.out / f"{kind}.schema.json", model.model_json_schema())
            print(f"Wrote {len(MODELS)} schemas to {args.out}")
            return 0
        catalog = Catalog(args.root)
        if args.command == "arms":
            for arm in catalog.documents["arms"].values():
                print(
                    f"{arm.id}@{arm.version}: {arm.authoring}; PulseDesigner={arm.pulse_designer}; Unitful={arm.unitful}; simulator={arm.simulator}"
                )
            return 0
        if args.command == "axes":
            for level, capabilities in SEQUENCE_CATEGORIES.items():
                active = {
                    capability
                    for task in catalog.documents["tasks"].values()
                    if task.sequence.level == level
                    for capability in task.sequence.capabilities
                }
                visible = [
                    capability for capability in capabilities if capability in active
                ]
                print(f"{level}: {', '.join(visible)}")
            print("\nCARDIAC ANATOMY (no motion)")
            for ref in ("cardiac_2d", "cardiac_3d"):
                doc = catalog.get("objects", ref)
                print(f"{doc.id}@{doc.version}: {doc.description}")
            print(
                "Scanner-strength-specific cardiac T1/T2; static off-resonance and fat chemical shift follow the scanner field."
            )
            print("\nHARDWARE (per-axis gradient limits)")
            for hw in catalog.documents["hardware"].values():
                print(
                    f"{hw.id}@{hw.version}: {hw.field_strength_T:g} T, "
                    f"{hw.max_gradient_mT_per_m:g} mT/m, "
                    f"{hw.max_slew_T_per_m_per_s:g} T/m/s"
                )
            return 0
        if args.command == "validate":
            print(
                f"Validated {sum(len(v) for v in catalog.documents.values())} manifests"
            )
            used_evaluators = {
                task.evaluation for task in catalog.documents["tasks"].values()
            }
            for evaluator in catalog.documents["evaluators"].values():
                if (
                    evaluator.scope == "benchmark"
                    and not evaluator.calibrated
                    and f"{evaluator.id}@{evaluator.version}" in used_evaluators
                ):
                    print(
                        f"DRAFT: {evaluator.id}@{evaluator.version} has uncalibrated thresholds"
                    )
            return 0
        if args.command == "list":
            cases = catalog.cases(catalog.get("suites", args.suite))
            width = max(len(case["case_id"]) for case in cases) + 2
            print(f"{'CASE':<{width}}SEQUENCE  EVALUATION")
            for case in cases:
                evaluator = case["evaluation"]
                label = (
                    evaluator["scope"]
                    if evaluator["calibrated"] or evaluator["scope"] == "smoke"
                    else "draft"
                )
                print(f"{case['case_id']:<{width}}{case['sequence_level']:<10}{label}")
            return 0
        if args.command == "describe":
            case = catalog.resolve(
                catalog.get("tasks", args.task), hardware_ref=args.hardware
            )
            print(prompt(case))
            return 0
        if args.command == "materialize":
            from .phantoms import materialize

            if not 0 <= args.seed < 2**32 or args.timeout_s <= 0:
                raise ValueError("seed must be uint32 and timeout must be positive")
            case = catalog.find_case(args.case)
            case["seed"] = args.seed
            materialize(case, catalog.root, args.out, args.timeout_s)
            print(f"KomaMRI phantom: {args.out / 'phantom.phantom'}")
            return 0
        if args.command == "evaluate":
            backend = (
                Backend.model_validate(load_yaml(args.backend))
                if args.backend
                else None
            )
            case = catalog.find_case(args.case)
            args.out.mkdir(parents=True, exist_ok=False)
            report = evaluate(args.submission, case, catalog.root, args.out, backend)
            write_json(args.out / "result.json", report)
            print(
                f"{report['case_id']}: {report['status']} ({args.out / 'result.json'})"
            )
            return 0 if report["status"] in ("passed", "smoke_passed") else 1
        data = load_yaml(args.experiment)
        if args.arm:
            selected_arm = catalog.get("arms", args.arm)
            data["arm"] = f"{selected_arm.id}@{selected_arm.version}"
        if args.author_image:
            data["arm_runtime"] = ArmRuntime(
                author_image=args.author_image, simulator_image=args.simulator_image
            ).model_dump(mode="json")
        elif args.simulator_image:
            raise ValueError("--simulator-image requires --author-image")
        experiment = Experiment.model_validate(data)
        output, reports = run(catalog, experiment, args.out)
        print(f"Results: {output / 'results.jsonl'}")
        return (
            0 if all(r["status"] in ("passed", "smoke_passed") for r in reports) else 1
        )
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
