import argparse
import signal
import sys
from pathlib import Path

from .catalog import Catalog, load_yaml
from .evaluation import evaluate, normalize_verdict
from .harness.arm_runtime import DockerUnavailable
from .harness.prepare import pi_model, prompt
from .harness.runner import CampaignStop, run
from .io import write_json
from .labels import SIMULATOR_LABELS, arm_label, case_labels
from .models import MODELS, Backend, Experiment


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
    schema = commands.add_parser(
        "schema", help="write JSON schemas for editor completion"
    )
    schema.add_argument("--out", type=Path, default=Path("schemas"))
    listing = commands.add_parser("list", help="show explicitly selected suite cases")
    listing.add_argument("--suite", default="core")
    describe = commands.add_parser(
        "describe", help="render agent instructions from the resolved contract"
    )
    describe.add_argument("task", help="task label (e.g. T2.6) or manifest reference")
    describe.add_argument(
        "--hardware", help="hardware label (H1–H3) or manifest reference"
    )
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
        "run",
        help="run or resume an experiment campaign (rerun the same --out to resume)",
    )
    running.add_argument("--experiment", type=Path, required=True)
    running.add_argument(
        "--out",
        type=Path,
        help="campaign directory, new or resumed (default: runs/<id>)",
    )
    running.add_argument("--cases", default="*", help="case_id glob, e.g. for a canary")
    running.add_argument(
        "--repeats", type=int, help="override repeats (selection only)"
    )
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
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
                    f"{arm_label(arm)} ({arm.id}@{arm.version}): {arm.authoring}; PulseDesigner={arm.pulse_designer}; Unitful={arm.unitful}; simulator={SIMULATOR_LABELS.get(arm.simulator, 'none')} ({arm.simulator}); tools={arm.tool_interface}"
                )
            return 0
        if args.command == "validate":
            for experiment in catalog.documents["experiments"].values():
                agent = experiment.agent
                if (
                    agent.uses_model
                    and agent.adapter == "pi"
                    and "REPLACE_WITH" not in agent.model
                ):
                    pi_model(agent.model, experiment.api_proxy)
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
            print(
                f"{'TASK':<8}{'LEVEL':<7}{'HW':<5}{'PHANTOM':<9}{'CASE':<{width}}EVALUATION"
            )
            for case in cases:
                evaluator = case["evaluation"]
                label = (
                    evaluator["scope"]
                    if evaluator["calibrated"] or evaluator["scope"] == "smoke"
                    else "draft"
                )
                labels = case_labels(case)
                print(
                    f"{labels['task'] or 'smoke':<8}{labels['level']:<7}"
                    f"{labels['hardware']:<5}{labels['phantom'] or 'assay':<9}"
                    f"{case['case_id']:<{width}}{label}"
                )
            return 0
        if args.command == "describe":
            case = catalog.resolve(
                catalog.get("tasks", args.task), hardware_ref=args.hardware
            )
            print(prompt(case))
            return 0
        if args.command == "evaluate":
            backend = (
                Backend.model_validate(load_yaml(args.backend))
                if args.backend
                else None
            )
            case = catalog.find_case(args.case)
            args.out.mkdir(parents=True, exist_ok=False)
            report = normalize_verdict(
                evaluate(args.submission, case, catalog.root, args.out, backend), case
            )
            write_json(args.out / "result.json", report)
            print(
                f"{report['case_id']}: {report['status']} ({args.out / 'result.json'})"
            )
            return 0 if report["status"] == "passed" else 1
        experiment = Experiment.model_validate(load_yaml(args.experiment))
        # systemd stops with SIGTERM: unwind so in-flight trials record interruption.
        signal.signal(signal.SIGTERM, signal.default_int_handler)
        output, reports, complete = run(
            catalog, experiment, args.out, cases=args.cases, repeats=args.repeats
        )
        print(f"Results: {output / 'results.jsonl'} ({len(reports)} trials)")
        # 0: every selected trial is final; 1: some are resumable (rerun later).
        return 0 if complete else 1
    except KeyboardInterrupt:
        print("interrupted; rerun the same command to resume", file=sys.stderr)
        return 130
    except CampaignStop as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 3
    except DockerUnavailable as exc:
        # Retryable: the systemd unit restarts on 4 (only 2 and 3 are final).
        print(f"infrastructure unavailable: {exc}", file=sys.stderr)
        return 4
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
