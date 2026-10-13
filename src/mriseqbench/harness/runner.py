"""Prepared, sandboxed case/repeat runs with separate trusted evaluation."""

import json
import shutil
import uuid
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from ..io import write_json
from ..phantoms import materialize
from ..process import execute, expand
from .arm_mailbox import Mailbox
from .arm_runtime import Simulator, arm_environment, launch_arm, resolve_runtime
from .feedback import FeedbackServer
from .grading import grade_frozen
from .prepare import prepare, prepare_pi
from .proxy import ApiProxy
from .sandbox import environment, launch


def final_evaluation(source, case, experiment, root, workspace):
    judge_dir = workspace / "evaluation"
    if (
        not source.is_file()
        or source.is_symlink()
        or source.stat().st_size > experiment.feedback.max_submission_bytes
    ):
        return {
            "case_id": case["case_id"],
            "status": "failed",
            "checks": [],
            "metrics": {},
            "reason": "Final submission is missing, a symlink, or oversized",
        }
    snapshot = judge_dir / "sequence.seq"
    shutil.copyfile(source, snapshot)
    write_json(judge_dir / "case.json", case)
    return grade_frozen(
        snapshot,
        case,
        root,
        judge_dir,
        experiment.feedback.timeout_s,
        execute,
        backend_path=workspace / "control/backend.json" if experiment.backend else None,
    )


def run_one(
    catalog, experiment, case, workspace, repeat, phantom_dir=None, *, runtime=None
):
    if experiment.arm is not None or experiment.arm_runtime is not None:
        experiment = type(experiment).model_validate(experiment.model_dump(mode="json"))
    arm = catalog.get("arms", experiment.arm) if experiment.arm else None
    if arm and runtime is None:
        runtime = resolve_runtime(experiment, arm)
    if arm and runtime["arm"] != arm.model_dump(mode="json"):
        raise ValueError("resolved runtime does not match the selected arm")
    agent_dir, judge_dir = workspace / "agent", workspace / "evaluation"
    agent_dir.mkdir(parents=True, mode=0o700)
    judge_dir.mkdir()
    with ExitStack() as stack:
        server = stack.enter_context(
            FeedbackServer(
                case,
                experiment.feedback,
                catalog.root,
                workspace / "control",
                experiment.backend,
                max_submissions=experiment.max_submissions,
            )
        )
        proxy = (
            stack.enter_context(
                ApiProxy(
                    experiment.api_proxy,
                    workspace / "api.jsonl",
                    model=experiment.agent.model or None,
                )
            )
            if experiment.api_proxy
            else None
        )
        web_access = case["task"].get("prompt_mode") == "research" and arm is None
        prompt_path, case_path, submission = prepare(
            agent_dir, case, experiment, server, phantom_dir, arm=arm
        )
        values = {
            "prompt": prompt_path,
            "prompt_text": prompt_path.read_text(),
            "case": case_path,
            "submission": submission,
            "model": experiment.agent.model,
            "api_base_url": ("http://127.0.0.1:8765/api" if arm else proxy.base_url)
            if proxy
            else "",
            "api_token": proxy.token if proxy else "",
        }
        if arm:
            values["python"] = "python3"
        command = expand(experiment.agent.command, catalog.root, agent_dir, **values)
        executable = shutil.which(command[0]) if not arm else None
        if executable:
            command[0] = executable
        env = (arm_environment if arm else environment)(
            experiment.agent, catalog.root, agent_dir, **values
        )
        if experiment.agent.adapter == "pi":
            env.update(
                prepare_pi(agent_dir, values["api_base_url"], offline=not web_access)
            )
            env["OPENROUTER_API_KEY"] = proxy.token
        if not arm:
            env["PATH"] = str(Path(command[0]).parent) + ":" + env["PATH"]
        ports = [server.port, *([proxy.port] if proxy else [])]
        simulator = None
        if arm:
            if arm.simulator != "none":
                frozen_inputs = workspace / "control/simulation_inputs"
                if (agent_dir / "inputs").exists():
                    shutil.copytree(agent_dir / "inputs", frozen_inputs)
                else:
                    frozen_inputs.mkdir()
                simulator = Simulator(
                    runtime,
                    experiment.arm_runtime,
                    agent_dir,
                    workspace / "control",
                    json.loads(case_path.read_text()),
                )
            stack.enter_context(
                Mailbox(workspace / "control/transport", server, proxy, simulator)
            )
        launcher = (
            launch_arm(runtime, command, agent_dir, workspace / "control", env)
            if arm
            else launch(
                experiment.agent.sandbox,
                command,
                catalog.root,
                agent_dir,
                workspace / "control",
                ports,
                internet=web_access,
            )
        )
        with launcher as (isolated_command, isolation):
            execution = execute(
                isolated_command,
                agent_dir,
                experiment.agent.timeout_s,
                workspace / "agent",
                # The trusted Docker client needs the host's daemon/context.
                # Container variables were already passed explicitly via --env;
                # Docker does not inherit the client's environment into it.
                env=None if arm else env,
                stop_when=server.submitted,
                timeout_credit=server.grading_credit,
            )
        if proxy:
            proxy.revoke()
        if server.submitted.is_set():
            source = server.submission_path
        elif (
            execution["status"] == "completed" and experiment.submission_mode == "file"
        ):
            source = submission
            if (
                experiment.max_submissions > 1
                and not server.submissions_started
                and submission.is_file()
                and not submission.is_symlink()
                and submission.stat().st_size
                <= experiment.feedback.max_submission_bytes
            ):
                server.handle("submit", server.token, submission.read_bytes())
        else:
            source = None
        # Drain any accepted attempt whose evaluator was active at the deadline.
        with server.lock:
            attempts = list(server.attempts)
            if server.evaluation_failure is not None:
                raise server.evaluation_failure
            if server.submitted.is_set():
                source = server.submission_path
        server.closed.set()
        report = {
            "case_id": case["case_id"],
            "status": "agent_not_submitted"
            if execution["status"] == "completed"
            else f"agent_{execution['status']}",
            "checks": [],
            "metrics": {},
        }
        if attempts:
            # Reuse the immutable attempt verdict; never regrade changed agent files.
            report = dict(attempts[-1]["evaluation"])
            shutil.copyfile(server.submission_path, judge_dir / "sequence.seq")
            write_json(judge_dir / "case.json", case)
            write_json(judge_dir / "result.json", report)
        elif source is not None:
            report = final_evaluation(source, case, experiment, catalog.root, workspace)
        report.update(
            repeat=repeat,
            prompt_mode=case["task"].get("prompt_mode", "detailed"),
            web_access=web_access,
            task_family=case["task"]["id"].removesuffix("_research"),
            agent=execution,
            isolation=isolation,
            feedback=server.stats(),
            api_requests=proxy.requests if proxy else 0,
            workspace=str(workspace),
            submissions=attempts,
            first_submission_status=attempts[0]["evaluation"]["status"]
            if attempts
            else None,
            passed_on_submission=next(
                (
                    a["attempt"]
                    for a in attempts
                    if a["evaluation"]["status"] in ("passed", "smoke_passed")
                ),
                None,
            ),
        )
        if arm:
            report.update(
                arm=experiment.arm,
                authoring=arm.authoring,
                simulator=arm.simulator,
                simulation_calls=simulator.calls if simulator else [],
            )
    return report


def run(catalog, experiment, output=None):
    # Validate before creating output, materializing phantoms, or starting agents.
    if experiment.arm is not None or experiment.arm_runtime is not None:
        experiment = type(experiment).model_validate(experiment.model_dump(mode="json"))
    arm = catalog.get("arms", experiment.arm) if experiment.arm else None
    runtime = resolve_runtime(experiment, arm) if arm else None
    cases = catalog.cases(catalog.get("suites", experiment.suite))
    if experiment.agent.adapter == "pi" and experiment.api_proxy is None:
        raise ValueError("the pi adapter requires an API proxy")
    output = (
        Path(output).resolve()
        if output
        else catalog.root
        / "runs"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    )
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_json(output / "experiment.json", experiment.model_dump(mode="json"))
    if runtime:
        write_json(output / "arm_runtime.json", runtime)
    reports = []
    for case in cases:
        case["seed"] = experiment.seed
        phantom_dir = None
        if case["evaluation"]["scope"] == "benchmark" or (
            arm and arm.simulator != "none"
        ):
            phantom_dir = output / "phantoms" / case["case_id"]
            materialize(case, catalog.root, phantom_dir)
        for repeat in range(experiment.repeats):
            workspace = output / f"{case['case_id']}--r{repeat}"
            report = run_one(
                catalog,
                experiment,
                case,
                workspace,
                repeat,
                phantom_dir,
                runtime=runtime,
            )
            write_json(workspace / "result.json", report)
            reports.append(report)
            with (output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(report, allow_nan=False) + "\n")
            print(f"{case['case_id']} repeat={repeat}: {report['status']}", flush=True)
    return output, reports
