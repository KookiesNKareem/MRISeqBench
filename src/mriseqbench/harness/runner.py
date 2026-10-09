"""Prepared, sandboxed case/repeat runs with separate trusted evaluation."""

import json
import shutil
import sys
import uuid
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from ..io import write_json
from ..phantoms import materialize
from ..process import execute, expand
from .feedback import FeedbackServer
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
    command = [
        sys.executable,
        "-m",
        "mriseqbench.evaluation.worker",
        "--mode",
        "evaluation",
        "--case",
        str(judge_dir / "case.json"),
        "--submission",
        str(snapshot),
        "--root",
        str(root),
        "--workspace",
        str(judge_dir),
    ]
    if experiment.backend:
        command += ["--backend", str(workspace / "control/backend.json")]
    execution = execute(
        command, judge_dir, experiment.feedback.timeout_s, judge_dir / "worker"
    )
    if execution["status"] == "completed":
        try:
            return json.loads((judge_dir / "result.json").read_text())
        except (OSError, ValueError):
            pass
    return {
        "case_id": case["case_id"],
        "status": "evaluator_error",
        "checks": [],
        "metrics": {},
        "reason": f"final worker {execution['status']} or invalid output",
    }


def run_one(catalog, experiment, case, workspace, repeat, phantom_dir=None):
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
        prompt_path, case_path, submission = prepare(
            agent_dir, case, experiment, server, phantom_dir
        )
        values = {
            "prompt": prompt_path,
            "prompt_text": prompt_path.read_text(),
            "case": case_path,
            "submission": submission,
            "model": experiment.agent.model,
            "api_base_url": proxy.base_url if proxy else "",
            "api_token": proxy.token if proxy else "",
        }
        command = expand(experiment.agent.command, catalog.root, agent_dir, **values)
        executable = shutil.which(command[0])
        if executable:
            command[0] = executable
        env = environment(experiment.agent, catalog.root, agent_dir, **values)
        if experiment.agent.adapter == "pi":
            env.update(prepare_pi(agent_dir, proxy.base_url))
            env["OPENROUTER_API_KEY"] = proxy.token
        env["PATH"] = str(Path(command[0]).parent) + ":" + env["PATH"]
        ports = [server.port, *([proxy.port] if proxy else [])]
        with launch(
            experiment.agent.sandbox,
            command,
            catalog.root,
            agent_dir,
            workspace / "control",
            ports,
        ) as (isolated_command, isolation):
            execution = execute(
                isolated_command,
                agent_dir,
                experiment.agent.timeout_s,
                workspace / "agent",
                env=env,
                stop_when=server.submitted,
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
        server.closed.set()
        # Drain any accepted attempt whose evaluator was active at the deadline.
        with server.lock:
            attempts = list(server.attempts)
            if server.submitted.is_set():
                source = server.submission_path
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
    return report


def run(catalog, experiment, output=None):
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
    reports = []
    for case in cases:
        case["seed"] = experiment.seed
        phantom_dir = None
        if case["evaluation"]["scope"] == "benchmark":
            phantom_dir = output / "phantoms" / case["case_id"]
            materialize(case, catalog.root, phantom_dir)
        for repeat in range(experiment.repeats):
            workspace = output / f"{case['case_id']}--r{repeat}"
            report = run_one(catalog, experiment, case, workspace, repeat, phantom_dir)
            write_json(workspace / "result.json", report)
            reports.append(report)
            with (output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(report, allow_nan=False) + "\n")
            print(f"{case['case_id']} repeat={repeat}: {report['status']}", flush=True)
    return output, reports
