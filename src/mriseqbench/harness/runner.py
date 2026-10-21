"""Resumable campaigns of sandboxed case/repeat trials with trusted evaluation.

Trial state lives only in workspace files, so rerunning a campaign resumes it:
`result.json` is final; frozen but ungraded submission bytes are regraded
without rerunning the agent; other interrupted attempts get a fresh sibling
workspace (`<case>--r<k>.retry<n>`), freely when they cost nothing and at most
`retries_after_spend` times otherwise.
"""

import copy
import fcntl
import fnmatch
import hashlib
import json
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from ..io import write_json
from ..phantoms import materialize
from ..process import execute, expand
from . import arm_runtime
from .accounting import read_api, summarize_api, summarize_trials, utc_now
from .arm_mailbox import Mailbox
from .arm_runtime import Simulator, arm_environment, launch_arm, resolve_runtime
from .feedback import FeedbackServer
from .grading import grade_frozen
from .prepare import prepare, prepare_pi
from .proxy import ApiProxy, Budget
from .sandbox import environment, launch


class ModelAPIError(RuntimeError):
    """Inference infrastructure failed before a candidate could be scored."""


class CampaignStop(RuntimeError):
    """Spend cap or provider account exhausted; unfinished trials stay resumable."""


def pi_session_settings(workspace, configured_effort):
    levels = []
    for path in sorted((Path(workspace) / "transcript").glob("*.jsonl")):
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if (
                    isinstance(row, dict)
                    and row.get("type") == "thinking_level_change"
                    and row.get("thinkingLevel")
                    in {"off", "minimal", "low", "medium", "high", "xhigh", "max"}
                ):
                    levels.append(row["thinkingLevel"])
    return {
        "requested_thinking_level": "off"
        if configured_effort == "none"
        else configured_effort,
        "session_reported_thinking_levels": list(dict.fromkeys(levels)),
        "host_enforced_reasoning_effort": configured_effort,
    }


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
    catalog,
    experiment,
    case,
    workspace,
    repeat,
    phantom_dir=None,
    *,
    runtime=None,
    grading_slots=None,
    abort_event=None,
    budget=None,
):
    """Persist accounting even when setup or trusted evaluation interrupts a trial."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    if (workspace / "accounting.json").exists():
        raise FileExistsError(f"Trial accounting already exists: {workspace}")
    started = time.monotonic()
    accounting = {
        "schema_version": 1,
        "started_at": utc_now(),
        "status": "running",
        "setup_wall_s": None,
        "agent_wall_s": None,
        "agent_charged_wall_s": None,
        "agent_excluded_grading_wall_s": None,
        "julia_setup_wall_s": None,
        "grading_wall_s": None,
        "post_agent_evaluation_wall_s": 0.0,
    }
    write_json(workspace / "accounting.json", accounting)
    report = None
    try:
        report = _run_one(
            catalog,
            experiment,
            case,
            workspace,
            repeat,
            phantom_dir,
            runtime=runtime,
            accounting=accounting,
            trial_started=started,
            grading_slots=grading_slots,
            abort_event=abort_event,
            budget=budget,
        )
        accounting["status"] = "finished"
    except BaseException as exc:
        accounting.update(status="interrupted", error_type=type(exc).__name__)
        raise
    finally:
        usage = summarize_api(
            workspace / "api.jsonl", configured=experiment.api_proxy is not None
        )
        accounting.update(
            finished_at=utc_now(), wall_s=time.monotonic() - started, usage=usage
        )
        write_json(workspace / "accounting.json", accounting)
    report["accounting"] = accounting
    write_json(workspace / "result.json", report)
    return report


def _run_one(
    catalog,
    experiment,
    case,
    workspace,
    repeat,
    phantom_dir=None,
    *,
    runtime=None,
    accounting,
    trial_started,
    grading_slots=None,
    abort_event=None,
    budget=None,
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
                grading_slots=grading_slots,
            )
        )
        proxy = (
            stack.enter_context(
                ApiProxy(
                    experiment.api_proxy,
                    workspace / "api.jsonl",
                    model=experiment.agent.model or None,
                    stop_when=server.submitted,
                    budget=budget,
                    pause=server.exclude,
                )
            )
            if experiment.api_proxy
            else None
        )
        prompt_path, case_path, submission = prepare(
            agent_dir, case, experiment, server, phantom_dir, arm=arm
        )
        values = {
            "prompt": prompt_path,
            "prompt_text": prompt_path.read_text(),
            "case": case_path,
            "submission": submission,
            "model": experiment.agent.model,
            "thinking_level": "off"
            if experiment.api_proxy and experiment.api_proxy.reasoning_effort == "none"
            else experiment.api_proxy.reasoning_effort
            if experiment.api_proxy
            else "off",
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
            if not arm:
                actual = subprocess.run(
                    [command[0], "--version"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                ).stdout.strip()
                if actual != experiment.agent.pi_version:
                    raise ValueError(
                        f"Pi version mismatch: expected {experiment.agent.pi_version}, found {actual}"
                    )
            env.update(prepare_pi(agent_dir, values["api_base_url"], experiment.agent))
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
        if proxy:
            # Cancel API streams before Mailbox waits for its workers, including
            # when setup/agent execution raises instead of returning normally.
            stack.callback(proxy.revoke)
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
            )
        )
        with launcher as (isolated_command, isolation):
            agent_started = time.monotonic()
            credit_started = server.grading_credit()
            server.start_agent_clock(experiment.agent.timeout_s)
            accounting["setup_wall_s"] = agent_started - trial_started
            accounting["julia_setup_wall_s"] = isolation.get("julia_session", {}).get(
                "setup_elapsed_s"
            )
            try:

                class StopAgent:
                    def is_set(self):
                        if abort_event is not None and abort_event.is_set():
                            server.closed.set()
                            return True
                        return server.submitted.is_set() or bool(
                            proxy and proxy.exhausted
                        )

                execution = execute(
                    isolated_command,
                    agent_dir,
                    experiment.agent.timeout_s,
                    workspace / "agent",
                    # The trusted Docker client needs the host's daemon/context.
                    # Container variables were already passed explicitly via --env;
                    # Docker does not inherit the client's environment into it.
                    env=None if arm else env,
                    stop_when=StopAgent(),
                    timeout_credit=server.grading_credit,
                )
                if abort_event is not None and abort_event.is_set():
                    raise RuntimeError("Campaign aborted; trial remains unscored")
            finally:
                elapsed = time.monotonic() - agent_started
                excluded = max(0.0, server.grading_credit() - credit_started)
                accounting.update(
                    agent_wall_s=elapsed,
                    agent_excluded_grading_wall_s=min(elapsed, excluded),
                    agent_charged_wall_s=max(0.0, elapsed - excluded),
                    grading_wall_s=server.grading_credit(),
                )
        if proxy:
            proxy.revoke()
            if proxy.exhausted in ("campaign", "account"):
                raise CampaignStop(f"Model spend stopped: {proxy.exhausted} exhausted")
            if proxy.policy_violations:
                raise RuntimeError(
                    f"Inference policy violation: {proxy.policy_violations}"
                )
        if server.submitted.is_set():
            source = server.submission_path
        elif (
            execution["status"] == "completed" and experiment.submission_mode == "file"
        ):
            source = submission
            if (
                not server.submissions_started
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
            "status": "agent_trial_budget_exhausted"
            if proxy and proxy.exhausted
            else "agent_not_submitted"
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
            grading_started = time.monotonic()
            try:
                with grading_slots or threading.BoundedSemaphore(1):
                    report = final_evaluation(
                        source, case, experiment, catalog.root, workspace
                    )
            finally:
                accounting["post_agent_evaluation_wall_s"] = (
                    time.monotonic() - grading_started
                )
                accounting["grading_wall_s"] = (
                    server.grading_credit() + accounting["post_agent_evaluation_wall_s"]
                )
        report.update(
            repeat=repeat,
            web_access=False,
            task_family=case["task"]["id"],
            agent=execution,
            pi_version=experiment.agent.pi_version
            if experiment.agent.adapter == "pi"
            else None,
            auto_compaction=experiment.agent.auto_compaction,
            cache_warming="off",
            pi_settings=pi_session_settings(
                agent_dir, experiment.api_proxy.reasoning_effort
            )
            if experiment.agent.adapter == "pi"
            else None,
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
        accounting["grading_wall_s"] = (
            server.grading_credit() + accounting["post_agent_evaluation_wall_s"]
        )
        accounting["feedback_wall_s"] = server.seconds
        if arm:
            report.update(
                arm=experiment.arm,
                authoring=arm.authoring,
                simulator=arm.simulator,
                simulation_calls=simulator.calls if simulator else [],
            )
    if proxy and proxy.policy_violations:
        raise RuntimeError(f"Inference policy violation: {proxy.policy_violations}")
    if proxy and report["status"].startswith("agent_"):
        requests = read_api(workspace / "api.jsonl")
        # Pi can exit successfully after an API error. A terminal routing/provider
        # failure is an interrupted trial, even when earlier calls succeeded.
        last = requests[-1] if requests else {}
        if (
            last.get("provider_error")
            or (last.get("status") or 0) >= 400
            or (
                requests
                and last.get("status") is None
                and execution["status"] == "completed"
            )
        ):
            raise ModelAPIError(
                f"Model API failed before submission (HTTP {last.get('status')}); "
                "trial remains unscored; see api.jsonl and agent stdout"
            )
    return report


def source_identity(root):
    """Hash running code and the catalog; a resumed campaign must not mix them."""
    package = Path(__file__).resolve().parents[1]
    paths = [p for p in package.rglob("*") if p.suffix in (".py", ".jl")]
    for kind in (
        "arms",
        "evaluators",
        "hardware",
        "objects",
        "physics",
        "suites",
        "tasks",
    ):
        paths += Path(root, "benchmark", kind).rglob("*.yaml")
    paths += [
        p for name in ("Manifest.toml", "uv.lock") if (p := Path(root, name)).exists()
    ]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(f"{path.name}\0".encode() + path.read_bytes())
    return digest.hexdigest()


def attempt_dirs(output, case_id, repeat):
    base = output / f"{case_id}--r{repeat}"
    dirs = [base] if base.exists() else []
    while dirs and (path := output / f"{base.name}.retry{len(dirs)}").exists():
        dirs.append(path)
    return dirs, output / (base.name + (f".retry{len(dirs)}" if dirs else ""))


def paid(workspace):
    return any(
        (row.get("usage") or {}).get("cost_credits") or row.get("status") == 200
        for row in read_api(workspace / "api.jsonl")
    )


def frozen(workspace):
    """Final submission bytes not yet recorded as the trial result."""
    attempts = sorted((workspace / "control/submissions").glob("attempt-*"))
    last = attempts[-1] if attempts else None
    if (
        last
        and (last / "receipt.json").exists()
        and not (last / "result.json").exists()
    ):
        return last
    judge = workspace / "evaluation"
    if (judge / "sequence.seq").exists():
        return judge


def finalize(workspace, experiment, report):
    path = workspace / "accounting.json"
    accounting = json.loads(path.read_text()) if path.exists() else {}
    accounting.update(
        status="finished",
        resumed_at=utc_now(),
        usage=summarize_api(
            workspace / "api.jsonl", configured=experiment.api_proxy is not None
        ),
    )
    write_json(path, accounting)
    report["accounting"] = accounting
    write_json(workspace / "result.json", report)
    return report


def regrade(catalog, experiment, case, workspace, repeat, directory):
    """Grade frozen bytes only; the agent session is never resumed or rerun."""
    done = directory / "result.json"
    report = (
        json.loads(done.read_text())
        if done.exists()
        else grade_frozen(
            directory / "sequence.seq",
            case,
            catalog.root,
            directory,
            experiment.feedback.timeout_s,
            execute,
            backend_path=workspace / "control/backend.json"
            if experiment.backend
            else None,
        )
    )
    judge = workspace / "evaluation"
    if directory != judge:
        judge.mkdir(exist_ok=True)
        shutil.copyfile(directory / "sequence.seq", judge / "sequence.seq")
        write_json(judge / "result.json", report)
    attempts = [
        {
            **json.loads((d / "receipt.json").read_text()),
            "evaluation": json.loads((d / "result.json").read_text()),
            "directory": str(d),
        }
        for d in sorted((workspace / "control/submissions").glob("attempt-*"))
        if (d / "result.json").exists()
    ]
    report.update(
        repeat=repeat,
        task_family=case["task"]["id"],
        workspace=str(workspace),
        submissions=attempts,
        resumed="regraded_frozen_submission",
    )
    return finalize(workspace, experiment, report)


def run(catalog, experiment, output=None, *, cases="*", repeats=None):
    """Launch or resume a campaign; returns (output, reports, complete)."""
    started, started_at = time.monotonic(), utc_now()
    if experiment.agent.uses_model:
        # Unpinned models/providers make scores incomparable across runs.
        for value in (experiment.agent.model, experiment.api_proxy.provider_endpoint):
            if not (value or "").strip() or "REPLACE_WITH" in value:
                raise ValueError(
                    "model runs require a pinned model and provider_endpoint"
                )
        if experiment.api_proxy.max_campaign_usd is None:
            raise ValueError("model runs require api_proxy.max_campaign_usd")
    # Validate before creating output, materializing phantoms, or starting agents.
    if experiment.arm is not None or experiment.arm_runtime is not None:
        experiment = type(experiment).model_validate(experiment.model_dump(mode="json"))
    output = (
        Path(output).resolve()
        if output
        else catalog.root
        / "runs"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    )
    arm = catalog.get("arms", experiment.arm) if experiment.arm else None
    if arm:
        arm_runtime.CAMPAIGN = hashlib.sha256(str(output).encode()).hexdigest()[:16]
        arm_runtime.remove_campaign_containers(arm_runtime.CAMPAIGN)
    runtime = resolve_runtime(experiment, arm) if arm else None
    identity = {
        # Repeats and case filters select trials; they do not change one.
        "experiment": experiment.model_dump(mode="json", exclude={"repeats"}),
        "source_sha256": source_identity(catalog.root),
        "images": [runtime["author_image"], runtime["simulator_image"]]
        if runtime
        else None,
    }
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    owner = (output / ".lock").open("a")  # Released when this process exits.
    try:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError(f"another process is running {output}") from None
    if (output / "campaign.json").exists():
        if json.loads((output / "campaign.json").read_text()) != identity:
            raise ValueError(
                f"{output} was launched with a different experiment, source or image; "
                "use a new --out"
            )
    elif any(p.name != ".lock" for p in output.iterdir()):
        raise ValueError(f"{output} is not empty and has no campaign.json")
    else:
        write_json(output / "campaign.json", identity)
        if runtime:
            write_json(output / "arm_runtime.json", runtime)
    budget = Budget(
        experiment.api_proxy.max_campaign_usd if experiment.api_proxy else None,
        sum(
            (row.get("usage") or {}).get("cost_credits") or 0
            for path in output.glob("*/api.jsonl")
            for row in read_api(path)
        ),
    )
    selected = [
        (case, repeat)
        for case in catalog.cases(catalog.get("suites", experiment.suite))
        if fnmatch.fnmatch(case["case_id"], cases)
        for repeat in range(experiment.repeats if repeats is None else repeats)
    ]
    if not selected:
        raise ValueError(f"no suite case matches {cases!r}")
    lock, stop, abort = threading.Lock(), threading.Event(), threading.Event()
    grading_slots = threading.BoundedSemaphore(experiment.concurrency.graders)
    campaign = {"schema_version": 1, "scope": "campaign", "started_at": started_at}
    stopped = []

    def publish():
        reports = [
            json.loads(p.read_text()) for p in sorted(output.glob("*/result.json"))
        ]
        text = "".join(json.dumps(r, allow_nan=False) + "\n" for r in reports)
        (output / "results.jsonl.tmp").write_text(text)
        (output / "results.jsonl.tmp").replace(output / "results.jsonl")
        campaign.update(
            summarize_trials(output),
            spent_usd=budget.spent,
            wall_s=time.monotonic() - started,
        )
        write_json(output / "accounting.json", campaign)

    def plan(case, repeat):
        dirs, fresh = attempt_dirs(output, case["case_id"], repeat)
        if any((d / "result.json").exists() for d in dirs):
            return None
        if dirs and (directory := frozen(dirs[-1])):
            return "regrade", dirs[-1], directory
        if sum(map(paid, dirs)) > experiment.retries_after_spend:
            return "give_up", dirs[-1], dirs
        return "agent", fresh, None

    def job(case, repeat, action, workspace, extra):
        if stop.is_set():
            return
        try:
            if action == "regrade":
                report = regrade(catalog, experiment, case, workspace, repeat, extra)
            elif action == "give_up":
                report = finalize(
                    workspace,
                    experiment,
                    {
                        "case_id": case["case_id"],
                        "status": "infrastructure_interrupted",
                        "checks": [],
                        "metrics": {},
                        "repeat": repeat,
                        "task_family": case["task"]["id"],
                        "workspace": str(workspace),
                        "attempt_workspaces": [str(d) for d in extra],
                    },
                )
            else:
                if budget.exhausted():
                    raise CampaignStop("campaign spend cap reached")
                report = run_one(
                    catalog,
                    experiment,
                    case,
                    workspace,
                    repeat,
                    extra,
                    runtime=runtime,
                    grading_slots=grading_slots,
                    abort_event=abort,
                    budget=budget,
                )
        except CampaignStop as exc:
            stop.set()
            stopped.append(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - one failed trial must not stop the rest
            print(
                f"{case['case_id']} repeat={repeat}: interrupted "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
            return
        with lock:
            publish()
            print(f"{case['case_id']} repeat={repeat}: {report['status']}", flush=True)

    jobs = []
    for case, repeat in selected:
        case = copy.deepcopy(case)
        case["seed"] = experiment.seed
        if not (step := plan(case, repeat)):
            continue
        action, workspace, extra = step
        if action == "agent" and (
            case["evaluation"]["scope"] == "benchmark"
            or (arm and arm.simulator != "none")
        ):
            extra = output / "phantoms" / case["case_id"]
            if not (extra / "artifacts.json").exists():
                shutil.rmtree(extra, ignore_errors=True)  # Derived; rebuild if partial.
                materialize(case, catalog.root, extra)
        jobs.append((case, repeat, action, workspace, extra))
    pool = ThreadPoolExecutor(max_workers=experiment.concurrency.agents)
    futures = [pool.submit(job, *args) for args in jobs]
    try:
        for future in as_completed(futures):
            future.result()
    except BaseException:
        stop.set()
        abort.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        with lock:
            publish()
    if stopped:
        raise CampaignStop(stopped[0])
    remaining = [step for case, repeat in selected if (step := plan(case, repeat))]
    reports = [json.loads(p.read_text()) for p in sorted(output.glob("*/result.json"))]
    return output, reports, not remaining
