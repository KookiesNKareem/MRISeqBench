"""Resumable campaigns of sandboxed case/repeat trials with trusted evaluation.

Trial state lives only in workspace files, so rerunning a campaign resumes it:
`result.json` is final; a trial whose session had ended is completed from its
recorded submissions without rerunning the agent (see `recover`); any other
interrupted trial is void and reruns in a fresh sibling workspace
(`<case>--r<k>.retry<n>`). Infrastructure interruptions are never terminal.
"""

import copy
import fcntl
import fnmatch
import hashlib
import json
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from ..evaluation.evaluate import normalize_verdict
from ..evaluation.s1_koma import NATIVE_INCONSISTENCY
from ..io import write_json
from ..phantoms import materialize
from ..process import execute, expand
from . import arm_runtime
from .accounting import (
    read_api,
    read_rejections,
    summarize_api,
    summarize_trials,
    utc_now,
)
from .arm_mailbox import Mailbox
from .arm_runtime import (
    DockerUnavailable,
    Simulator,
    SimulatorSlots,
    arm_environment,
    launch_arm,
    resolve_runtime,
)
from .feedback import FeedbackServer
from .grading import EvaluationIncomplete, grade_frozen, grading_slot
from .grading_pool import GradingSlots
from .prepare import pi_model, prepare, prepare_pi
from .proxy import ApiProxy
from .sandbox import environment, launch

REMOTE_AUTHORS = None  # Initialized only by an explicitly configured fleet launcher.


class ModelAPIError(RuntimeError):
    """Inference infrastructure failed before a candidate could be scored."""


class ProxyRejected(ModelAPIError):
    """The proxy refused the agent's model route or model: a configuration error."""


class CampaignStop(RuntimeError):
    """Provider account has no credit; unfinished trials stay resumable."""


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


def grade(experiment, case, root, workspace, directory, stop_when):
    """Grade the frozen `directory/sequence.seq`, resuming its checkpoints."""
    return grade_frozen(
        directory / "sequence.seq",
        case,
        root,
        directory,
        experiment.feedback.timeout_s,
        execute,
        backend_path=workspace / "control/backend.json" if experiment.backend else None,
        stop_when=stop_when,
    )


def final_evaluation(source, case, experiment, root, workspace, stop_when=None):
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
    judge_dir = workspace / "evaluation"
    shutil.copyfile(source, judge_dir / "sequence.seq")
    return grade(experiment, case, root, workspace, judge_dir, stop_when)


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
    simulator_slots=None,
    abort_event=None,
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
        "simulator_queue_wall_s": None,
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
            simulator_slots=simulator_slots,
            abort_event=abort_event,
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


def grading_wall(server):
    """Excluded grading and provider back-off; simulator-queue waits are reported apart."""
    return server.grading_credit() - server.excluded_by_kind["simulator_queue"]


def proxy_refusal(path):
    """A 403 policy rejection (route/model) means the run is misconfigured."""
    refused = [r for r in read_rejections(path) if r.get("status") == 403]
    if refused:
        return (
            f"API proxy refused {len(refused)} request(s): {refused[-1]['error']} "
            f"({refused[-1]['path']}); check agent.model, the Pi model descriptor "
            "and api_proxy.allowed_routes"
        )
    return None


def model_api_failure(requests):
    """The last request, when it ended a session that made no final submission.

    Pi can exit successfully after an API error, including after a retryable
    failed attempt. A terminal routing/provider failure is an interrupted trial,
    even when earlier calls succeeded.
    """
    last = requests[-1] if requests else {}
    if (
        last.get("provider_error")
        or (last.get("status") or 0) >= 400
        or (
            # Upstream truncation, whatever the agent did next. A response the
            # host or agent cancelled (deadline, stop) is not a failure.
            requests
            and last.get("response_complete") is not True
            and not last.get("cancelled")
        )
    ):
        return last
    return None


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
    simulator_slots=None,
    abort_event=None,
):
    arm = catalog.get("arms", experiment.arm) if experiment.arm else None
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
                abort=abort_event,
            )
        )
        proxy = (
            stack.enter_context(
                ApiProxy(
                    experiment.api_proxy,
                    workspace / "api.jsonl",
                    model=experiment.agent.model or None,
                    stop_when=server.submitted,
                    pause=server.backoff,
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
            if not experiment.api_proxy
            or experiment.api_proxy.reasoning_effort == "none"
            else experiment.api_proxy.reasoning_effort,
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
            env.update(
                prepare_pi(
                    agent_dir,
                    values["api_base_url"],
                    experiment.agent,
                    experiment.api_proxy,
                )
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
                    slots=simulator_slots,
                    waiting=lambda: server.excluded("simulator_queue"),
                    infrastructure=lambda reason: server.interrupt(
                        EvaluationIncomplete(
                            f"Simulator infrastructure failure: {reason}; "
                            "trial remains unscored"
                        )
                    ),
                )
            mailbox = stack.enter_context(
                Mailbox(workspace / "control/transport", server, proxy, simulator)
            )
        if proxy:
            # Cancel API streams before Mailbox waits for its workers, including
            # when setup/agent execution raises instead of returning normally.
            stack.callback(proxy.revoke)
        launcher = (
            (REMOTE_AUTHORS.launch if REMOTE_AUTHORS is not None else launch_arm)(
                runtime, command, agent_dir, workspace / "control", env
            )
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
                # Container processes can outlive the killed client during slow
                # teardown: cut off feedback, inference and simulation now.
                # Accepted grading still drains; file mode submits with the new token.
                server.token = secrets.token_hex(32)
                if proxy:
                    proxy.revoke()
                if simulator:
                    simulator.closed.set()
                if abort_event is not None and abort_event.is_set():
                    raise RuntimeError("Campaign aborted; trial remains unscored")
                if execution["status"] == "launch_error":
                    raise RuntimeError(
                        f"Agent failed to launch ({execution.get('error')}); "
                        "trial remains unscored"
                    )
                if (
                    arm
                    and not server.submitted.is_set()
                    and (
                        reason := arm_runtime.infrastructure_failure(
                            isolation, execution
                        )
                    )
                ):
                    raise RuntimeError(f"{reason}; trial remains unscored")
                # Durable before the slow cleanup below; recover() trusts it
                # only if nothing afterwards showed an infrastructure fault.
                write_json(
                    workspace / "session-exit.json",
                    {
                        "ended": execution["status"],
                        "submission_mode": experiment.submission_mode,
                    },
                )
            finally:
                elapsed = time.monotonic() - agent_started
                excluded = max(0.0, server.grading_credit() - credit_started)
                accounting.update(
                    agent_wall_s=elapsed,
                    agent_excluded_grading_wall_s=min(elapsed, excluded),
                    agent_charged_wall_s=max(0.0, elapsed - excluded),
                    grading_wall_s=grading_wall(server),
                    simulator_queue_wall_s=server.excluded_by_kind["simulator_queue"],
                )
        if proxy:
            proxy.revoke()
            if not proxy.settle(30):
                raise RuntimeError(
                    "Model API requests did not finish after revocation; "
                    "trial remains unscored"
                )
            if proxy.exhausted == "account":
                raise CampaignStop("Provider account has no credit (HTTP 402)")
            if proxy.policy_violations:
                # Durable, so the session-exit marker cannot finalise it.
                raise server.interrupt(
                    EvaluationIncomplete(
                        f"Inference policy violation: {proxy.policy_violations}"
                    )
                )
        if arm:
            # Drain simulator/feedback workers so a fault during teardown
            # latches before the final infrastructure check below.
            mailbox.close()
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
        final = server.submitted.is_set()
        if proxy and not final:
            last_assistant = None
            for line in (workspace / "agent.stdout.log").open():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if (
                    event.get("type") == "message_end"
                    and event.get("message", {}).get("role") == "assistant"
                ):
                    last_assistant = event["message"]
            if last_assistant and last_assistant.get("stopReason") == "error":
                raise ModelAPIError(
                    "Author session ended with an inference/transport error; trial remains unscored; see agent stdout"
                )
            if proxy.exhausted == "context_budget":
                raise ModelAPIError(
                    "Local context budget guard rejected the request; trial remains unscored; see api.jsonl"
                )
            # Pi exits 0 after refused requests; never score that as the agent's.
            if refusal := proxy_refusal(workspace / "api.jsonl"):
                raise ProxyRejected(refusal)
            last = model_api_failure(read_api(workspace / "api.jsonl"))
            if last is not None:
                raise ModelAPIError(
                    f"Model API failed before submission (HTTP {last.get('status')}); "
                    "trial remains unscored; see api.jsonl and agent stdout"
                )
        # The session ended for a reason attributable to the agent; recovery
        # may now finalise from the recorded submissions (see recover()).
        write_json(
            workspace / "session.json",
            {
                "ended": execution["status"],
                "final_submission": final,
                "submissions": len(attempts),
                "agent_charged_wall_s": accounting["agent_charged_wall_s"],
            },
        )
        report = no_submission(case, execution["status"])
        if attempts:
            # Reuse the immutable attempt verdict; never regrade changed agent files.
            report = dict(attempts[-1]["evaluation"])
            shutil.copyfile(server.submission_path, judge_dir / "sequence.seq")
            write_json(judge_dir / "case.json", case)
            write_json(judge_dir / "result.json", report)
        elif source is not None:
            grading_started = time.monotonic()
            try:
                with grading_slot(grading_slots, abort_event):
                    report = final_evaluation(
                        source, case, experiment, catalog.root, workspace, abort_event
                    )
            finally:
                accounting["post_agent_evaluation_wall_s"] = (
                    time.monotonic() - grading_started
                )
                accounting["grading_wall_s"] = (
                    grading_wall(server) + accounting["post_agent_evaluation_wall_s"]
                )
        report.update(
            repeat=repeat,
            agent=execution,
            pi_settings=pi_session_settings(
                agent_dir, experiment.api_proxy.reasoning_effort
            )
            if experiment.agent.adapter == "pi"
            else None,
            isolation=isolation,
            feedback=server.stats(),
            workspace=str(workspace),
            submissions=attempts,
        )
        accounting["grading_wall_s"] = (
            grading_wall(server) + accounting["post_agent_evaluation_wall_s"]
        )
        accounting["feedback_wall_s"] = server.seconds
        if arm:
            report.update(
                arm=experiment.arm,
                authoring=arm.authoring,
                simulator=arm.simulator,
                simulation_calls=simulator.calls if simulator else [],
            )
    return report


def source_identity(root):
    """Hash running code, the whole catalog (manifests, references, assets),
    participant docs and grader environment files; a resumed campaign must
    not mix them. Arm images are pinned separately by immutable ID."""
    root = Path(root).resolve()
    package = Path(__file__).resolve().parents[1]
    files = {
        f"{package.name}/{p.relative_to(package).as_posix()}": p
        for p in package.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    for p in [
        *(root / "benchmark").rglob("*"),
        *(root / "docs").glob("*.md"),
        *(root / "runtimes/bart").rglob("*"),
        # Grader environments: Julia project, its preferences, Python locks.
        *(
            root / name
            for name in (
                "Project.toml",
                "Manifest.toml",
                "LocalPreferences.toml",
                "pyproject.toml",
                "uv.lock",
            )
        ),
    ]:
        if p.is_file() and "__pycache__" not in p.parts:
            files[p.relative_to(root).as_posix()] = p
    digest = hashlib.sha256()
    for name in sorted(files):
        data = files[name].read_bytes()
        digest.update(f"{name}\0{len(data)}\0".encode() + data)
    return digest.hexdigest()


def attempt_dirs(output, case_id, repeat):
    base = output / f"{case_id}--r{repeat}"
    dirs = [base] if base.exists() else []
    while dirs and (path := output / f"{base.name}.retry{len(dirs)}").exists():
        dirs.append(path)
    return dirs, output / (base.name + (f".retry{len(dirs)}" if dirs else ""))


FAILURE_KINDS = {
    "completed": "no_submission",
    "timeout": "agent_timeout",
    "error": "agent_error",
}


def no_submission(case, agent_status):
    """Outcomes are binary: a genuine session end without a submission fails."""
    return {
        "case_id": case["case_id"],
        "status": "failed",
        "failure_kind": FAILURE_KINDS.get(agent_status, f"agent_{agent_status}"),
        "checks": [],
        "metrics": {},
    }


def read_status(path):
    try:
        return json.loads(path.read_text()).get("status")
    except (OSError, ValueError, AttributeError):
        return None


def clean_exit(workspace):
    """The agent's own exit, recorded before cleanup, if no later check
    attributed the trial's end to infrastructure."""
    path = workspace / "session-exit.json"
    if (
        not path.exists()
        or (workspace / "control/infrastructure-failure.json").exists()
    ):
        return None
    ended = json.loads(path.read_text())
    # File mode submits only after cleanup, so its final bytes may be unrecorded.
    if ended.get("submission_mode") != "submit":
        return None
    api = workspace / "api.jsonl"
    if proxy_refusal(api) or model_api_failure(read_api(api)):
        return None
    return {"ended": ended["ended"]}


def recover(workspace, max_submissions):
    """How an interrupted trial completes without its agent, or None to rerun.

    An agent session cannot be resumed. A trial is finalised from its files
    only when its session had ended: `session.json` records a genuine end, or
    the last accepted submission ended it (a pass, or the final allowance).
    Any other interruption, including grading of a non-final attempt, voids
    the trial: it reruns in a fresh workspace with the full time budget and
    submission allowance, and the void workspace is kept only for audit.
    Infrastructure interruptions are never terminal.
    """
    # A cache HIT revokes the trial; a crash may precede its durable latch.
    if (workspace / "control/infrastructure-failure.json").exists() or any(
        (row.get("response_cache_status") or "").upper() == "HIT"
        for row in read_api(workspace / "api.jsonl")
    ):
        return None
    path = workspace / "session.json"
    session = json.loads(path.read_text()) if path.exists() else clean_exit(workspace)
    attempts = [
        d
        for d in sorted((workspace / "control/submissions").glob("attempt-*"))
        if (d / "receipt.json").exists()
    ]
    if attempts:
        last = attempts[-1]
        number = int(last.name.removeprefix("attempt-"))
        if (
            session
            or number >= max_submissions
            or read_status(last / "result.json") == "passed"
        ):
            return "regrade", last
        return None
    if session:
        judge = workspace / "evaluation"
        if (judge / "sequence.seq").exists():
            return "regrade", judge
        return "closed", session
    return None


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


def regrade(
    catalog,
    experiment,
    case,
    workspace,
    repeat,
    directory,
    grading_slots=None,
    abort_event=None,
):
    """Grade frozen bytes only; the agent session is never resumed or rerun."""
    done = directory / "result.json"
    report = None
    if done.exists():
        # A cached verdict obeys the same binary-outcome rules as a fresh one.
        try:
            report = normalize_verdict(json.loads(done.read_text()), case)
        except ValueError:
            pass
        if not isinstance(report, dict) or report.get("status") not in {
            "passed",
            "failed",
        }:
            report = None
    if report is None:
        with grading_slot(grading_slots, abort_event):
            report = grade(
                experiment, case, catalog.root, workspace, directory, abort_event
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
        workspace=str(workspace),
        submissions=attempts,
        resumed="regraded_frozen_submission",
    )
    return finalize(workspace, experiment, report)


def run(
    catalog, experiment, output=None, *, cases="*", repeats=None, selected_keys=None
):
    """Launch or resume a campaign; returns (output, reports, complete)."""
    started, started_at = time.monotonic(), utc_now()
    if experiment.agent.uses_model:
        # Unpinned models/providers make scores incomparable across runs.
        for value in (experiment.agent.model, experiment.api_proxy.provider_endpoint):
            if not (value or "").strip() or "REPLACE_WITH" in value:
                raise ValueError(
                    "model runs require a pinned model and provider_endpoint"
                )
        if experiment.agent.adapter == "pi":
            pi_model(experiment.agent.model, experiment.api_proxy)
    output = (
        Path(output).resolve()
        if output
        else catalog.root
        / "runs"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    )
    # Lock before any side effect: a second process on the same --out must
    # not remove the first one's containers or race its identity check.
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    owner = (output / ".lock").open("a")  # Released when this process exits.
    try:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        owner.close()
        raise ValueError(f"another process is running {output}") from None
    arm = catalog.get("arms", experiment.arm) if experiment.arm else None
    if arm:
        arm_runtime.CAMPAIGN = hashlib.sha256(str(output).encode()).hexdigest()[:16]
        arm_runtime.remove_campaign_containers(arm_runtime.CAMPAIGN)
    runtime = resolve_runtime(experiment, arm) if arm else None
    identity = {
        # Repeats and case filters select trials, and concurrency is host
        # logistics; neither changes a trial.
        "experiment": experiment.model_dump(
            mode="json", exclude={"repeats", "concurrency"}
        ),
        "source_sha256": source_identity(catalog.root),
        "images": [runtime["author_image"], runtime["simulator_image"]]
        if runtime
        else None,
    }
    if (output / "campaign.json").exists():
        recorded = json.loads((output / "campaign.json").read_text())
        # Campaigns launched before concurrency left the identity recorded it.
        recorded.get("experiment", {}).pop("concurrency", None)
        if recorded != identity:
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
    selected = [
        (case, repeat)
        for case in catalog.cases(catalog.get("suites", experiment.suite))
        if fnmatch.fnmatch(case["case_id"], cases)
        for repeat in range(experiment.repeats if repeats is None else repeats)
    ]
    if selected_keys is not None:
        requested = set(selected_keys)
        available = {case["case_id"] + "--r" + str(repeat) for case, repeat in selected}
        if not requested or requested - available:
            raise ValueError(
                "recovery selection must contain only exact existing case/repeat keys"
            )
        selected = [
            (case, repeat)
            for case, repeat in selected
            if case["case_id"] + "--r" + str(repeat) in requested
        ]
    if not selected:
        raise ValueError(f"no suite case matches {cases!r}")
    lock, stop, abort = threading.Lock(), threading.Event(), threading.Event()
    grading_slots = GradingSlots.for_campaign(output, experiment.concurrency.graders)
    simulator_slots = SimulatorSlots(experiment.concurrency.simulators)
    campaign = {"schema_version": 1, "scope": "campaign", "started_at": started_at}
    stopped, misconfigured, unavailable = [], [], []

    def publish():
        paths = sorted(output.glob("*/result.json"))
        reports = [json.loads(p.read_text()) for p in paths]
        text = "".join(json.dumps(r, allow_nan=False) + "\n" for r in reports)
        (output / "results.jsonl.tmp").write_text(text)
        (output / "results.jsonl.tmp").replace(output / "results.jsonl")
        costs = [
            (row.get("usage") or {}).get("cost_credits")
            for path in output.glob("*/api.jsonl")
            for row in read_api(path)
        ]
        known = [cost for cost in costs if cost is not None]
        campaign.update(
            summarize_trials(output),
            # The total is null while any request lacks a recorded cost.
            observed_spent_usd=sum(known),
            spent_usd=sum(known) if len(known) == len(costs) else None,
            # Judge/native reader disagreements may be judge bugs: check by hand.
            needs_review=[
                p.parent.name for p in paths if NATIVE_INCONSISTENCY in p.read_text()
            ],
            wall_s=time.monotonic() - started,
            grading_pool=grading_slots.stats(),
        )
        write_json(output / "accounting.json", campaign)

    def plan(case, repeat):
        dirs, fresh = attempt_dirs(output, case["case_id"], repeat)
        if any((d / "result.json").exists() for d in dirs):
            return None
        if dirs and (step := recover(dirs[-1], experiment.max_submissions)):
            return step[0], dirs[-1], step[1]
        return "agent", fresh, None

    def job(case, repeat, action, workspace, extra):
        if stop.is_set() or simulator_slots.broken.is_set():
            return
        try:
            if action == "regrade":
                report = regrade(
                    catalog,
                    experiment,
                    case,
                    workspace,
                    repeat,
                    extra,
                    grading_slots,
                    abort,
                )
            elif action == "closed":
                report = finalize(
                    workspace,
                    experiment,
                    {
                        **no_submission(case, extra["ended"]),
                        "repeat": repeat,
                        "workspace": str(workspace),
                        "resumed": "recorded_session_end",
                    },
                )
            else:
                report = run_one(
                    catalog,
                    experiment,
                    case,
                    workspace,
                    repeat,
                    extra,
                    runtime=runtime,
                    grading_slots=grading_slots,
                    simulator_slots=simulator_slots,
                    abort_event=abort,
                )
        except CampaignStop as exc:
            stop.set()
            stopped.append(str(exc))
            return
        except DockerUnavailable as exc:
            stop.set()  # Exit 4 below; systemd restarts and resumes.
            unavailable.append(exc)
            return
        except ProxyRejected as exc:
            # A configuration error repeats in every trial: stop, keep them resumable.
            stop.set()
            misconfigured.append(str(exc))
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
        simulator_slots.drain()
        with lock:
            publish()
    if misconfigured:
        raise ValueError(misconfigured[0])
    if stopped:
        raise CampaignStop(stopped[0])
    if unavailable:
        raise unavailable[0]
    if simulator_slots.broken.is_set():
        raise DockerUnavailable("simulator containers could not be removed")
    reports = [json.loads(p.read_text()) for p in sorted(output.glob("*/result.json"))]
    return output, reports, not any(plan(case, repeat) for case, repeat in selected)
