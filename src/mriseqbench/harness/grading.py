"""Trusted, restartable evaluation of frozen bytes.

Reported outcomes are only passed or failed. Adaptive grading has no time limit;
a verdict unresolved at the maximum spatial sampling or the finest timestep
fails with failure_kind unresolved_at_sample_cap or unresolved_at_timestep_cap.
Infrastructure crashes interrupt the job instead of becoming benchmark rows.
"""

import json
import os
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

from ..evaluation.evaluate import ADAPTIVE, normalize_verdict
from ..io import write_json
from .grading_pool import GradingSlots


class EvaluationIncomplete(RuntimeError):
    """The frozen submission remains available for resumed grading, without a score."""


class AnyEvent:
    """Set when any of the given events is set (e.g. trial closed or campaign abort)."""

    def __init__(self, *events):
        self.events = [event for event in events if event is not None]

    def is_set(self):
        return any(event.is_set() for event in self.events)


@contextmanager
def grading_slot(slots, stop=None):
    """Hold a campaign grading slot; a stop (campaign abort) ends the wait."""
    slots = slots or threading.BoundedSemaphore(1)
    if isinstance(slots, GradingSlots):
        if not slots.acquire(stop=stop):
            raise EvaluationIncomplete("Grading stopped; trial remains unscored")
    else:
        while not slots.acquire(timeout=1):
            if stop is not None and stop.is_set():
                raise EvaluationIncomplete("Grading stopped; trial remains unscored")
    try:
        if stop is not None and stop.is_set():
            raise EvaluationIncomplete("Grading stopped; trial remains unscored")
        yield
    finally:
        slots.release()


def worker_command(mode, case_path, snapshot, root, workspace):
    # MRISEQBENCH_EVALUATION_WORKER names an executable taking the worker's
    # own arguments, e.g. ops/local_gpu_worker.py, which grades on a GPU host at
    # identical absolute paths. Preflight (parsing) always stays local.
    remote = os.environ.get("MRISEQBENCH_EVALUATION_WORKER")
    program = (
        [remote]
        if mode == "evaluation" and remote
        else [sys.executable, "-m", "mriseqbench.evaluation.worker"]
    )
    return [
        *program,
        "--mode",
        mode,
        "--case",
        str(case_path),
        "--submission",
        str(snapshot),
        "--root",
        str(root),
        "--workspace",
        str(workspace),
    ]


def bounded_preflight(case, case_path, snapshot, root, directory, timeout_s, run, stop):
    """Parse under the lint/check time and memory bounds before the unbounded
    evaluation. A file that hangs or exhausts the parser, or fails preflight,
    is the verdict; evaluation starts only after an explicit pass."""
    gate = directory / "preflight"
    gate.mkdir(exist_ok=True)
    (gate / "result.json").unlink(missing_ok=True)
    command = worker_command("preflight", case_path, snapshot, root, gate)
    execution = run(command, gate, timeout_s, gate / "worker", stop_when=stop)
    if execution["status"] == "timeout":
        return {
            "case_id": case["case_id"],
            "status": "failed",
            "failure_kind": "preflight_time_limit",
            "checks": [
                {
                    "name": "preflight_time_limit",
                    "passed": False,
                    "detail": f"parsing the file exceeded {timeout_s:g} s",
                }
            ],
            "metrics": {},
        }
    try:
        if execution["status"] != "completed":
            raise ValueError(f"preflight worker {execution['status']}")
        result = json.loads((gate / "result.json").read_text())
        if result.get("status") not in {"preflight_passed", "failed"}:
            raise ValueError("preflight worker returned no verdict")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise EvaluationIncomplete(
            f"Grading interrupted; resume frozen submission at {directory}: {exc}"
        ) from exc
    return None if result["status"] == "preflight_passed" else result


def grade_frozen(
    snapshot,
    case,
    root,
    directory,
    timeout_s,
    run_process,
    *,
    backend_path=None,
    stop_when=None,
):
    """Transient worker failures restart in the same checkpoint directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    case_path = directory / "case.json"
    write_json(case_path, case)
    command = worker_command("evaluation", case_path, snapshot, root, directory)
    if backend_path:
        command += ["--backend", str(backend_path)]
    failed = bounded_preflight(
        case, case_path, snapshot, root, directory, timeout_s, run_process, stop_when
    )
    if failed is not None:
        write_json(directory / "result.json", failed)
        return failed
    adaptive = case["evaluation"]["id"] in ADAPTIVE
    result_path = directory / "result.json"
    # A resumed invocation shares checkpoints, but must not overwrite the logs
    # and diagnostics of the interrupted invocation.
    prior_indices = [
        int(match[1])
        for path in directory.iterdir()
        if (
            match := re.fullmatch(
                r"(?:worker(?:-execution|-diagnostic)?|previous-result)-(\d+)\..+",
                path.name,
            )
        )
    ]
    first_index = max(prior_indices, default=-1) + 1
    for recovery in range(3):
        index = first_index + recovery
        if stop_when is not None and stop_when.is_set():
            raise EvaluationIncomplete(
                f"Grading interrupted; resume frozen submission at {directory}"
            )
        # Never consume a stale result after a crashed/restarted worker.
        if result_path.exists():
            result_path.replace(directory / f"previous-result-{index:03d}.json")
        execution = run_process(
            command,
            directory,
            None if adaptive else timeout_s,
            directory / f"worker-{index:03d}",
            stop_when=stop_when,
        )
        write_json(directory / f"worker-execution-{index:03d}.json", execution)
        try:
            result = json.loads(result_path.read_text())
            if execution["status"] != "completed":
                raise ValueError("submission evaluator did not complete")
            result = normalize_verdict(result, case)
            if not isinstance(result, dict) or result.get("status") not in {
                "passed",
                "failed",
            }:
                raise ValueError("submission evaluator returned no supported verdict")
        except (OSError, ValueError, TypeError) as exc:
            if result_path.exists():
                result_path.replace(directory / f"worker-diagnostic-{index:03d}.json")
            if recovery == 2 or (stop_when is not None and stop_when.is_set()):
                raise EvaluationIncomplete(
                    f"Grading interrupted; resume frozen submission at {directory}: {exc}"
                ) from exc
            continue
        if recovery:
            result["worker_recoveries"] = recovery
        write_json(result_path, result)
        return result
