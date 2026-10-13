"""Trusted, restartable evaluation of frozen bytes; interruptions have no verdict."""

import json
import sys
from pathlib import Path

from ..io import write_json


class EvaluationIncomplete(RuntimeError):
    """The frozen submission remains available for resumed grading, without a score."""


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
    retry_context=None,
):
    """Adaptive built-in grading has no participant deadline or terminal budget cap.

    Transient worker failures restart in the same checkpoint directory. Persistent
    infrastructure failures interrupt the job instead of becoming benchmark rows.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    case_path = directory / "case.json"
    write_json(case_path, case)
    command = [
        sys.executable,
        "-m",
        "mriseqbench.evaluation.worker",
        "--mode",
        "evaluation",
        "--case",
        str(case_path),
        "--submission",
        str(snapshot),
        "--root",
        str(root),
        "--workspace",
        str(directory),
    ]
    if backend_path:
        command += ["--backend", str(backend_path)]
    adaptive = backend_path is None and case["evaluation"]["id"] in {
        "s1",
        "s2",
        "s3",
        "s4",
    }
    budget = None if adaptive else timeout_s
    result_path = directory / "result.json"
    for recovery in range(3):
        if stop_when is not None and stop_when.is_set():
            raise EvaluationIncomplete(
                f"Grading interrupted; resume frozen submission at {directory}"
            )
        # Never consume a stale result after a crashed/restarted worker.
        if result_path.exists():
            result_path.replace(directory / f"previous-result-{recovery:03d}.json")
        execution = run_process(
            command,
            directory,
            budget,
            directory / f"worker-{recovery:03d}",
            stop_when=stop_when,
        )
        write_json(directory / f"worker-execution-{recovery:03d}.json", execution)
        try:
            result = json.loads(result_path.read_text())
            if execution["status"] != "completed":
                raise ValueError("submission evaluator did not complete")
            if not isinstance(result, dict) or result.get("status") not in {
                "passed",
                "failed",
                "smoke_passed",
            }:
                raise ValueError("submission evaluator returned no supported verdict")
        except (OSError, ValueError, TypeError) as exc:
            if result_path.exists():
                result_path.replace(
                    directory / f"worker-diagnostic-{recovery:03d}.json"
                )
            if recovery == 2 or (stop_when is not None and stop_when.is_set()):
                raise EvaluationIncomplete(
                    f"Grading interrupted; resume frozen submission at {directory}: {exc}"
                ) from exc
            continue
        if recovery:
            result["worker_recoveries"] = recovery
            write_json(result_path, result)
        return result
