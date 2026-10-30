"""Trusted stage machine. Only a passed stage unlocks the next user request."""

import base64
import hmac
import io
import json
from copy import deepcopy

from ..storage import atomic_json
from .accounting import read_api, summarize_requests
from .feedback import FeedbackServer, disclosed
from .grading import EvaluationIncomplete
from .prepare import brief_yaml, case_evaluation_summary, prompt


def public_case(case):
    value = deepcopy(case)
    value["hardware"] = {
        k: v for k, v in value["hardware"].items() if k not in {"sar", "provenance"}
    }
    return value


class WorkflowServer(FeedbackServer):
    def __init__(self, *args, workflow, stage_timeout_s, **kwargs):
        super().__init__(*args, **kwargs)
        self.workflow = workflow
        self.stage_timeout_s = stage_timeout_s
        self.stage_index = 0
        self.stage_started_charged = 0.0
        self.pending = False
        self.on_stage = None
        self.next_responses = {}
        self.api_start = 0
        self.stages = [
            {
                "index": i,
                "id": s["id"],
                "status": "not_reached",
                "submissions": 0,
                "charged_wall_s": 0.0,
                "api_request_start": 0,
            }
            for i, s in enumerate(workflow["stages"])
        ]
        self.stages[0]["status"] = "running"
        self.checkpoint()

    def stage_elapsed(self):
        return max(
            0.0, super().runtime_status()["charged_wall_s"] - self.stage_started_charged
        )

    def stage_expired(self):
        return (
            self.agent_started is not None
            and self.stage_elapsed() >= self.stage_timeout_s
        )

    def runtime_status(self):
        value = super().runtime_status()
        row = self.stages[self.stage_index]
        stage_remaining = max(0.0, self.stage_timeout_s - self.stage_elapsed())
        value.update(
            stage_index=self.stage_index,
            stage_id=row["id"],
            stage_remaining_s=stage_remaining,
            awaiting_next_turn=self.pending,
            submissions_remaining=max(0, self.max_submissions - row["submissions"]),
        )
        if value["remaining_s"] is not None:
            value["remaining_s"] = min(value["remaining_s"], stage_remaining)
        return value

    def stats(self):
        return {
            **super().stats(),
            "stage_index": self.stage_index,
            "max_submissions_per_stage": self.max_submissions,
        }

    def submission_metadata(self):
        return {
            "workflow": self.workflow["id"],
            "stage_index": self.stage_index,
            "stage_id": self.stages[self.stage_index]["id"],
            "stage_attempt": self.stages[self.stage_index]["submissions"] + 1,
        }

    def _submit(self, body):
        if self.pending:
            return 409, {
                "status": "awaiting_next_turn",
                "reason": "This stage passed; finish your response and wait for the next user request.",
            }
        if self.stage_expired():
            return 409, {
                "status": "stage_timeout",
                "reason": "Stage time allowance exhausted.",
            }
        return super()._submit(body)

    def complete_submission(self, attempt, result):
        row = self.stages[self.stage_index]
        row["submissions"] += 1
        row["evaluation"] = deepcopy(result)
        row["charged_wall_s"] = self.stage_elapsed()
        row["last_attempt"] = attempt
        if result["status"] == "passed":
            row["status"] = "passed"
            row["accepted_sha256"] = self.attempts[-1]["sha256"]
            if self.stage_index + 1 < len(self.stages):
                self.pending = True
                self.checkpoint()
                return 200, {
                    "status": "stage_passed",
                    "stage_index": self.stage_index,
                    "attempt": attempt,
                    "evaluation": disclosed(result),
                    "reason": "Stage passed. Finish your response now; a new user request will follow. Do not submit again yet.",
                }
        elif row["submissions"] < self.max_submissions:
            self.checkpoint()
            return 200, {
                "status": "retry_required",
                "attempt": attempt,
                "stage_index": self.stage_index,
                "submissions_remaining": self.max_submissions - row["submissions"],
                "evaluation": disclosed(result),
                "reason": "Revise sequence.seq to meet the current stage and submit again.",
            }
        else:
            row["status"] = "failed"
        row["usage"], row["api_request_end"] = self.stage_usage()
        self.checkpoint(terminal=True)
        self.submitted.set()
        return 200, {
            "status": "submitted",
            "attempt": attempt,
            "reason": "Workflow finished; stop now.",
        }

    def stage_usage(self):
        path = self.directory.parent / "api.jsonl"
        rows = read_api(path)
        return summarize_requests(
            rows[self.api_start :], configured=path.exists()
        ), len(rows)

    def checkpoint(self, terminal=False):
        value = self.report()
        value["terminal"] = terminal
        try:
            atomic_json(self.directory / "workflow-state.json", value)
        except OSError as exc:
            raise self.interrupt(
                EvaluationIncomplete(f"Workflow checkpoint failed: {exc}")
            ) from exc

    def report(self, ended=None):
        rows = deepcopy(self.stages)
        if ended is not None:
            row = rows[self.stage_index]
            row["api_request_end"] = len(read_api(self.directory.parent / "api.jsonl"))
            refresh_usage(rows, self.directory.parent / "api.jsonl")
            if row["status"] == "running":
                row.update(
                    status="failed", reason=ended, charged_wall_s=self.stage_elapsed()
                )
        passed = all(s["status"] == "passed" for s in rows)
        return {
            "case_id": self.workflow["case_id"],
            "status": "passed" if passed else "failed",
            "checks": [],
            "metrics": {},
            "workflow": {
                "id": self.workflow["id"],
                "mode": self.workflow.get("mode", "generate_edit"),
                "qualification": self.workflow["qualification"],
                "stages": rows,
                "complete": passed,
            },
            "reason": "All stages passed"
            if passed
            else "Workflow did not pass every stage",
        }

    def handle(self, kind, token, body):
        if kind != "next":
            return super().handle(kind, token, body)
        if not isinstance(token, str) or not hmac.compare_digest(
            token.encode(), self.token.encode()
        ):
            return 403, {"status": "forbidden"}
        try:
            previous = json.loads(body)["after_stage"]
            if type(previous) is not int:
                raise ValueError("invalid stage")
        except (ValueError, KeyError, TypeError):
            return 400, {"status": "invalid_stage"}
        with self.lock:
            if self.stopping.is_set() or self.submitted.is_set():
                return 409, {"status": "closed"}
            if previous in self.next_responses:
                return 200, self.next_responses[previous]
            if self.stage_expired():
                return 409, {"status": "stage_timeout"}
            if not self.pending or previous != self.stage_index:
                return 409, {"status": "not_ready"}
            try:
                self.stages[self.stage_index]["usage"], self.api_start = (
                    self.stage_usage()
                )
                self.stages[self.stage_index]["api_request_end"] = self.api_start
                self.stages[self.stage_index]["charged_wall_s"] = self.stage_elapsed()
                self.stage_index += 1
                self.pending = False
                stage = self.workflow["stages"][self.stage_index]
                self.case = deepcopy(stage["case"])
                self.stages[self.stage_index]["status"] = "running"
                self.stages[self.stage_index]["api_request_start"] = self.api_start
                self.stage_started_charged = super().runtime_status()["charged_wall_s"]
                atomic_json(self.directory / "case.json", self.case)
                if self.on_stage:
                    self.on_stage(public_case(self.case))
                message = (
                    stage["request"]
                    + "\n\n"
                    + prompt(self.case)
                    + (
                        "\nThis is the complete current contract. Earlier changes persist unless explicitly replaced. "
                        "After stage_passed, finish your response and wait for the next user request."
                    )
                )
                value = {
                    "status": "next_stage",
                    "stage_index": self.stage_index,
                    "message": message,
                    "files": {
                        "case.json": json.dumps(public_case(self.case), indent=2),
                        "task.yaml": brief_yaml(self.case),
                        "EVALUATION.md": case_evaluation_summary(self.case),
                        "CURRENT_STAGE.md": message,
                    },
                }
                if self.case["task"]["id"] == "rf_excitation_2d":
                    import numpy as np

                    from ..evaluation.excitation_target import target_arrays

                    buffer = io.BytesIO()
                    np.savez_compressed(
                        buffer,
                        **target_arrays(
                            self.case["task"]["sequence"].get("excitation_target")
                        ),
                    )
                    value["target_npz_base64"] = base64.b64encode(
                        buffer.getvalue()
                    ).decode()
                self.next_responses[previous] = value
                self.checkpoint()
                return 200, value
            except Exception as exc:
                raise self.interrupt(
                    EvaluationIncomplete(f"Workflow transition storage failed: {exc}")
                ) from exc


def refresh_usage(stages, path):
    requests = read_api(path)
    for row in stages:
        if row["status"] != "not_reached":
            row["usage"] = summarize_requests(
                requests[
                    row.get("api_request_start", 0) : row.get(
                        "api_request_end", len(requests)
                    )
                ],
                configured=path.exists(),
            )


def recover_workflow(workspace, session):
    path = workspace / "control/workflow-state.json"
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    if not state.pop("terminal", False) and not session:
        return None  # Conversation cannot be faithfully resumed: void and retry whole workflow.
    for row in state["workflow"]["stages"]:
        if row["status"] == "running":
            row.update(status="failed", reason="recorded_session_end")
    refresh_usage(state["workflow"]["stages"], workspace / "api.jsonl")
    return state
