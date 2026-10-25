"""Immutable submission attempts with failure-driven retries."""

import hashlib
import hmac
import json
import secrets
import threading
import time
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from ..io import write_json
from ..process import execute
from .grading import (
    AnyEvent,
    EvaluationIncomplete,
    grade_frozen,
    grading_slot,
    worker_command,
)
from .proxy import QuietHTTPServer


def disclosed(result):
    """Like a scanner: say whether SAR was exceeded, never the model or values."""

    def redact(checks):
        sar = [c for c in checks if str(c.get("name", "")).startswith("sar_")]
        if not sar:
            return checks
        passed = all(c["passed"] for c in sar)
        return [c for c in checks if c not in sar] + [
            {
                "name": "sar",
                "passed": passed,
                "detail": "within the scanner SAR limit"
                if passed
                else "scanner SAR limit exceeded; the scan was refused",
            }
        ]

    value = {**result}
    for key in ("checks", "image_checks", "application_checks"):
        if isinstance(value.get(key), list):
            value[key] = redact(value[key])
    return value


class FeedbackServer:
    def __init__(
        self,
        case,
        config,
        root,
        control_dir,
        backend=None,
        max_submissions=3,
        grading_slots=None,
        abort=None,
    ):
        self.case, self.config, self.root = case, config, Path(root).resolve()
        self.directory = Path(control_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.directory / "case.json", case)
        self.backend_path = None
        if backend:
            self.backend_path = self.directory / "backend.json"
            write_json(self.backend_path, backend.model_dump(mode="json"))
        self.token = secrets.token_hex(32)
        self.submitted = threading.Event()
        self.closed = threading.Event()
        # Workers stop when the trial closes or the campaign aborts, even
        # while the runner drains them after the agent has exited.
        self.stopping = AnyEvent(self.closed, abort)
        self.lock = threading.Lock()
        self.checks_used, self.lints, self.seconds = 0, 0, 0.0
        self.max_submissions = max_submissions
        self.grading_slots = grading_slots or threading.BoundedSemaphore(1)
        self.attempts = []
        self.submissions_started = 0
        self.evaluation_failure = None  # Infrastructure latch; see interrupt().
        # Grading, simulator-queue and provider back-off waits; overlapping
        # waits are credited once (in total and per kind) for the time waited.
        # Keys are kinds, plus None for the union of all kinds.
        self.active, self.since, self.excluded_seconds = Counter(), {}, 0.0
        self.excluded_by_kind = {
            "grading": 0.0,
            "simulator_queue": 0.0,
            "provider_backoff": 0.0,
        }
        self.retry_time_lock = threading.Lock()
        self.submission_path = self.directory / "accepted.seq"
        self.agent_started = None
        self.agent_timeout_s = None
        self.agent_credit_start = 0.0
        self.httpd = QuietHTTPServer(("127.0.0.1", 0), self._handler())
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        # Handler workers observe closed and their process groups are killed.
        self.closed.set()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(2)

    def stats(self):
        return {
            "checks_used": self.checks_used,
            "lints": self.lints,
            "feedback_wall_s": round(self.seconds, 6),
            "submitted": self.submitted.is_set(),
            "submissions_used": self.submissions_started,
            "max_submissions": self.max_submissions,
            "grading_wall_s": round(self.excluded_by_kind["grading"], 6),
            "provider_backoff_wall_s": round(
                self.excluded_by_kind["provider_backoff"], 6
            ),
            "simulator_queue_wall_s": round(
                self.excluded_by_kind["simulator_queue"], 6
            ),
        }

    def start_agent_clock(self, timeout_s):
        self.agent_timeout_s = timeout_s
        self.agent_credit_start = self.grading_credit()
        self.agent_started = time.monotonic()

    def runtime_status(self):
        # Lock-free: status stays available while a submission is being graded.
        elapsed = (
            time.monotonic() - self.agent_started
            if self.agent_started is not None
            else 0.0
        )
        credit = max(0.0, self.grading_credit() - self.agent_credit_start)
        charged = max(0.0, elapsed - credit)
        remaining = (
            max(0.0, self.agent_timeout_s - charged)
            if self.agent_timeout_s is not None
            else None
        )
        return {
            "status": "closed"
            if self.closed.is_set() or self.submitted.is_set()
            else "running"
            if self.agent_started is not None
            else "preparing",
            "agent_wall_s": elapsed,
            "charged_wall_s": charged,
            "excluded_grading_wall_s": min(elapsed, credit),
            "remaining_s": remaining,
            "submissions_used": self.submissions_started,
            "submissions_remaining": max(
                0, self.max_submissions - self.submissions_started
            ),
        }

    def grading_credit(self):
        """Grading, simulator-queue waits and provider back-off are excluded from the agent deadline."""
        with self.retry_time_lock:
            started = self.since.get(None)
            return self.excluded_seconds + (
                time.monotonic() - started if started is not None else 0.0
            )

    def backoff(self):
        """Exclude a provider back-off wait (ended early by cancellation)."""
        return self.excluded("provider_backoff")

    def interrupt(self, failure):
        """Latch an infrastructure failure: stop the agent and leave the trial
        interrupted and resumable (never a benchmark outcome). Shared by
        trusted grading, lint/check workers and the agent-side simulator."""
        if not isinstance(failure, EvaluationIncomplete):
            failure = EvaluationIncomplete(
                f"Infrastructure failure: {type(failure).__name__}: {failure}"
            )
        with self.retry_time_lock:
            if self.evaluation_failure is None:
                self.evaluation_failure = failure
                write_json(
                    self.directory / "infrastructure-failure.json",
                    {"error": str(failure), "at_unix_s": time.time()},
                )
        self.submitted.set()
        return failure

    @contextmanager
    def excluded(self, kind):
        """Live-credit a wait so the running agent deadline extends while it lasts."""
        with self.retry_time_lock:
            now = time.monotonic()
            for key in (kind, None):
                if not self.active[key]:
                    self.since[key] = now
                self.active[key] += 1
        try:
            yield
        finally:
            with self.retry_time_lock:
                now = time.monotonic()
                for key in (kind, None):
                    self.active[key] -= 1
                    if not self.active[key]:
                        elapsed = now - self.since.pop(key)
                        if key is None:
                            self.excluded_seconds += elapsed
                        else:
                            self.excluded_by_kind[key] += elapsed

    def _submit(self, body):
        """Called under the run lock; grade the frozen bytes before any disclosure."""
        attempt = len(self.attempts) + 1
        self.submissions_started += 1
        directory = self.directory / "submissions" / f"attempt-{attempt:03d}"
        try:
            directory.mkdir(parents=True, exist_ok=False)
            snapshot = directory / "sequence.seq"
            snapshot.write_bytes(body)
            receipt = {
                "case_id": self.case["case_id"],
                "attempt": attempt,
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
                "accepted_at_unix_s": time.time(),
            }
            write_json(directory / "receipt.json", receipt)
            started = time.monotonic()
            with (
                self.excluded("grading"),
                grading_slot(self.grading_slots, self.stopping),
            ):
                result = grade_frozen(
                    snapshot,
                    self.case,
                    self.root,
                    directory,
                    self.config.timeout_s,
                    execute,
                    backend_path=self.backend_path,
                    stop_when=self.stopping,
                )
            self.seconds += time.monotonic() - started
            self.submission_path.write_bytes(body)
        except Exception as exc:
            # Any trusted grading or storage failure interrupts the trial: the
            # accepted bytes stay frozen for recovery and nothing is finalised.
            failure = self.interrupt(
                exc
                if isinstance(exc, EvaluationIncomplete)
                else EvaluationIncomplete(
                    f"Trusted grading or storage failed at {directory}: "
                    f"{type(exc).__name__}: {exc}"
                )
            )
            if failure is exc:
                raise
            raise failure from exc
        self.attempts.append(
            {**receipt, "evaluation": result, "directory": str(directory)}
        )
        if result["status"] == "failed" and attempt < self.max_submissions:
            return 200, {
                "status": "retry_required",
                "attempt": attempt,
                "submissions_remaining": self.max_submissions - attempt,
                "evaluation": disclosed(result),
                "reason": "Submission failed. Revise sequence.seq and submit again. Lint/check provide only basic file, timing and hardware checks.",
            }
        self.submitted.set()
        return 200, {
            "status": "submitted",
            "attempt": attempt,
            "submissions_remaining": 0,
            "reason": "Submission recorded. The run is finished; stop now.",
        }

    def handle(self, kind, token, body):
        if not isinstance(token, str) or not hmac.compare_digest(
            token.encode(), self.token.encode()
        ):
            return 403, {"status": "forbidden", "reason": "unknown run token"}
        if kind not in {"lint", "check", "submit", "status", "receipt"}:
            return 403, {"status": "feedback_disabled"}
        if kind == "status":
            return 200, self.runtime_status()
        if kind == "receipt":
            with self.lock:
                path = self.directory / "last-submit-feedback.json"
                if not path.exists():
                    path = self.directory / "last-feedback.json"
                return (
                    (200, json.loads(path.read_text()))
                    if path.exists()
                    else (
                        404,
                        {
                            "status": "no_feedback",
                            "reason": "No feedback has been disclosed yet.",
                        },
                    )
                )
        code, value = self._handle(kind, body)
        value = {**value, "runtime": self.runtime_status()}
        # Store only the response actually disclosed to the participant. The
        # blind final-submission path never gains access to the private verdict.
        with self.lock:
            write_json(self.directory / "last-feedback.json", value)
            if kind == "submit":
                write_json(self.directory / "last-submit-feedback.json", value)
        return code, value

    def _handle(self, kind, body):
        if not body or len(body) > self.config.max_submission_bytes:
            return 400, {
                "status": "invalid_submission",
                "reason": "empty or oversized sequence",
            }
        with self.lock:
            if self.stopping.is_set() or self.submitted.is_set():
                return 403, {"status": "closed", "reason": "run is already finished"}
            if kind == "submit":
                return self._submit(body)
            if kind == "lint":
                self.lints += 1
            else:
                self.checks_used += 1
            request_dir = (
                self.directory / f"request-{self.lints + self.checks_used:04d}"
            )
            sequence_path = request_dir / "sequence.seq"
            try:
                request_dir.mkdir()
                sequence_path.write_bytes(body)
            except OSError as exc:
                raise self.interrupt(
                    EvaluationIncomplete(f"{kind} request storage failed: {exc}")
                ) from exc
            # Both commands are basic checks; neither invokes SAR or image scoring.
            command = worker_command(
                "lint" if kind == "lint" else "preflight",
                self.directory / "case.json",
                sequence_path,
                self.root,
                request_dir,
            )
            started = time.monotonic()
            execution = execute(
                command,
                request_dir,
                self.config.timeout_s,
                request_dir / "worker",
                stop_when=self.stopping,
            )
            self.seconds += time.monotonic() - started
            if self.stopping.is_set():
                return 403, {"status": "closed", "reason": "run is already finished"}
            if execution["status"] == "timeout":
                # Only the submitted bytes can make a bounded preflight hang.
                return 200, {
                    "status": "failed",
                    "checks": [
                        {
                            "name": "preflight_time_limit",
                            "passed": False,
                            "detail": f"{kind} exceeded {self.config.timeout_s:g} s",
                        }
                    ],
                    "metrics": {},
                }
            try:
                if execution["status"] != "completed":
                    raise ValueError(f"feedback worker {execution['status']}")
                result = json.loads((request_dir / "result.json").read_text())
                if not isinstance(result, dict) or result.get("status") not in {
                    "lint_passed",
                    "preflight_passed",
                    "failed",
                }:
                    raise ValueError("feedback worker returned no valid result")
            except (OSError, ValueError) as exc:
                # Malformed input is a failed check inside the worker; a worker
                # that cannot launch, crashes or returns nothing is infrastructure.
                raise self.interrupt(
                    EvaluationIncomplete(f"{kind} worker failed: {exc}")
                ) from exc
            return 200, result

    def _handler(server):
        class Handler(BaseHTTPRequestHandler):
            def reply(self, code, value):
                body = json.dumps(value, allow_nan=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self.connection.settimeout(10)
                if self.path not in ("/lint", "/check", "/submit"):
                    return self.reply(404, {"status": "not_found"})
                if self.headers.get("Transfer-Encoding"):
                    return self.reply(400, {"status": "invalid_body"})
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= server.config.max_submission_bytes:
                        return self.reply(413, {"status": "invalid_body"})
                    body = self.rfile.read(length)
                    if len(body) != length:
                        return self.reply(400, {"status": "invalid_body"})
                except (OSError, ValueError):
                    return self.reply(400, {"status": "invalid_body"})
                try:
                    reply = server.handle(
                        self.path[1:], self.headers.get("X-MRISeqBench-Token", ""), body
                    )
                except EvaluationIncomplete as exc:
                    return self.reply(503, {"error": str(exc)})
                self.reply(*reply)

            def do_GET(self):
                if self.path not in ("/status", "/receipt"):
                    return self.reply(404, {"status": "not_found"})
                self.reply(
                    *server.handle(
                        self.path[1:], self.headers.get("X-MRISeqBench-Token", ""), b""
                    )
                )

            def log_message(self, *args):
                pass

        return Handler
