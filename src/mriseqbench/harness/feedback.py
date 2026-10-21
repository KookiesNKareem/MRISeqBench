"""Immutable submission attempts with failure-driven retries."""

import hashlib
import hmac
import json
import secrets
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from ..io import write_json
from ..process import execute
from .grading import EvaluationIncomplete, grade_frozen
from .proxy import QuietHTTPServer


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
        self.lock = threading.Lock()
        self.checks_used, self.lints, self.seconds, self.requests = 0, 0, 0.0, 0
        self.max_submissions = max_submissions
        self.grading_slots = grading_slots or threading.BoundedSemaphore(1)
        self.attempts = []
        self.submissions_started = 0
        self.evaluation_failure = None
        self.grading_seconds = 0.0
        self.backoff_seconds = 0.0
        self.grading_started = None
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
        self.closed.set()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(2)
        # Handler workers observe closed and their process groups are killed.
        with self.lock:
            write_json(self.directory / "stats.json", self.stats())

    def stats(self):
        return {
            "checks_used": self.checks_used,
            "lints": self.lints,
            "feedback_wall_s": round(self.seconds, 6),
            "submitted": self.submitted.is_set(),
            "submissions_used": self.submissions_started,
            "max_submissions": self.max_submissions,
            "grading_wall_s": round(self.grading_credit() - self.backoff_seconds, 6),
            "provider_backoff_wall_s": round(self.backoff_seconds, 6),
        }

    def start_agent_clock(self, timeout_s):
        self.agent_timeout_s = timeout_s
        self.agent_credit_start = self.grading_credit()
        self.agent_started = time.monotonic()

    def runtime_status(self):
        # Do not acquire the submission lock: status must remain available while
        # a submission is being graded or is waiting for a grader slot.
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
        """Submission grading and provider back-off are excluded from the agent deadline."""
        with self.retry_time_lock:
            started = self.grading_started
            return (
                self.grading_seconds
                + self.backoff_seconds
                + (time.monotonic() - started if started is not None else 0.0)
            )

    def exclude(self, seconds):
        with self.retry_time_lock:
            self.backoff_seconds += seconds

    @contextmanager
    def grading(self):
        with self.retry_time_lock:
            self.grading_started = time.monotonic()
        try:
            yield
        finally:
            with self.retry_time_lock:
                self.grading_seconds += time.monotonic() - self.grading_started
                self.grading_started = None

    def _submit(self, body):
        """Called under the run lock; grade the frozen bytes before any disclosure."""
        attempt = len(self.attempts) + 1
        self.submissions_started += 1
        directory = self.directory / "submissions" / f"attempt-{attempt:03d}"
        directory.mkdir(parents=True, exist_ok=False)
        snapshot = directory / "sequence.seq"
        snapshot.write_bytes(body)
        receipt = {
            "case_id": self.case["case_id"],
            "attempt": attempt,
            "blind": attempt == 1,
            "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
            "accepted_at_unix_s": time.time(),
        }
        write_json(directory / "receipt.json", receipt)
        started = time.monotonic()
        try:
            with self.grading(), self.grading_slots:
                result = grade_frozen(
                    snapshot,
                    self.case,
                    self.root,
                    directory,
                    self.config.timeout_s,
                    execute,
                    backend_path=self.backend_path,
                    stop_when=self.closed,
                )
        except EvaluationIncomplete as exc:
            self.evaluation_failure = exc
            self.submitted.set()
            raise
        self.seconds += time.monotonic() - started
        record = {**receipt, "evaluation": result, "directory": str(directory)}
        self.attempts.append(record)
        write_json(self.directory / "submissions.json", self.attempts)
        self.submission_path.write_bytes(body)
        write_json(self.directory / "receipt.json", receipt)
        self._log({"kind": "submit", **record})
        if result["status"] == "failed" and attempt < self.max_submissions:
            return 200, {
                "status": "retry_required",
                "attempt": attempt,
                "submissions_remaining": self.max_submissions - attempt,
                "evaluation": result,
                "reason": "Submission failed. Revise sequence.seq and submit again. Lint/check provide only basic file, timing and hardware checks.",
            }
        self.submitted.set()
        return 200, {
            "status": "submitted",
            "attempt": attempt,
            "submissions_remaining": 0,
            "reason": "Submission recorded. The run is finished; stop now.",
        }

    def _log(self, value):
        with (self.directory / "feedback.jsonl").open("a") as log:
            log.write(json.dumps(value, allow_nan=False) + "\n")

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
            write_json(self.directory / f"last-{kind}-feedback.json", value)
        return code, value

    def _handle(self, kind, body):
        if not body or len(body) > self.config.max_submission_bytes:
            return 400, {
                "status": "invalid_submission",
                "reason": "empty or oversized sequence",
            }
        with self.lock:
            if self.closed.is_set() or self.submitted.is_set():
                return 403, {"status": "closed", "reason": "run is already finished"}
            if kind == "submit":
                return self._submit(body)
            if kind not in ("lint", "check"):
                return 403, {"status": "feedback_disabled"}
            if kind == "lint":
                self.lints += 1
            else:
                self.checks_used += 1
            self.requests += 1
            request_dir = self.directory / f"request-{self.requests:04d}"
            request_dir.mkdir()
            sequence_path = request_dir / "sequence.seq"
            sequence_path.write_bytes(body)
            # Both commands are basic checks; neither invokes SAR or image scoring.
            mode = "lint" if kind == "lint" else "preflight"
            command = [
                sys.executable,
                "-m",
                "mriseqbench.evaluation.worker",
                "--mode",
                mode,
                "--case",
                str(self.directory / "case.json"),
                "--submission",
                str(sequence_path),
                "--root",
                str(self.root),
                "--workspace",
                str(request_dir),
            ]
            started = time.monotonic()
            execution = execute(
                command,
                request_dir,
                self.config.timeout_s,
                request_dir / "worker",
                stop_when=self.closed,
            )
            self.seconds += time.monotonic() - started
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
                result = {"error": str(exc)}
            self._log(
                {
                    "kind": kind,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "result": result,
                    **self.stats(),
                }
            )
            return 503 if "error" in result else 200, result

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
