"""Blind first submission, immutable attempts and failure-driven retries."""

import hashlib
import hmac
import json
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from ..io import write_json
from ..process import execute
from .proxy import QuietHTTPServer


class FeedbackServer:
    def __init__(
        self, case, config, root, control_dir, backend=None, max_submissions=3
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
        if not 1 <= max_submissions <= 3:
            raise ValueError("max_submissions must be between 1 and 3")
        self.max_submissions = max_submissions
        self.attempts = []
        self.submissions_started = 0
        self.submission_path = self.directory / "accepted.seq"
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
            "max_checks": self.config.max_checks if self.max_submissions == 1 else None,
            "lints": self.lints,
            "feedback_wall_s": round(self.seconds, 6),
            "submitted": self.submitted.is_set(),
            "submissions_used": self.submissions_started,
            "max_submissions": self.max_submissions,
        }

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
        command = [
            sys.executable,
            "-m",
            "mriseqbench.evaluation.worker",
            "--mode",
            "evaluation",
            "--case",
            str(self.directory / "case.json"),
            "--submission",
            str(snapshot),
            "--root",
            str(self.root),
            "--workspace",
            str(directory),
        ]
        if self.backend_path:
            command += ["--backend", str(self.backend_path)]
        started = time.monotonic()
        execution = execute(
            command,
            directory,
            self.config.timeout_s,
            directory / "worker",
            stop_when=self.closed,
        )
        self.seconds += time.monotonic() - started
        try:
            result = json.loads((directory / "result.json").read_text())
            if execution["status"] != "completed":
                raise ValueError("submission evaluator did not complete")
        except (OSError, ValueError):
            result = {
                "case_id": self.case["case_id"],
                "status": "evaluator_error",
                "checks": [],
                "metrics": {},
                "reason": f"submission worker {execution['status']} or invalid output",
            }
            write_json(directory / "result.json", result)
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
        if not hmac.compare_digest(token.encode(), self.token.encode()):
            return 403, {"status": "forbidden", "reason": "unknown run token"}
        if not body or len(body) > self.config.max_submission_bytes:
            return 400, {
                "status": "invalid_submission",
                "reason": "empty or oversized sequence",
            }
        with self.lock:
            if self.closed.is_set() or self.submitted.is_set():
                return 403, {"status": "closed", "reason": "run is already finished"}
            if kind == "submit":
                if self.max_submissions > 1:
                    return self._submit(body)
                # Publish the receipt last; acceptance cannot be changed by later agent writes.
                self.submissions_started += 1
                self.submission_path.write_bytes(body)
                receipt = {
                    "case_id": self.case["case_id"],
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "bytes": len(body),
                    "accepted_at_unix_s": time.time(),
                }
                temporary = self.directory / "receipt.tmp"
                write_json(temporary, receipt)
                temporary.replace(self.directory / "receipt.json")
                self._log({"kind": "submit", **receipt})
                self.submitted.set()
                return 200, {
                    "status": "submitted",
                    "reason": "Final sequence accepted. Stop now; no verdict is shown.",
                }
            if kind not in ("lint", "check") or (
                self.max_submissions == 1 and self.config.mode == "none"
            ):
                return 403, {"status": "feedback_disabled"}
            if (
                self.max_submissions == 1
                and kind == "check"
                and self.checks_used >= self.config.max_checks
            ):
                return 403, {"status": "budget_exhausted", **self.stats()}
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
                result = (
                    json.loads((request_dir / "result.json").read_text())
                    if execution["status"] == "completed"
                    else {
                        "status": "evaluator_error",
                        "reason": f"feedback worker {execution['status']}",
                    }
                )
            except (OSError, ValueError):
                result = {
                    "status": "evaluator_error",
                    "reason": "feedback worker returned no valid result",
                }
            refunded = kind == "check" and result["status"] in (
                "evaluator_error",
                "incomplete",
            )
            if refunded:
                self.checks_used -= 1
            self._log(
                {
                    "kind": kind,
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "refunded": refunded,
                    "result": result,
                    **self.stats(),
                }
            )
            return 200, {**result, "check_refunded": refunded, "budget": self.stats()}

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
                self.reply(
                    *server.handle(
                        self.path[1:], self.headers.get("X-MRISeqBench-Token", ""), body
                    )
                )

            def log_message(self, *args):
                pass

        return Handler
