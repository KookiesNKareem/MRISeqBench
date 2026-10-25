"""Credential-holding streaming API gateway, adapted from KAB's ApiProxy.

The host alone loads the real key. An agent receives a revocable per-run token
and can reach only configured routes at a fixed upstream HTTPS origin.
"""

import hashlib
import hmac
import http.client
import json
import os
import secrets
import socket
import threading
import time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .accounting import UsageCapture, utc_now

HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
    "authorization",
    "content-encoding",
}


class QuietHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def handle_error(self, request, client_address):
        pass


RETRY_STATUS = {429, 500, 502, 503, 504}


def in_flight_budget_error(body):
    """OpenRouter's temporary reservation limit is distinct from an empty account."""
    try:
        value = json.loads(body)
        error = value.get("error", value)
        return error.get("metadata", {}).get("reason") == "in_flight_budget_exhausted"
    except (ValueError, TypeError, AttributeError):
        return False


def retry_delay(response, retry, *, budget=False):
    try:
        delay = float(response.getheader("Retry-After", ""))
    except ValueError:
        delay = 120 if budget else 5 * 2**retry
    return min(max(delay, 1), 300 if budget else 60)


def strip_cache_control(value):
    if isinstance(value, dict):
        value.pop("cache_control", None)
        for item in value.values():
            strip_cache_control(item)
    elif isinstance(value, list):
        for item in value:
            strip_cache_control(item)


def endpoint_max_output_tokens(model):
    """The endpoint's recorded maximum output length, or None for unknown models."""
    models = json.loads(Path(__file__).with_name("pi_models.json").read_text())
    descriptor = models.get(model or "")
    return descriptor.get("maxOutputTokens") if isinstance(descriptor, dict) else None


def configure_request(payload, model, config, route="/chat/completions"):
    """Host-owned sampling, output cap and caching; client-sent values never pass through."""
    for key in ("temperature", "top_p"):
        value = getattr(config, key)
        if value is None:
            payload.pop(key, None)  # Omitted: the endpoint may not support it.
        else:
            payload[key] = value
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        payload.pop(key, None)
    # Always explicit: with the cap omitted OpenRouter applies a provider
    # default (65536 for anthropic/* on 2026-10-23), not the model's maximum.
    cap = config.max_output_tokens or endpoint_max_output_tokens(model)
    if cap is not None:
        key = "max_output_tokens" if route == "/responses" else "max_tokens"
        payload[key] = cap
    if (model or "").startswith("anthropic/"):
        # One automatic 5-minute breakpoint instead of the client's own
        # breakpoints (duplicates, or unapproved 1-hour cache writes).
        payload.pop("top_k", None)
        strip_cache_control(payload)
        payload["cache_control"] = {"type": "ephemeral"}
    return payload


class ApiProxy:
    def __init__(
        self,
        config,
        log_path,
        *,
        model=None,
        test_key=None,
        stop_when=None,
        pause=None,
    ):
        self.config = config
        self.pause = pause
        self.exhausted = None
        self.model = model
        self.stop_when = stop_when
        self.key = test_key if test_key is not None else os.environ.get(config.key_env)
        if not self.key:
            raise ValueError(
                f"API proxy requires host environment variable {config.key_env}"
            )
        self.token = secrets.token_hex(32)
        self.session_id = "mriseqbench-" + secrets.token_hex(16)
        self.policy_violations = []
        self.log_path = Path(log_path)
        self.lock = threading.RLock()
        self.idle = threading.Condition(self.lock)
        self.pending = {}
        self.active = True
        self.revoked = threading.Event()
        self.httpd = QuietHTTPServer(("127.0.0.1", 0), self._handler())
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def revoke(self):
        with self.lock:
            self.active = False
            self.revoked.set()
            sockets = [
                request["socket"] or request["connection"].sock
                for request in self.pending.values()
            ]
        for upstream_socket in sockets:
            self.cancel_socket(upstream_socket)

    def record(self, row):
        with self.lock, self.log_path.open("a") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")

    def allowed(self):
        """Whether upstream sends may still happen for this run."""
        with self.lock:
            # Pi may start another generation immediately after a terminal tool
            # reply, before the process supervisor next polls its stop event.
            return (
                self.active
                and not self.exhausted
                and not (self.stop_when is not None and self.stop_when.is_set())
            )

    def settle(self, timeout):
        """Wait until every request has written its `finished` row."""
        with self.idle:
            return self.idle.wait_for(lambda: not self.pending, timeout=timeout)

    def begin(self, request_id, conn, route, effective_request=None):
        with self.lock:
            if not self.allowed():
                return False
            self.pending[request_id] = {"connection": conn, "socket": None}
            self.record(
                {
                    "event": "started",
                    "request_id": request_id,
                    "route": route,
                    "started_at": utc_now(),
                    "effective_request": effective_request,
                }
            )
            return True

    def attach_socket(self, request_id, upstream_socket):
        with self.lock:
            self.pending[request_id]["socket"] = upstream_socket
        active = self.allowed()  # A stop during connection setup counts too.
        if not active:
            self.cancel_socket(upstream_socket)
        return active

    @staticmethod
    def cancel_socket(upstream_socket):
        if upstream_socket is not None:
            try:
                upstream_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def __exit__(self, *args):
        self.revoke()
        self.settle(30)
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(2)

    def _handler(proxy):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def reject(self, code, message):
                # Journaled apart from model requests (no usage); the runner
                # treats a 403 policy rejection as a configuration failure.
                proxy.record(
                    {
                        "event": "rejected",
                        "request_id": secrets.token_hex(16),
                        "path": self.path,
                        "status": code,
                        "error": message,
                        "at": utc_now(),
                    }
                )
                return self.reply(code, message)

            def reply(self, code, message):
                body = json.dumps({"error": {"message": message}}).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def forward(self):
                self.connection.settimeout(proxy.config.timeout_s)
                token = (
                    self.headers.get("Authorization", "")
                    .removeprefix("Bearer ")
                    .strip()
                )
                with proxy.lock:
                    authorized = proxy.active and hmac.compare_digest(
                        token.encode(), proxy.token.encode()
                    )
                if not authorized:
                    return self.reply(401, "unknown or expired run token")
                route = (
                    self.path.removeprefix("/v1")
                    if self.path.startswith("/v1/")
                    else ""
                )
                if route not in proxy.config.allowed_routes:
                    return self.reject(403, "route not allowed")
                if self.headers.get("Transfer-Encoding"):
                    return self.reply(400, "chunked request bodies are unsupported")
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 <= length <= proxy.config.max_request_bytes:
                        return self.reply(413, "request too large")
                    body = self.rfile.read(length) if length else None
                    if body is not None and len(body) != length:
                        return self.reply(400, "truncated body")
                except (ValueError, OSError):
                    return self.reply(400, "invalid request body")
                effective_request = None
                if route in ("/chat/completions", "/responses"):
                    try:
                        payload = json.loads(body or b"")
                    except (ValueError, UnicodeError):
                        return self.reply(400, "invalid model request")
                    if not isinstance(payload, dict) or any(
                        k in payload for k in ("models", "preset", "route")
                    ):
                        return self.reject(
                            403, "model routing overrides are prohibited"
                        )
                    if proxy.model and payload.get("model") != proxy.model:
                        return self.reject(
                            403, "request must use the run's configured model"
                        )
                    provider = proxy.config.provider_endpoint
                    if provider:
                        payload["provider"] = {
                            "only": [provider],
                            "order": [provider],
                            "allow_fallbacks": False,
                            "require_parameters": True,
                        }
                    configure_request(payload, proxy.model, proxy.config, route)
                    payload["session_id"] = proxy.session_id
                    payload.pop("reasoning_effort", None)
                    payload["reasoning"] = {"effort": proxy.config.reasoning_effort}
                    effective_request = {
                        k: payload[k]
                        for k in (
                            "model",
                            "provider",
                            "temperature",
                            "top_p",
                            "session_id",
                            "reasoning",
                            "max_tokens",
                            "max_output_tokens",
                            "stream",
                            "cache_control",
                        )
                        if k in payload
                    }
                    effective_request["tool_schema_sha256"] = hashlib.sha256(
                        json.dumps(payload.get("tools", []), sort_keys=True).encode()
                    ).hexdigest()
                    effective_request["response_cache_enabled"] = False
                    effective_request["client_session_header_present"] = bool(
                        self.headers.get("x-session-id")
                    )
                    body = json.dumps(payload, allow_nan=False).encode()
                upstream = urlsplit(proxy.config.upstream_url)
                # model_validate enforces HTTPS; model_construct is used only in local tests.
                conn_cls = (
                    http.client.HTTPSConnection
                    if upstream.scheme == "https"
                    else http.client.HTTPConnection
                )
                conn = conn_cls(
                    upstream.hostname, upstream.port, timeout=proxy.config.timeout_s
                )
                started, status, count = time.monotonic(), None, 0
                request_id = secrets.token_hex(16)
                if not proxy.begin(request_id, conn, route, effective_request):
                    return self.reply(401, "unknown or expired run token")
                capture = None
                last_receipt = None
                temporary_budget = False
                buffered_error = None
                framed = client_closed = False
                sent_headers = False
                try:
                    headers = {
                        k: v
                        for k, v in self.headers.items()
                        if k.lower() not in HOP
                        and k.lower() != "x-session-id"
                        and not k.lower().startswith("x-openrouter-cache")
                    }
                    headers["Authorization"] = f"Bearer {proxy.key}"
                    headers["Accept-Encoding"] = "identity"
                    headers["X-OpenRouter-Cache"] = "false"
                    headers["x-session-id"] = proxy.session_id
                    for retry in range(5):
                        if retry and not proxy.allowed():
                            # Access ended during back-off: never send again.
                            raise OSError("run access ended during provider back-off")
                        # Publish the socket before sending: a revoke() during
                        # connection setup must stop the request, not miss it.
                        conn.connect()
                        if not proxy.attach_socket(request_id, conn.sock):
                            raise OSError("run access ended before the request")
                        conn.request(
                            self.command,
                            upstream.path.rstrip("/") + route,
                            body=body,
                            headers=headers,
                        )
                        response = conn.getresponse()
                        status = response.status
                        temporary_budget = False
                        buffered_error = None
                        if status == 402:
                            # Inspect only the error envelope; preserve the bytes
                            # if this is a permanent error or retries run out.
                            buffered_error = response.read(65536)
                            temporary_budget = in_flight_budget_error(buffered_error)
                        if (
                            (status not in RETRY_STATUS and not temporary_budget)
                            or retry == 4
                            or not proxy.allowed()
                        ):
                            break
                        # Transient provider failures back off here instead of
                        # ending the agent's session; the wait is not charged.
                        delay = retry_delay(response, retry, budget=temporary_budget)
                        response.read()
                        conn.close()
                        proxy.record(
                            {
                                "event": "upstream_retry",
                                "request_id": request_id,
                                "route": route,
                                "status": status,
                                "delay_s": delay,
                                "reason": "in_flight_budget_exhausted"
                                if temporary_budget
                                else "transient_provider_error",
                            }
                        )
                        # Excluded from the agent clock for the time actually
                        # waited; revoke() ends the wait early.
                        with proxy.pause() if proxy.pause else nullcontext():
                            proxy.revoked.wait(delay)
                        conn = conn_cls(
                            upstream.hostname,
                            upstream.port,
                            timeout=proxy.config.timeout_s,
                        )
                        with proxy.lock:
                            proxy.pending[request_id].update(
                                connection=conn, socket=None
                            )
                    capture = UsageCapture(response.getheader("Content-Type", ""))
                    capture.generation_id = response.getheader("X-Generation-Id")
                    response_cache_status = response.getheader(
                        "X-OpenRouter-Cache-Status"
                    )
                    if response_cache_status and response_cache_status.upper() == "HIT":
                        proxy.policy_violations.append("response_cache_hit")
                        proxy.revoke()
                        return self.reply(
                            502, "response caching violates benchmark policy"
                        )
                    self.send_response(status)
                    for key, value in response.getheaders():
                        if key.lower() not in HOP:
                            self.send_header(key, value)
                    # Close-delimited HTTP/1.1 preserves SSE without buffering the response.
                    self.send_header("Connection", "close")
                    self.end_headers()
                    sent_headers = True
                    self.close_connection = True
                    while True:
                        chunk = (
                            buffered_error
                            if buffered_error is not None
                            else response.read1(65536)
                        )
                        buffered_error = None
                        if not chunk:
                            # Content-Length bodies must arrive in full; http.client
                            # raises for truncated chunked bodies.
                            framed = not response.length
                            break
                        capture.feed(chunk)
                        receipt = capture.receipt()
                        if receipt != last_receipt and (
                            receipt["generation_id"] or receipt["usage"]
                        ):
                            proxy.record(
                                {
                                    "event": "progress",
                                    "request_id": request_id,
                                    "route": route,
                                    **receipt,
                                }
                            )
                            last_receipt = receipt
                        try:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        except OSError:
                            client_closed = True  # The agent went away.
                            break
                        count += len(chunk)
                except (OSError, http.client.HTTPException):
                    if not sent_headers:
                        self.reply(502, "upstream request failed")
                    self.close_connection = True
                finally:
                    conn.close()
                    if capture:
                        capture.finish()
                    # Complete = full framing and the protocol's terminal event.
                    # An incomplete response the host or agent did not cancel
                    # is an upstream truncation (an infrastructure failure).
                    response_complete = framed and bool(capture and capture.terminal)
                    cancelled = client_closed or proxy.revoked.is_set()
                    proxy.record(
                        {
                            "event": "finished",
                            "request_id": request_id,
                            "route": route,
                            "status": status,
                            "finished_at": utc_now(),
                            "response_bytes": count,
                            "wall_s": round(time.monotonic() - started, 6),
                            "response_complete": response_complete,
                            "cancelled": cancelled and not response_complete,
                            "response_cache_status": response_cache_status
                            if capture
                            else None,
                            **(capture.receipt() if capture else {"usage": None}),
                        }
                    )
                    with proxy.idle:
                        if status == 402 and not temporary_budget:
                            proxy.exhausted = "account"
                        proxy.pending.pop(request_id, None)
                        proxy.idle.notify_all()

            do_GET = do_POST = forward

            def log_message(self, *args):
                pass

        return Handler
