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


class Budget:
    """Campaign spend from provider-reported cost; in-flight requests may overshoot."""

    def __init__(self, cap, spent=0.0):
        self.cap, self.spent, self.lock = cap, spent, threading.Lock()

    def add(self, cost):
        with self.lock:
            self.spent += cost

    def exhausted(self):
        return self.cap is not None and self.spent >= self.cap


class ApiProxy:
    def __init__(
        self,
        config,
        log_path,
        *,
        model=None,
        test_key=None,
        stop_when=None,
        budget=None,
        pause=None,
    ):
        self.config = config
        self.budget, self.pause = budget, pause
        self.cost, self.exhausted = 0.0, None
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
        self.requests = 0
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
            sockets = [
                request["socket"] or request["connection"].sock
                for request in self.pending.values()
            ]
        for upstream_socket in sockets:
            self.cancel_socket(upstream_socket)

    def record(self, row):
        with self.lock, self.log_path.open("a") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")

    def begin(self, request_id, conn, route, effective_request=None):
        with self.lock:
            # Pi may start another generation immediately after a terminal tool
            # reply, before the process supervisor next polls its stop event.
            if not self.active or (
                self.stop_when is not None and self.stop_when.is_set()
            ):
                return False
            if self.budget is not None and self.budget.exhausted():
                self.exhausted = "campaign"
            elif (
                self.config.max_trial_usd is not None
                and self.cost >= self.config.max_trial_usd
            ):
                self.exhausted = self.exhausted or "trial"
            if self.exhausted:
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
            active = self.active
        if not active:
            self.cancel_socket(upstream_socket)

    @staticmethod
    def cancel_socket(upstream_socket):
        if upstream_socket is not None:
            try:
                upstream_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def __exit__(self, *args):
        self.revoke()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(2)
        with self.idle:
            self.idle.wait_for(lambda: not self.pending, timeout=2)

    def _handler(proxy):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

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
                    return self.reply(403, "route not allowed")
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
                    if proxy.model and (
                        not isinstance(payload, dict)
                        or payload.get("model") != proxy.model
                        or "models" in payload
                    ):
                        return self.reply(
                            403, "request must use the run's configured model"
                        )
                    if not isinstance(payload, dict) or any(
                        k in payload for k in ("models", "preset", "route")
                    ):
                        return self.reply(403, "model routing overrides are prohibited")
                    provider = proxy.config.provider_endpoint
                    if provider:
                        payload["provider"] = {
                            "only": [provider],
                            "order": [provider],
                            "allow_fallbacks": False,
                            "require_parameters": True,
                        }
                    payload.update(
                        temperature=proxy.config.temperature,
                        top_p=proxy.config.top_p,
                        session_id=proxy.session_id,
                    )
                    payload.pop("reasoning_effort", None)
                    payload["reasoning"] = {"effort": proxy.config.reasoning_effort}
                    payload.pop("max_tokens", None)
                    payload.pop("max_completion_tokens", None)
                    payload.pop("max_output_tokens", None)
                    payload[
                        "max_output_tokens" if route == "/responses" else "max_tokens"
                    ] = proxy.config.max_output_tokens
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
                response_complete = False
                sent_headers = False
                try:
                    headers = {
                        k: v for k, v in self.headers.items() if k.lower() not in HOP
                    }
                    headers["Authorization"] = f"Bearer {proxy.key}"
                    headers["Accept-Encoding"] = "identity"
                    for key in list(headers):
                        if (
                            key.lower().startswith("x-openrouter-cache")
                            or key.lower() == "x-session-id"
                        ):
                            del headers[key]
                    headers["X-OpenRouter-Cache"] = "false"
                    headers["x-session-id"] = proxy.session_id
                    for retry in range(5):
                        conn.request(
                            self.command,
                            upstream.path.rstrip("/") + route,
                            body=body,
                            headers=headers,
                        )
                        proxy.attach_socket(request_id, conn.sock)
                        response = conn.getresponse()
                        status = response.status
                        if status not in RETRY_STATUS or retry == 4 or not proxy.active:
                            break
                        # Transient provider failures back off here instead of
                        # ending the agent's session; the wait is not charged.
                        try:
                            delay = float(response.getheader("Retry-After", ""))
                        except ValueError:
                            delay = 5 * 2**retry
                        delay = min(max(delay, 1), 60)
                        response.read()
                        conn.close()
                        proxy.record(
                            {
                                "event": "upstream_retry",
                                "request_id": request_id,
                                "route": route,
                                "status": status,
                                "delay_s": delay,
                            }
                        )
                        if proxy.pause:
                            proxy.pause(delay)
                        time.sleep(delay)
                        conn = conn_cls(
                            upstream.hostname,
                            upstream.port,
                            timeout=proxy.config.timeout_s,
                        )
                        with proxy.lock:
                            proxy.pending[request_id]["connection"] = conn
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
                        chunk = response.read1(65536)
                        if not chunk:
                            response_complete = True
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
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        count += len(chunk)
                except (OSError, http.client.HTTPException):
                    if not sent_headers:
                        self.reply(502, "upstream request failed")
                    self.close_connection = True
                finally:
                    conn.close()
                    if capture:
                        capture.finish()
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
                            "response_cache_status": response.getheader(
                                "X-OpenRouter-Cache-Status"
                            )
                            if capture
                            else None,
                            **(capture.receipt() if capture else {"usage": None}),
                        }
                    )
                    cost = ((capture.receipt()["usage"] or {}) if capture else {}).get(
                        "cost_credits"
                    )
                    with proxy.idle:
                        if cost:
                            proxy.cost += cost
                            if proxy.budget is not None:
                                proxy.budget.add(cost)
                        if status == 402:
                            proxy.exhausted = "account"
                        proxy.requests += 1
                        proxy.pending.pop(request_id, None)
                        proxy.idle.notify_all()

            do_GET = do_POST = forward

            def log_message(self, *args):
                pass

        return Handler
