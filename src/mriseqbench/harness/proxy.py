"""Credential-holding streaming API gateway, adapted from KAB's ApiProxy.

The host alone loads the real key. An agent receives a revocable per-run token
and can reach only configured routes at a fixed upstream HTTPS origin.
"""

import hmac
import http.client
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

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


class ApiProxy:
    def __init__(self, config, log_path, *, model=None, test_key=None):
        self.config = config
        self.model = model
        self.key = test_key if test_key is not None else os.environ.get(config.key_env)
        if not self.key:
            raise ValueError(
                f"API proxy requires host environment variable {config.key_env}"
            )
        self.token = secrets.token_hex(32)
        self.log_path = Path(log_path)
        self.lock = threading.Lock()
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

    def __exit__(self, *args):
        self.revoke()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(2)

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
                if proxy.model and route in ("/chat/completions", "/responses"):
                    try:
                        payload = json.loads(body or b"")
                    except (ValueError, UnicodeError):
                        return self.reply(400, "invalid model request")
                    if (
                        not isinstance(payload, dict)
                        or payload.get("model") != proxy.model
                        or "models" in payload
                    ):
                        return self.reply(
                            403, "request must use the run's configured model"
                        )
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
                sent_headers = False
                try:
                    headers = {
                        k: v for k, v in self.headers.items() if k.lower() not in HOP
                    }
                    headers["Authorization"] = f"Bearer {proxy.key}"
                    headers["Accept-Encoding"] = "identity"
                    conn.request(
                        self.command,
                        upstream.path.rstrip("/") + route,
                        body=body,
                        headers=headers,
                    )
                    response = conn.getresponse()
                    status = response.status
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
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        count += len(chunk)
                except (OSError, http.client.HTTPException):
                    if not sent_headers:
                        self.reply(502, "upstream request failed")
                    self.close_connection = True
                finally:
                    conn.close()
                    with proxy.lock:
                        proxy.requests += 1
                        with proxy.log_path.open("a") as log:
                            log.write(
                                json.dumps(
                                    {
                                        "route": route,
                                        "status": status,
                                        "response_bytes": count,
                                        "wall_s": round(time.monotonic() - started, 6),
                                    }
                                )
                                + "\n"
                            )

            do_GET = do_POST = forward

            def log_message(self, *args):
                pass

        return Handler
