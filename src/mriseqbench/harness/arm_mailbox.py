"""File-backed transport to fixed host gateways for networkless containers.

Only feedback, the model proxy and the selected simulator are addressable.
Responses are streamed as JSONL frames, including model SSE responses.
"""

import base64
import json
import os
import re
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NAME = re.compile(r"[a-f0-9]{32}\.request\Z")
LIMIT = 24 * 1024 * 1024


def read_request(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        import stat

        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > LIMIT:
            raise ValueError("invalid mailbox request")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            return json.loads(stream.read(LIMIT + 1))
    finally:
        os.close(fd)


class Mailbox:
    def __init__(self, directory, feedback, proxy=None, simulator=None):
        self.directory = Path(directory)
        self.feedback, self.proxy, self.simulator = feedback, proxy, simulator
        self.stop = threading.Event()
        self.seen = set()
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(2)
        self.pool.shutdown(wait=True, cancel_futures=True)

    def watch(self):
        while not self.stop.wait(0.02):
            for path in self.directory.glob("*.request"):
                if NAME.fullmatch(path.name) and path.name not in self.seen:
                    self.seen.add(path.name)
                    self.pool.submit(self.handle, path)

    def handle(self, path):
        response_path = path.with_suffix(".response")
        try:
            fd = os.open(
                response_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
        except OSError:
            return
        with os.fdopen(fd, "w") as output:

            def frame(value):
                output.write(json.dumps(value) + "\n")
                output.flush()

            def result(status, value):
                frame({"status": status, "content_type": "application/json"})
                frame({"data": base64.b64encode(json.dumps(value).encode()).decode()})
                frame({"done": True})

            try:
                request = read_request(path)
                if not isinstance(request, dict):
                    raise TypeError("mailbox request must be an object")
                body = base64.b64decode(request.get("body", ""), validate=True)
                if len(body) > 16777216:
                    raise ValueError("mailbox body exceeds 16 MiB")
                target, route = request.get("target"), request.get("path")
                if (
                    target == "feedback"
                    and route in {"/lint", "/check", "/submit"}
                    and request.get("method") == "POST"
                ):
                    status, value = self.feedback.handle(
                        route[1:], request.get("token", ""), body
                    )
                    result(status, value)
                elif (
                    target == "simulate"
                    and route == "/simulate"
                    and request.get("method") == "POST"
                ):
                    if self.simulator is None:
                        result(403, {"error": "simulation is disabled for this arm"})
                    else:
                        status, value = self.simulator(body)
                        result(status, value)
                elif (
                    target == "api"
                    and self.proxy
                    and route in {"/chat/completions", "/responses", "/models"}
                    and request.get("method") in {"GET", "POST"}
                ):
                    if request["method"] == "POST":
                        payload = json.loads(body)
                        if not isinstance(payload, dict):
                            raise TypeError("model request must be an object")
                        # Preserve normal agent function tools, but do not permit
                        # provider-side web/code/MCP tools to escape arm policy.
                        if any(
                            k in payload
                            for k in ("plugins", "web_search_options", "mcp_servers")
                        ) or any(
                            not isinstance(tool, dict) or tool.get("type") != "function"
                            for tool in payload.get("tools", [])
                        ):
                            result(
                                403,
                                {
                                    "error": "remote built-in tools are disabled for restricted arms"
                                },
                            )
                            return
                    req = urllib.request.Request(
                        self.proxy.base_url + route,
                        data=body if request["method"] == "POST" else None,
                        method=request["method"],
                        headers={
                            "Authorization": "Bearer " + self.proxy.token,
                            "Content-Type": "application/json",
                        },
                    )
                    try:
                        upstream = urllib.request.urlopen(
                            req, timeout=self.proxy.config.timeout_s
                        )
                    except urllib.error.HTTPError as exc:
                        upstream = exc
                    with upstream:
                        frame(
                            {
                                "status": upstream.status,
                                "content_type": upstream.headers.get(
                                    "Content-Type", "application/json"
                                ),
                            }
                        )
                        while True:
                            chunk = upstream.read1(65536)
                            if not chunk:
                                break
                            frame({"data": base64.b64encode(chunk).decode()})
                        frame({"done": True})
                else:
                    result(403, {"error": "route is not available in this arm"})
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                frame({"error": f"{type(exc).__name__}: {exc}", "done": True})
