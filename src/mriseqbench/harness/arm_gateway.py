"""Standalone container-local HTTP gateway over the host's narrow file mailbox."""

import base64
import json
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DIRECTORY = Path("/run/mriseqbench")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def handle_request(self):
        target, _, route = self.path.lstrip("/").partition("/")
        if target not in {"feedback", "api", "simulate"}:
            self.send_error(403)
            return
        length = int(self.headers.get("Content-Length", 0))
        if not 0 <= length <= 16777216:
            self.send_error(413)
            return
        name = uuid.uuid4().hex
        request = DIRECTORY / (name + ".request")
        temporary = DIRECTORY / (name + ".tmp")
        response = DIRECTORY / (name + ".response")
        temporary.write_text(
            json.dumps(
                {
                    "target": target,
                    "path": "/" + route,
                    "method": self.command,
                    "token": self.headers.get("X-MRISeqBench-Token", ""),
                    "headers": {
                        k: v
                        for k, v in self.headers.items()
                        if k.lower() == "x-session-id"
                    },
                    "body": base64.b64encode(self.rfile.read(length)).decode(),
                }
            )
        )
        temporary.rename(request)
        deadline = (
            float("inf")
            if target == "simulate" or (target == "feedback" and route == "submit")
            else time.monotonic() + 1800
        )
        headers_sent = False
        try:
            while not response.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("arm gateway timed out")
                time.sleep(0.02)
            with response.open() as stream:
                pending = ""
                while time.monotonic() < deadline:
                    data = stream.readline()
                    if not data or not data.endswith("\n"):
                        pending += data
                        time.sleep(0.02)
                        continue
                    value = json.loads(pending + data)
                    pending = ""
                    if "status" in value:
                        self.send_response(value["status"])
                        self.send_header("Content-Type", value["content_type"])
                        for key, header_value in value.get("headers", {}).items():
                            if (
                                key.lower()
                                in {"x-openrouter-cache-status", "x-generation-id"}
                                and isinstance(header_value, str)
                                and "\r" not in header_value
                                and "\n" not in header_value
                            ):
                                self.send_header(key, header_value)
                        self.end_headers()
                        headers_sent = True
                    if "error" in value:
                        raise ValueError(value["error"])
                    if "data" in value:
                        self.wfile.write(base64.b64decode(value["data"]))
                        self.wfile.flush()
                    if value.get("done"):
                        return
                raise TimeoutError("arm gateway timed out")
        except (OSError, ValueError, TimeoutError):
            if not headers_sent:
                self.send_error(502)
        finally:
            request.unlink(missing_ok=True)
            response.unlink(missing_ok=True)

    do_GET = do_POST = handle_request

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        raise SystemExit(subprocess.call(json.loads(sys.argv[1])))
    finally:
        server.shutdown()
        server.server_close()
