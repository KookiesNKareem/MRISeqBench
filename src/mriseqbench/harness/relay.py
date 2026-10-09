"""Narrow Unix-to-loopback gateways for bubblewrap's isolated network namespace.

This file is also a standalone stdlib-only namespace launcher.
"""

import json
import select
import socket
import socketserver
import subprocess
import sys
import threading
import uuid
from pathlib import Path


def bridge(left, right):
    while True:
        readable, _, _ = select.select([left, right], [], [], 1)
        for source in readable:
            data = source.recv(65536)
            if not data:
                return
            (right if source is left else left).sendall(data)


class QuietServer(socketserver.ThreadingMixIn):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass


class UnixServer(QuietServer, socketserver.UnixStreamServer):
    pass


class TCPServer(QuietServer, socketserver.TCPServer):
    allow_reuse_address = True


class HostRelay:
    def __init__(self, port, directory):
        self.path = Path(directory) / f"bridge-{uuid.uuid4().hex[:8]}.sock"
        # Linux limits Unix socket names to 108 bytes. Use a private short tmp directory if needed.
        self._temp = None
        if len(str(self.path).encode()) >= 104:
            import tempfile

            self._temp = tempfile.TemporaryDirectory(prefix="mriseqbridge-")
            self.path = Path(self._temp.name) / "bridge.sock"

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                with socket.create_connection(
                    ("127.0.0.1", port), timeout=10
                ) as upstream:
                    upstream.settimeout(None)
                    bridge(self.request, upstream)

        self.server = UnixServer(str(self.path), Handler)
        self.path.chmod(0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.path.unlink(missing_ok=True)
        if self._temp:
            self._temp.cleanup()


def namespace_main():
    servers = []
    for port, path in json.loads(sys.argv[1]):

        def handler_for(socket_path):
            class Handler(socketserver.BaseRequestHandler):
                def handle(self):
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
                        upstream.connect(socket_path)
                        bridge(self.request, upstream)

            return Handler

        server = TCPServer(("127.0.0.1", port), handler_for(path))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
    try:
        return subprocess.call(sys.argv[2:])
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    raise SystemExit(namespace_main())
