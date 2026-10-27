"""Keep a streaming client connected while trusted admission/backoff waits.

Only SSE comments are added. Provider bytes and usage remain unchanged, and
the proxy journal retains the real upstream HTTP status.
"""

import json
import selectors
import socket
import threading

HEARTBEAT_S = 10.0


class QueuedStream:
    def __init__(self, handler, on_disconnect=lambda: None):
        self.handler = handler
        self.on_disconnect = on_disconnect
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.disconnected = threading.Event()
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True
        self.write(b": mriseqbench transport ready\n\n")
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()

    def connected(self):
        if self.disconnected.is_set():
            return False
        try:
            sock = self.handler.connection
            # DefaultSelector uses epoll/kqueue and supports descriptors above
            # select()'s FD_SETSIZE limit during a large author burst.
            with selectors.DefaultSelector() as poller:
                poller.register(sock, selectors.EVENT_READ)
                readable = bool(poller.select(0))
            if readable:
                try:
                    if not sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                        self.disconnect()
                except BlockingIOError:
                    pass  # Another checker consumed readiness; still connected.
        except (OSError, ValueError):
            self.disconnect()
        return not self.disconnected.is_set()

    def disconnect(self):
        if not self.disconnected.is_set():
            self.disconnected.set()
            self.on_disconnect()

    def write(self, data):
        with self.lock:
            try:
                self.handler.wfile.write(data)
                self.handler.wfile.flush()
            except OSError:
                self.disconnect()
                raise

    def error(self, status, message):
        body = json.dumps({"error": {"message": message, "code": status}}).encode()
        self.write(b"data: " + body + b"\n\n")

    def heartbeat(self):
        while not self.stopped.wait(HEARTBEAT_S):
            if not self.connected():
                return
            try:
                self.write(b": mriseqbench transport keepalive\n\n")
            except OSError:
                return

    def stop_keepalives(self):
        # Provider chunks can end inside a JSON string or UTF-8 character.
        # Insert comments only while waiting, before any provider bytes.
        self.stopped.set()
        self.thread.join(timeout=1)

    def close(self):
        self.stop_keepalives()
