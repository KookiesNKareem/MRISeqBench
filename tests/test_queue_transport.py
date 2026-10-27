"""Real local HTTP/SSE tests: queued admission, cancellation and upstream errors."""

import base64
import importlib.util
import json
import os
import resource
import socket
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace

from mriseqbench.harness import credit_gate, queued_stream
from mriseqbench.harness.arm_mailbox import Mailbox
from mriseqbench.harness.credit_gate import CreditGate
from mriseqbench.harness.proxy import ApiProxy, QuietHTTPServer
from mriseqbench.models import ApiProxyConfig


def test_stream_admission_supports_descriptors_above_select_limit(
    tmp_path, monkeypatch
):
    """The production 128-worker controller can hold more than 1024 files."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, 4096), hard))
    descriptors = []
    try:
        while not descriptors or descriptors[-1] < 1100:
            descriptors.append(os.open(os.devnull, os.O_RDONLY))
        setup_gate(monkeypatch, credit_gate.ProviderAdmission())
        with upstream() as (server, calls, body):
            with ApiProxy(
                config(server, timeout=5),
                tmp_path / "high-fd.jsonl",
                test_key="high-fd",
            ) as proxy:
                for _ in range(3):
                    with urllib.request.urlopen(request(proxy), timeout=5) as response:
                        assert body in response.read()
                assert proxy.settle(2)
            assert len(calls) == 3
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_keepalives_never_split_a_provider_json_event(tmp_path, monkeypatch):
    setup_gate(monkeypatch, credit_gate.ProviderAdmission())
    body = b'data: {"choices":[{"delta":{"content":"continuous"}}]}\n\ndata: [DONE]\n\n'

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            for start in range(0, len(body), 7):
                self.wfile.write(body[start : start + 7])
                self.wfile.flush()
                time.sleep(0.015)  # Partial JSON spans several keepalive ticks.

        def log_message(self, *_):
            pass

    server = QuietHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with ApiProxy(
            config(server, timeout=5), tmp_path / "split.jsonl", test_key="split"
        ) as proxy:
            with urllib.request.urlopen(request(proxy), timeout=5) as response:
                received = response.read()
            assert body in received
            assert b"keepalive" not in received[received.index(b"data: ") :]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@contextmanager
def upstream(status=200):
    calls = []
    body = (
        b'data: {"id":"control","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
        b'data: {"id":"control","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"cost":0.0001}}\n\n'
        b"data: [DONE]\n\n"
        if status == 200
        else b'{"error":{"message":"credit control rejection","code":402}}'
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(status)
            self.send_header(
                "Content-Type",
                "text/event-stream" if status == 200 else "application/json",
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = QuietHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, calls, body
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def payload():
    return {
        "model": "transport-control",
        "stream": True,
        "messages": [{"role": "user", "content": "control"}],
        "max_tokens": 8,
    }


def config(server, timeout=0.1):
    return ApiProxyConfig.model_construct(
        upstream_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
        timeout_s=timeout,
        max_output_tokens=8,
    )


def setup_gate(monkeypatch, gate):
    monkeypatch.setenv("MRISEQBENCH_HAIKU_CREDIT_GATE", "1")
    monkeypatch.setattr(credit_gate, "shared_gate", lambda _: gate)
    monkeypatch.setattr(queued_stream, "HEARTBEAT_S", 0.01)


def request(proxy):
    return urllib.request.Request(
        proxy.base_url + "/chat/completions",
        data=json.dumps(payload()).encode(),
        headers={
            "Authorization": "Bearer " + proxy.token,
            "Content-Type": "application/json",
        },
    )


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_mailbox_survives_admission_longer_than_its_socket_timeout(
    tmp_path, monkeypatch
):
    gate = CreditGate(lambda: 10, margin=0)
    assert gate.acquire(10, lambda: True)
    setup_gate(monkeypatch, gate)
    pauses = []

    @contextmanager
    def pause():
        started = time.monotonic()
        yield
        pauses.append(time.monotonic() - started)

    with upstream() as (server, calls, body):
        log = tmp_path / "api.jsonl"
        with ApiProxy(config(server), log, test_key="control", pause=pause) as proxy:
            directory = tmp_path / "mailbox"
            with Mailbox(directory, None, proxy):
                name = "a" * 32
                (directory / (name + ".request")).write_text(
                    json.dumps(
                        {
                            "target": "api",
                            "path": "/chat/completions",
                            "method": "POST",
                            "body": base64.b64encode(
                                json.dumps(payload()).encode()
                            ).decode(),
                        }
                    )
                )
                response = directory / (name + ".response")
                wait_for(
                    lambda: response.exists()
                    and '"status": 200' in response.read_text()
                )
                time.sleep(0.25)  # Exceeds both mailbox and upstream socket timeout.
                assert not calls
                assert '"error"' not in response.read_text()
                gate.release(10, 0)
                wait_for(lambda: '"done": true' in response.read_text())
                frames = [
                    json.loads(line) for line in response.read_text().splitlines()
                ]
                received = b"".join(
                    base64.b64decode(f["data"]) for f in frames if "data" in f
                )
                assert body in received and b"transport keepalive" in received
                assert not any("error" in f for f in frames)
                assert len(calls) == 1 and calls[0]["max_tokens"] == 8
            assert proxy.settle(2)
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        end = next(r for r in rows if r["event"] == "finished")
        assert end["status"] == 200 and end["response_complete"]
        assert end["response_bytes"] == len(body)  # Comments never become model usage.
        assert end["usage"]["cost_credits"] == 0.0001
        assert pauses and pauses[0] >= 0.25


def test_cancelled_queued_client_never_reaches_provider(tmp_path, monkeypatch):
    gate = CreditGate(lambda: 10, margin=0)
    assert gate.acquire(10, lambda: True)
    setup_gate(monkeypatch, gate)
    with upstream() as (server, calls, _body):
        log = tmp_path / "api.jsonl"
        with ApiProxy(config(server), log, test_key="control") as proxy:
            sock = socket.create_connection(("127.0.0.1", proxy.port))
            data = json.dumps(payload()).encode()
            sock.sendall(
                f"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {proxy.token}\r\nContent-Length: {len(data)}\r\n\r\n".encode()
                + data
            )
            assert b"200" in sock.recv(4096)
            sock.close()
            wait_for(
                lambda: log.exists()
                and '"event": "admission_cancelled"' in log.read_text()
            )
            gate.release(10, 0)
            time.sleep(0.05)
            assert calls == []
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert any(
            r["event"] == "admission_cancelled" and r["upstream_sent"] is False
            for r in rows
        )
        assert not any(r["event"] == "started" for r in rows)


def test_early_stream_keeps_real_upstream_http_error_in_journal(tmp_path, monkeypatch):
    setup_gate(monkeypatch, CreditGate(lambda: 10, margin=0))
    with upstream(402) as (server, calls, _body):
        log = tmp_path / "api.jsonl"
        with ApiProxy(config(server), log, test_key="control") as proxy:
            with urllib.request.urlopen(request(proxy), timeout=1) as response:
                data = response.read()
            assert b'"code": 402' in data and b"credit control rejection" in data
            assert proxy.settle(2) and proxy.exhausted == "account"
            assert len(calls) == 1
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert next(r for r in rows if r["event"] == "finished")["status"] == 402


def test_128_concurrent_streaming_clients_survive_bounded_admission(
    tmp_path, monkeypatch
):
    gate = CreditGate(lambda: 10, margin=0)
    assert gate.acquire(10, lambda: True)
    setup_gate(monkeypatch, gate)
    # Real sockets/handlers, a full 128-author burst, no external API calls.
    with (
        upstream() as (server, calls, body),
        ApiProxy(
            config(server, timeout=5), tmp_path / "api.jsonl", test_key="control"
        ) as proxy,
    ):

        def client(_):
            with urllib.request.urlopen(request(proxy), timeout=5) as response:
                return response.read()

        with ThreadPoolExecutor(max_workers=128) as pool:
            futures = [pool.submit(client, i) for i in range(128)]
            wait_for(lambda: len(gate.waiters) >= 64, timeout=10)
            gate.release(10, 0)
            outputs = [f.result(timeout=30) for f in futures]
        assert len(outputs) == len(calls) == 128
        assert all(body in output for output in outputs)
        assert proxy.settle(5)


def test_credit_admission_is_fifo_and_cancelled_head_releases_followers():
    gate = CreditGate(lambda: 10, margin=0)
    assert gate.acquire(10, lambda: True)
    order = []
    cancelled = threading.Event()
    first = threading.Thread(
        target=lambda: gate.acquire(2, lambda: not cancelled.is_set())
    )
    first.start()
    wait_for(lambda: len(gate.waiters) == 1)

    def next_request():
        assert gate.acquire(1, lambda: True)
        order.append("next")

    second = threading.Thread(target=next_request)
    second.start()
    wait_for(lambda: len(gate.waiters) == 2)
    cancelled.set()
    gate.release(10, 0)
    first.join(timeout=2)
    second.join(timeout=2)
    assert not first.is_alive() and not second.is_alive() and order == ["next"]


def test_gateway_api_uses_controller_lifetime_not_a_second_wall_clock(tmp_path):
    path = (
        Path(__file__).resolve().parents[1] / "src/mriseqbench/harness/arm_gateway.py"
    )
    spec = importlib.util.spec_from_file_location("gateway_timeout_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DIRECTORY = tmp_path
    started = time.monotonic()
    # Fast-forward only the gateway's wall clock beyond its old 30-minute limit.
    module.time = SimpleNamespace(
        monotonic=lambda: time.monotonic()
        + (3600 if time.monotonic() - started > 0.05 else 0),
        sleep=time.sleep,
    )
    server = QuietHTTPServer(("127.0.0.1", 0), module.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def response():
        wait_for(lambda: bool(list(tmp_path.glob("*.request"))))
        request_path = next(tmp_path.glob("*.request"))
        time.sleep(0.15)
        request_path.with_suffix(".response").write_text(
            "\n".join(
                json.dumps(frame)
                for frame in [
                    {"status": 200, "content_type": "application/json"},
                    {"data": base64.b64encode(b'{"ok":true}').decode()},
                    {"done": True},
                ]
            )
            + "\n"
        )

    writer = threading.Thread(target=response)
    writer.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/api/chat/completions",
            data=b"{}",
        )
        with urllib.request.urlopen(req, timeout=2) as result:
            assert json.load(result) == {"ok": True}
    finally:
        writer.join(timeout=3)
        server.shutdown()
        server.server_close()
        thread.join()


def test_transient_provider_budget_retries_rejoin_fifo_without_timeout(
    tmp_path, monkeypatch
):
    from mriseqbench.harness import proxy as proxy_module
    from mriseqbench.harness.credit_gate import ProviderAdmission

    calls = []
    gate = ProviderAdmission(maximum=4)
    setup_gate(monkeypatch, gate)
    monkeypatch.setattr(proxy_module, "retry_delay", lambda *_a, **_kw: 0.02)
    error = json.dumps(
        {
            "error": {
                "message": "settling",
                "metadata": {"reason": "in_flight_budget_exhausted"},
            }
        }
    ).encode()
    success = b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"cost":0.01}}\n\ndata: [DONE]\n\n'

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(1)
            status, body = (402, error) if len(calls) <= 6 else (200, success)
            self.send_response(status)
            self.send_header(
                "Content-Type",
                "application/json" if status == 402 else "text/event-stream",
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = QuietHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        log = tmp_path / "api.jsonl"
        with ApiProxy(config(server, timeout=1), log, test_key="control") as proxy:
            with urllib.request.urlopen(request(proxy), timeout=2) as response:
                assert success in response.read()
            assert proxy.settle(3) and proxy.exhausted is None
        assert len(calls) == 7 and gate.inflight == 0 and gate.capacity < 4
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert len([r for r in rows if r["event"] == "upstream_retry"]) == 6
        end = next(r for r in rows if r["event"] == "finished")
        assert end["response_complete"] and end["status"] == 200
        assert end["usage"]["cost_credits"] == 0.01
        assert end["network_wall_s"] < end["wall_s"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
